"""
Offline tests for the new piano sound engine (no audio device needed).

Run:  python test_audio.py
"""
import os
import time
import wave

import numpy as np

from pianoplayer.audio import SoundEngine, midi_of, midi_from_name, frequency

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_audio_out")
os.makedirs(OUT, exist_ok=True)


def stereo_energy(mix, t0, t1, sr=44100):
    a, b = int(t0 * sr), int(t1 * sr)
    seg = mix[a:b]
    if not len(seg):
        return 0.0, 0.0
    return (float(np.sqrt(np.mean(seg[:, 0] ** 2))),
            float(np.sqrt(np.mean(seg[:, 1] ** 2))))


def dominant_freqs(mix, t0, t1, top=6, sr=44100):
    """Return the strongest frequencies present in the given time window."""
    seg = mix[int(t0 * sr):int(t1 * sr), 0]
    if len(seg) < 64:
        return []
    w = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), 1.0 / sr)
    spec[:8] = 0                              # ignore DC and infrasound
    order = np.argsort(spec)[::-1][:top]
    out = []
    for i in sorted(order, key=lambda i: freqs[i]):
        # parabolic interpolation for a sub-bin estimate
        if 0 < i < len(spec) - 1:
            a, b, c = spec[i - 1], spec[i], spec[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        else:
            d = 0.0
        out.append(float(freqs[i] + d * (freqs[1] - freqs[0])))
    return out


def test_chord_and_two_hands():
    """A chord must contain all its notes, and the two hands must overlap."""
    eng = SoundEngine(volume=1.0, backend="none")
    ev = []
    # right hand chord (C4 E4 G4) from 0 to 1 s
    for m in (60, 64, 67):
        ev.append((0.0, 'on', m, 0.7, 0.35))
        ev.append((1.0, 'off', m, 0.0, 0.35))
    # left hand bass C3 from 0 to 2 s, and G2 from 0.4 to 1.4 s
    ev.append((0.0, 'on', 48, 0.8, -0.35))
    ev.append((2.0, 'off', 48, 0.0, -0.35))
    ev.append((0.4, 'on', 43, 0.75, -0.35))
    ev.append((1.4, 'off', 43, 0.0, -0.35))

    mix = eng.render(ev, 3.0)
    path = eng.export_wav(ev, 3.0, os.path.join(OUT, "chord_two_hands.wav"))

    assert mix.shape[0] == 3 * 44100, mix.shape
    peak = float(np.max(np.abs(mix)))
    assert 0.05 < peak <= 1.0, f"peak={peak}"

    # 1) all four notes are really sounding at the same time (spectrum)
    found = dominant_freqs(mix, 0.05, 0.40, top=8)
    for expected in (130.81, 261.63, 329.63, 392.00):     # C3 C4 E4 G4
        err = min(abs(f - expected) for f in found)
        assert err < 8.0, f"missing ~{expected} Hz, closest={err:.1f} of {found}"
    # 2) the second left-hand note (0.4 s) sounds while the RH chord still rings
    found2 = dominant_freqs(mix, 0.45, 0.9, top=8)
    assert any(abs(f - 98.0) < 6.0 for f in found2), f"G2 missing: {found2}"
    # 3) everything is damped after the last release
    e_tail, _ = stereo_energy(mix, 2.6, 2.99)
    e_all, _ = stereo_energy(mix, 0.05, 0.9)
    assert e_tail < 0.35 * max(e_all, 1e-6), f"note not damped: {e_tail} vs {e_all}"

    print(f"  ok  chord+2 hands, peak={peak:.3f}, partials={[round(f,1) for f in found]}")
    print(f"      wrote {path}")
    return path


def test_stereo_panning():
    """Left hand must be louder on the left channel, and vice versa."""
    eng = SoundEngine(volume=1.0, backend="none")
    lh = eng.render([(0.0, 'on', 48, 0.7, -0.35), (0.8, 'off', 48, 0.0, -0.35)], 1.2)
    rl, rr = stereo_energy(lh, 0.05, 0.8)
    assert rl > rr * 1.05, f"LH not panned left: {rl} vs {rr}"

    rh = eng.render([(0.0, 'on', 72, 0.7, 0.35), (0.8, 'off', 72, 0.0, 0.35)], 1.2)
    hl, hr = stereo_energy(rh, 0.05, 0.8)
    assert hr > hl * 1.05, f"RH not panned right: {hl} vs {hr}"
    print(f"  ok  panning: LH {rl:.3f}/{rr:.3f}, RH {hl:.3f}/{hr:.3f}")


def test_timing_accuracy():
    """A note scheduled at 1.0 s must not be heard before 1.0 s."""
    eng = SoundEngine(volume=1.0, backend="none")
    ev = [(1.0, 'on', 72, 0.8, 0.0), (1.5, 'off', 72, 0.0, 0.0)]
    mix = eng.render(ev, 2.0)
    before_l, _ = stereo_energy(mix, 0.0, 0.95)
    after_l, _ = stereo_energy(mix, 1.02, 1.4)
    assert before_l < 1e-4, f"early sound: {before_l}"
    assert after_l > 0.01, f"late/missing sound: {after_l}"
    print(f"  ok  timing: silence before onset={before_l:.6f}, "
          f"after onset RMS={after_l:.4f}")


def test_polyphony_and_speed():
    """Rendering 10 s of a fast passage must be much faster than real time."""
    eng = SoundEngine(volume=1.0, backend="none")
    ev = []
    t = 0.0
    mids = [60, 62, 64, 65, 67, 69, 71, 72]
    for i in range(160):                       # 160 sixteenths ~ 10 s
        m = mids[i % len(mids)]
        ev.append((t, 'on', m, 0.5 + 0.3 * (i % 3) / 2.0, 0.2))
        ev.append((t + 0.06, 'off', m, 0.0, 0.2))
        t += 0.0625
    t0 = time.time()
    mix = eng.render(ev, t + 0.5)
    dt = time.time() - t0
    real = t + 0.5
    peak = float(np.max(np.abs(mix)))
    assert peak <= 1.0
    assert dt < real, f"rendering too slow: {dt:.2f}s for {real:.2f}s of audio"
    print(f"  ok  10 s passage rendered in {dt:.2f} s "
          f"({real / dt:.1f}x real time), peak={peak:.3f}")
    eng.export_wav(ev, t + 0.5, os.path.join(OUT, "fast_passage.wav"))


def test_helpers():
    assert midi_from_name("C", 4) == 60
    assert midi_from_name("A", 4) == 69
    assert midi_from_name("C#", 4) == 61
    assert abs(frequency(69) - 440.0) < 1e-9
    from music21.note import Note
    assert midi_of(Note("C5")) == 72
    print("  ok  midi helpers")


def test_read_score():
    """Build events from a real score, exactly like the 3D player does."""
    from music21 import converter
    from pianoplayer.scorereader import reader
    from pianoplayer.hand import Hand
    from pianoplayer.audio import midi_of

    here = os.path.dirname(os.path.abspath(__file__))
    sf = converter.parse(os.path.join(here, "scores", "test_chords.xml"))

    events = []
    for beam, pan in ((0, 0.35), (1, -0.35)):
        h = Hand("right" if beam == 0 else "left")
        h.verbose = False
        h.noteseq = reader(sf, beam=beam)
        if not h.noteseq:
            continue
        h.generate(nmeasures=4)
        for n in h.noteseq:
            events.append((n.time, 'on', midi_of(n), 0.65, pan))
            events.append((max(n.time + n.duration, n.time + 0.05),
                           'off', midi_of(n), 0.0, pan))

    if not events:
        print("  ok  score has a single beam only (skipped)")
        return
    eng = SoundEngine(volume=1.0, backend="none")
    end = max(e[0] for e in events) + 1.5
    mix = eng.render(events, end)
    peak = float(np.max(np.abs(mix)))
    assert 0.02 < peak <= 1.0, f"peak={peak}"
    path = eng.export_wav(events, end, os.path.join(OUT, "test_chords_score.wav"))
    print(f"  ok  score rendered ({len(events)} events, {end:.1f}s), "
          f"peak={peak:.3f} -> {path}")


if __name__ == "__main__":
    print("audio engine tests:")
    test_helpers()
    test_timing_accuracy()
    test_chord_and_two_hands()
    test_stereo_panning()
    test_polyphony_and_speed()
    test_read_score()
    print("all audio tests passed.")
