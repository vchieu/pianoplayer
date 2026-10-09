"""
Render a demo soundtrack with the built-in synth, offline (no audio device).

    python make_demo.py                     # fixed excerpt  -> demo_after.wav
    python make_demo.py out.wav             # same excerpt, other file name
    python make_demo.py scores/bach_invention4.xml demo_score.wav 16

The fixed excerpt is the one used to compare the sound before/after a change
of the synthesis model; the score mode runs exactly the same code path as the
3D player (same tempo, same velocities, same note releases).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pianoplayer.audio import SoundEngine

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_audio_out")
os.makedirs(OUT, exist_ok=True)


def demo_events():
    """Simple excerpt: Alberti bass + melody + a closing chord."""
    ev = []
    t = 0.0
    step = 0.30                     # eighth notes at ~100 bpm

    bass = [48, 55, 52, 55, 45, 52, 48, 52,       # C major
            41, 48, 45, 48, 43, 50, 47, 50]       # F / G7
    for i, m in enumerate(bass):
        ti = i * step * 2
        ev.append((ti, 'on', m, 0.55, -0.35))
        ev.append((ti + step * 1.6, 'off', m, 0.0, -0.35))

    mel = [72, 74, 76, 77, 79, 77, 76, 74,
           72, 76, 79, 81, 79, 77, 76, 74]
    for i, m in enumerate(mel):
        ti = 0.3 + i * step
        vel = 0.45 + 0.25 * ((i % 4) == 0) + 0.15 * (i % 2)
        ev.append((ti, 'on', m, min(vel, 1.0), 0.35))
        ev.append((ti + step * 0.9, 'off', m, 0.0, 0.35))

    for m in (36, 48, 60, 64, 67, 72):
        pan = -0.35 if m < 60 else 0.35
        ev.append((5.6, 'on', m, 0.75, pan))
        ev.append((7.4, 'off', m, 0.0, pan))
    return ev, max(e[0] for e in ev) + 2.0


def score_events(fn, nmeasures=16):
    """Same event list as the 3D player builds (tempo + dynamics included)."""
    from music21 import converter
    from pianoplayer.hand import Hand
    from pianoplayer.scorereader import reader
    from pianoplayer.utils import score_tempo
    from pianoplayer.vkeyboard import VirtualKeyboard

    sf = converter.parse(fn)
    vk = VirtualKeyboard(songname=fn, tempo=score_tempo(sf), offscreen=True)
    vk.verbose = False

    ev, end = [], 1.0
    for beam, sign, pan in ((0, +1.0, +0.30), (1, -1.0, -0.30)):
        h = Hand("right" if sign > 0 else "left", "M")
        h.verbose = False
        h.noteseq = reader(sf, beam=beam)
        if not h.noteseq:
            continue
        h.generate(1, nmeasures)
        for e in vk._events_for(h, sign, pan):
            if e.midi is None:
                continue
            ev.append((e.t_on, 'on', e.midi, e.vel, pan))
            ev.append((e.t_off, 'off', e.midi, 0.0, pan))
            end = max(end, e.t_off)
    return ev, end + 1.5


def main():
    args = sys.argv[1:]
    if args and args[0].lower().endswith((".xml", ".mxl", ".mid", ".mscz")):
        score = args[0]
        path = args[1] if len(args) > 1 else os.path.join(OUT, "demo_score.wav")
        nmeas = int(args[2]) if len(args) > 2 else 16
        events, end = score_events(score, nmeas)
        print("score:", score, "events:", len(events))
    else:
        path = args[0] if args else os.path.join(OUT, "demo_after.wav")
        events, end = demo_events()

    eng = SoundEngine(volume=0.8, backend="none")
    eng.export_wav(events, end, path)
    mix = eng.render(events, end)
    import numpy as np
    vels = [e[3] for e in events if e[1] == 'on']
    print("wrote %s  (%.1f s, peak %.3f, rms %.4f, velocity %.2f..%.2f)"
          % (path, end, float(np.abs(mix).max()),
             float(np.sqrt((mix ** 2).mean())), min(vels), max(vels)))


if __name__ == "__main__":
    main()
