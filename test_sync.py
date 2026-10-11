"""
Audio/visual sync tests (Phase A3 of PIANOPLAYER_FIX_PLAN.md).

Uses a FakeEngine like test_regression.py::test_audio_epoch_tracks_latency,
with a fake engine_wall and output latency. For each note:

    heard  = engine_wall + when + latency   (when = engine time handed to note_on)
    visual = vk._t0 + e.t_on                (score time at which _press fires)

Acceptance: |heard - visual| < 0.02 s for every note — including notes
scheduled before a mid-playback output_latency jump (F10: the jump must be
compensated for notes scheduled after it; notes already inside the 1 s
AUDIO_LOOKAHEAD window cannot be retro-fixed and will show the gap).

NOTE: the latency-jump case is expected to FAIL until Phase C (C2/C3/C7).
Keep the assertion strict — do NOT weaken it; record deltas in Deviations.

    python test_sync.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import note, stream, tempo

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.vkeyboard import AUDIO_LOOKAHEAD, LEAD_IN, VirtualKeyboard

FAILS = []

SYNC_TOL = 0.02     # s
LAT1 = 0.10         # initial fake output latency [s]
LAT2 = 0.35         # jumped fake output latency [s] (250 ms underrun padding)
WALL = 100.0        # fake engine wall time [s]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class FakeEngine:
    underruns = 0

    def __init__(self):
        self.output_latency = LAT1
        self.sent = []          # (midi, when) in schedule order

    def pump(self, *a, **k):
        return 0

    def note_on(self, midi, vel, when=None, pan=0.0):
        self.sent.append((midi, when))

    def note_off(self, midi, when=None):
        pass


def make_score(n=16, ql=0.5, hold=0.5, bpm=120):
    pitches = [60, 64, 67, 72, 67, 64, 60, 64, 67, 72, 76, 72, 67, 64, 60, 55]
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    for i in range(n):
        p.insert(i * ql, note.Note(pitches[i % len(pitches)],
                                   quarterLength=hold))
    s = stream.Score()
    s.insert(0, p)
    return s


def make_vk(n=16):
    h = Hand("right", "M")
    h.verbose = False
    h.noteseq = reader(make_score(n), beam=0)
    h.generate(1, 1000)
    vk = VirtualKeyboard(songname="synctest", tempo=120.0, offscreen=True)
    vk.verbose = False
    vk.playsounds = False
    vk.build_RH(h)
    vk._prepare()
    eng = FakeEngine()
    vk.engine = eng
    vk._engine_wall = WALL
    vk._t0 = WALL + LEAD_IN
    vk._latency = LAT1
    vk._audio_epoch = WALL + LAT1 - vk._t0
    return vk, eng


def run_playback(vk, eng, jump_at=None):
    """Drive vk._frame() over the whole piece; jump latency at jump_at (s)."""
    total = vk._total + 0.2
    now = -LEAD_IN
    dt = 1.0 / 60.0
    jumped = False
    while now < total:
        if jump_at is not None and not jumped and now >= jump_at:
            eng.output_latency = LAT2      # an underrun pads 250 ms of silence
            jumped = True
        vk._frame(now, dt)
        now += dt
    vk.engine = None
    check(jumped or jump_at is None, "jump never triggered")
    return eng.sent


def deltas(vk, sent, lat_before, lat_after, jump_at):
    """Per-note |heard - visual|; notes heard after the jump use lat_after."""
    notes = [e for p in vk._players.values() for e in p.notes]
    with_midi = [e for e in notes if e.midi is not None]
    check(len(sent) == len(with_midi),
          "engine got %d note_ons for %d midi notes" % (len(sent), len(with_midi)))
    out = []
    for (midi, when), e in zip(sent, with_midi):
        visual = vk._t0 + e.t_on
        lat = lat_after if (jump_at is not None and e.t_on >= jump_at) else lat_before
        heard = vk._engine_wall + when + lat
        out.append((e.t_on, midi, abs(heard - visual)))
    return out


def report(d, label):
    worst = max(x[2] for x in d)
    n_bad = sum(1 for x in d if x[2] >= SYNC_TOL)
    print("   %s: %d notes, worst |heard-visual| = %.4f s, %d over %.2f s"
          % (label, len(d), worst, n_bad, SYNC_TOL))
    for t_on, midi, dv in d:
        if dv >= SYNC_TOL:
            print("   note t_on=%.3f midi=%s delta=%+.4f s" % (t_on, midi, dv))
    return worst, n_bad


def test_sync_constant_latency():
    """No jump: every note must land within 20 ms."""
    vk, eng = make_vk()
    sent = run_playback(vk, eng)
    d = deltas(vk, sent, LAT1, LAT1, None)
    worst, n_bad = report(d, "constant latency")
    check(n_bad == 0, "%d/%d notes off by >= %.2f s (worst %.4f s)"
          % (n_bad, len(d), SYNC_TOL, worst))


def test_sync_latency_jump():
    """Latency jumps mid-playback: notes scheduled after it must be
    compensated (|heard-visual| < 0.02 s for ALL notes)."""
    vk, eng = make_vk()
    jump_at = vk._total / 2.0
    sent = run_playback(vk, eng, jump_at=jump_at)
    d = deltas(vk, sent, LAT1, LAT2, jump_at)
    worst, n_bad = report(d, "latency %.2f->%.2f at t=%.2f (lookahead %.1f s)"
                          % (LAT1, LAT2, jump_at, AUDIO_LOOKAHEAD))
    check(n_bad == 0, "%d/%d notes off by >= %.2f s after latency jump "
          "(worst %.4f s; F10/C7)" % (n_bad, len(d), SYNC_TOL, worst))


# --------------------------------------------------------------------------- #
def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception:
            FAILS.append(t.__name__)
            print("FAIL", t.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(tests) - len(FAILS), len(tests)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
