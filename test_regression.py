"""
Regression tests for finger / key / sound / fingering consistency.
Offscreen, no audio device needed.

    python test_regression.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import chord, converter, duration, note, pitch, stream, tempo, tie
from music21.articulations import Fingering

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import keypos, score_tempo, score_tempo_map
from pianoplayer.vkeyboard import LEAD_IN, VirtualKeyboard, _key_x

FAILS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_score(pitches, ql=0.25, hold=1.0, bpm=120):
    """One part; `ql` between onsets, each note held `hold` quarters."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    for i, m in enumerate(pitches):
        p.insert(i * ql, note.Note(m, quarterLength=hold))
    s = stream.Score()
    s.insert(0, p)
    return s


def make_hand(sf, side="right", start=1, nmeas=1000):
    h = Hand(side, "M")
    h.verbose = False
    h.noteseq = reader(sf, beam=0)
    h.generate(start, nmeas)
    return h


def make_vk(hands, tempo_bpm=120.0):
    # hands="3d": this test reads the rig's finger targets (the 2D backend is
    # covered by test_hands2d.py)
    vk = VirtualKeyboard(songname="regress", tempo=tempo_bpm, offscreen=True,
                         hands="3d")
    vk.verbose = False
    vk.playsounds = False
    for h, side in hands:
        (vk.build_RH if side == "right" else vk.build_LH)(h)
    return vk


FAST = [60, 64, 67, 72, 67, 64, 60, 64, 67, 72, 76, 72, 67, 64, 60, 55,
        59, 62, 67, 71, 67, 62, 59, 62, 67, 71, 74, 71, 67, 62, 59, 55]


# --------------------------------------------------------------------------- #
def test_finger_stays_on_held_key():
    """Bug 1: a finger holding a key must target exactly that key's x, even
    while the hand is already aiming at a later note."""
    sf = make_score(FAST, ql=0.25, hold=1.0)       # 0.125 s per note @120 bpm
    h = make_hand(sf)
    vk = make_vk([(h, "right")])
    vk._prepare()

    dt, now = 1.0 / 60.0, -LEAD_IN
    total = vk._total + 0.1
    ahead = checked = 0
    while now < total:
        vk._frame(now, dt)
        for p in vk._players.values():
            last = {}
            for e in p.held:                      # newest wins, as in _pump
                if e.finger and e.wx is not None:
                    last[e.finger] = e
            for f, e in last.items():
                checked += 1
                if p.aim > p.fire - 1:
                    ahead += 1
                check(abs(p.finger_x[f] - e.wx) < 1e-9,
                      "t=%.3f finger %d targets x=%.2f but holds key x=%.2f"
                      % (now, f, p.finger_x[f], e.wx))
                check(abs(p.rig._targets_x[f] - e.wx) < 1e-9,
                      "rig target of finger %d differs from the held key" % f)
        now += dt
    check(checked > 50, "scenario too weak: only %d held-finger checks" % checked)
    check(ahead > 10, "test never had the hand aiming ahead (%d)" % ahead)


def test_every_note_has_a_finger():
    """Bug 2: the last note (and short scores) must get a finger 1..5."""
    for n in (32, 12, 9, 5, 2):
        sf = make_score(FAST[:n], ql=0.5, hold=0.5)
        h = make_hand(sf)
        bad = [i for i, x in enumerate(h.noteseq) if x.fingering not in (1, 2, 3, 4, 5)]
        check(not bad, "N=%d: notes without finger at %s" % (n, bad))
        check(len(h.fingerseq) == n, "N=%d: %d fans" % (n, len(h.fingerseq)))


def test_scan_range_limits_playback():
    """Bug 3: -n N must not play notes outside the scanned range."""
    fn = os.path.join(HERE, "scores", "bach_invention4.xml")
    if not os.path.isfile(fn):
        print("   (skipped: scores/bach_invention4.xml not found)")
        return
    sf = converter.parse(fn)
    h = make_hand(sf, nmeas=2)
    check(0 < len(h.fingerseq_index) < len(h.noteseq), "range did not limit notes")
    vk = make_vk([])
    evs = vk._events_for(h, 1.0, 0.3)
    check(len(evs) == len(h.fingerseq_index),
          "%d events for %d scanned notes" % (len(evs), len(h.fingerseq_index)))
    check(all(e.finger in (1, 2, 3, 4, 5) for e in evs), "event without finger")


def test_enharmonic_key_sound_finger_agree():
    """Bug 4: x used by the algorithm == physical key of the sounding midi."""
    check(abs(keypos(note.Note("B#3")) - keypos(note.Note("C4"))) < 1e-9, "B#3")
    check(abs(keypos(note.Note("C-4")) - keypos(note.Note("B3"))) < 1e-9, "Cb4")
    check(abs(keypos(pitch.Pitch("E#4")) - keypos(pitch.Pitch("F4"))) < 1e-9, "E#4")

    names = ["B#3", "C-4", "E#4", "F-4", "C##4", "D--4", "B3", "C4", "E4",
             "G4", "A#4", "B-4"]
    sf = make_score(names, ql=0.5, hold=0.5)
    for side, sign in (("right", 1.0), ("left", -1.0)):
        h = make_hand(sf, side)
        vk = make_vk([])
        evs = vk._events_for(h, sign, 0.0)
        check(len(evs) == len(names), "%s: %d events" % (side, len(evs)))
        for e in evs:
            check(abs(e.wx - _key_x(e.midi)) < 1e-6,
                  "%s: %s midi %d: finger x %.2f != key x %.2f"
                  % (side, e.name, e.midi, e.wx, _key_x(e.midi)))
            check(vk._key_for(e) in vk.KB, "no 3D key for midi %d" % e.midi)


def test_chord_uses_true_onset():
    """Bug 5: chord notes sound together at the true onset."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    p.insert(0, chord.Chord(["C4", "E4", "G4", "C5"], quarterLength=1))
    for i in range(12):
        p.insert(1 + i * 0.5, note.Note(60 + (i % 5) * 2, quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    h = make_hand(s)
    vk = make_vk([], 120.0)
    evs = vk._events_for(h, 1.0, 0.0)
    first = [e for e in evs if e.t_on < 0.01]
    check(len(first) == 4, "chord notes at onset 0: %d" % len(first))
    check(all(abs(e.t_off - 0.5) < 1e-6 for e in first),
          "chord duration should be 1 quarter = 0.5 s: %s"
          % [round(e.t_off, 3) for e in first])


def test_tie_and_simultaneous_voices():
    """Bugs 7 and 9: ties extend the first note; simultaneous notes are kept."""
    p = stream.Part()
    a = note.Note("C4", quarterLength=1); a.tie = tie.Tie("start")
    b = note.Note("C4", quarterLength=1); b.tie = tie.Tie("stop")
    p.insert(0, a)
    p.insert(1, b)
    p.insert(0, note.Note("E4", quarterLength=2))      # same onset as `a`
    p.insert(2, note.Note("D4", quarterLength=1))
    s = stream.Score()
    s.insert(0, p)
    seq = reader(s, 0)
    got = sorted((x.name, x.time_real, x.dur_real) for x in seq)
    check(("C", 0.0, 2.0) in got, "tie not merged: %s" % got)
    check(("E", 0.0, 2.0) in got, "simultaneous voice dropped: %s" % got)
    check(len(seq) == 3, "unexpected notes: %s" % got)


def test_tempo_referent_and_map():
    """Bug 8: referent honoured, tempo changes integrated piecewise."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120,
                                    referent=duration.Duration("eighth")))
    p.insert(2, tempo.MetronomeMark(number=60))
    p.insert(0, note.Note("C4", quarterLength=1))
    s = stream.Score()
    s.insert(0, p)
    check(abs(score_tempo(s) - 60.0) < 1e-6, "eighth=120 should be 60 qbpm")
    check(score_tempo_map(s) == [(0.0, 60.0), (2.0, 60.0)], score_tempo_map(s))

    vk = make_vk([])
    vk.tempo_map = [(0.0, 60.0), (2.0, 120.0)]
    check(abs(vk._beat_seconds(2.0) - 2.0) < 1e-9, "first segment")
    check(abs(vk._beat_seconds(4.0) - 3.0) < 1e-9, "second segment")
    vk.speedfactor = 2.0
    check(abs(vk._beat_seconds(4.0) - 1.5) < 1e-9, "speedfactor")


def test_audio_epoch_tracks_latency():
    """Bug 6: epoch follows a growing output latency (fake engine)."""
    class FakeEngine:
        underruns = 0
        def __init__(self): self.output_latency = 0.1; self.sent = []
        def pump(self, *a, **k): return 0
        def note_on(self, midi, vel, when=None, pan=0.0): self.sent.append(when)
        def note_off(self, midi, when=None): pass

    sf = make_score(FAST, ql=0.5, hold=0.5)
    vk = make_vk([(make_hand(sf), "right")])
    vk._prepare()
    eng = FakeEngine()
    vk.engine = eng
    vk._engine_wall = 100.0
    vk._t0 = 100.0 + LEAD_IN
    vk._latency = 0.1
    vk._audio_epoch = 100.0 + 0.1 - vk._t0
    before = vk._audio_epoch
    eng.output_latency = 0.35                     # an underrun padded 250 ms
    vk._frame(0.0, 1.0 / 60.0)
    check(abs((vk._audio_epoch - before) - 0.25) < 1e-9,
          "epoch moved by %.3f, expected 0.25" % (vk._audio_epoch - before))
    vk.engine = None


def test_left_mirror_idempotent():
    """generate() called twice on the same noteseq must keep
    x mirrored, give the same fingering and not double the
    Fingering annotation."""
    # slow notes: the first run picks its window depth with
    # autodepth, so this proves a second run reproduces it
    names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"]
    sf = make_score(names, ql=1.0, hold=1.0, bpm=60)
    h = make_hand(sf, "left")
    x1 = [n.x for n in h.noteseq]
    f1 = [n.fingering for n in h.noteseq]
    check(all(v < 0 for v in x1), "left hand x not mirrored: %s" % x1[:3])
    h.generate()
    x2 = [n.x for n in h.noteseq]
    f2 = [n.fingering for n in h.noteseq]
    check(x1 == x2, "generate() flipped x twice: %s -> %s" % (x1[:3], x2[:3]))
    check(f1 == f2, "fingering changed on 2nd run: %s -> %s" % (f1, f2))
    check(all(len(n.note21.articulations) == 1 for n in h.noteseq),
          "Fingering annotation doubled")
    vk = make_vk([])
    evs = vk._events_for(h, -1.0, 0.0)
    check(len(evs) == len(names), "%d events" % len(evs))
    for e in evs:
        check(abs(e.wx - _key_x(e.midi)) < 1e-6,
              "left: %s midi %d: wx %.2f != key x %.2f"
              % (e.name, e.midi, e.wx, _key_x(e.midi)))


def test_empty_range_plays_nothing():
    """A generated-but-empty range (-s past the end) plays no
    note, while a never-generated hand still plays everything."""
    # the -s / -n range is measured in measure numbers, so the
    # notes must live in real Measures
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    for mnum in range(1, 6):
        m = stream.Measure(number=mnum)
        for i in range(2):
            m.append(note.Note(60 + (mnum + i) % 5, quarterLength=0.5))
        p.append(m)
    sf = stream.Score()
    sf.insert(0, p)
    h = make_hand(sf, start=1000)          # no note in the range
    check(h.fingerseq_index == [], "empty range left a non-empty index")
    vk = make_vk([])
    check(vk._events_for(h, 1.0, 0.0) == [], "empty range still plays")

    h2 = Hand("right", "M")
    h2.verbose = False
    h2.noteseq = reader(sf, beam=0)        # generate() never ran
    check(h2.fingerseq_index is None, "fresh hand has a non-None index")
    evs = vk._events_for(h2, 1.0, 0.0)
    check(len(evs) == 10, "never-generated hand plays %d of 10" % len(evs))


def test_chord_pair_keeps_both_fingers():
    """A 2-note chord carries one Fingering per pitch, and
    re-running generate() neither drops nor doubles them."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    for i in range(6):
        p.insert(i * 1.0, chord.Chord([60 + i, 64 + i],
                                        quarterLength=1))
    s = stream.Score()
    s.insert(0, p)
    h = make_hand(s)

    def fvals():
        return [[a.fingerNumber for a in n.chord21.articulations
                 if isinstance(a, Fingering)] for n in h.noteseq]

    f1 = fvals()
    check(all(len(f) == 2 for f in f1),
          "2-note chord lost a finger: %s" % f1[:2])
    h.generate()
    check(fvals() == f1, "chord fingering changed on 2nd run")
    check(all(len(f) == 2 for f in fvals()),
          "chord fingering doubled")


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
