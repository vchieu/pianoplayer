"""Span/size tests for the fingering algorithm (Phase 1, step 1.1).

Reproduces F1 (hand size cancels out of chord/span decisions) and F2
(thumb-pinky MAX_SPAN below an octave).

    python test_fingering_spans.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import chord, stream, tempo

from pianoplayer.hand import HARD, Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import keypos
from pianoplayer import handspec as HS

FAILS = []

OCTAVE_CM = 16.5   # one octave on the keyboard (utils.KEYBSIZE)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def hand(side, size):
    h = Hand(side, size)
    h.verbose = False
    return h


def score_with_chords(chords, bpm=100, ql=1.0):
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    for i, pitches in enumerate(chords):
        p.insert(i * ql, chord.Chord(pitches, quarterLength=ql))
    s = stream.Score()
    s.insert(0, p)
    return s


def solved(side, size, chords):
    h = hand(side, size)
    h.noteseq = reader(score_with_chords(chords), beam=0)
    h.generate()
    return h


# ------------------------------------------------------------------ tests --- #
def test_hand_size_matters():
    """F1: _span_cost(1, 5, octave) non-increasing with size; XXS >= HARD."""
    sizes = ["XXS", "S", "M", "L", "XXL"]
    vals = [hand("right", s)._span_cost(1, 5, OCTAVE_CM) for s in sizes]
    print("   span_cost(1, 5, octave): %s"
          % ", ".join("%s=%.2f" % (s, v) for s, v in zip(sizes, vals)))
    for a, b in zip(vals, vals[1:]):
        check(b <= a,
              "span cost rises with hand size: %s" % (vals,))
    check(vals[0] >= HARD, "XXS octave cost %.2f < HARD" % vals[0])
    check(vals[3] < HARD, "L octave cost %.2f >= HARD" % vals[3])


def test_octave_chord_fingering():
    """F2: repeated octave chords get {1, 5}, low pitch = thumb side."""
    for side in ("right", "left"):
        for size in ("M", "L", "XL"):
            h = solved(side, size, [["C4", "C5"]] * 6)
            by_chord = {}
            for x in h.noteseq:
                if x.isChord:
                    by_chord.setdefault(x.chordID, []).append(x)
            check(len(by_chord) == 6, "%s %s: %d chords" % (side, size, len(by_chord)))
            for cid, members in sorted(by_chord.items()):
                check(len(members) == 2, "chord %s size" % cid)
                lo = min(members, key=lambda m: m.chordnr)
                hi = max(members, key=lambda m: m.chordnr)
                check({lo.fingering, hi.fingering} == {1, 5},
                      "%s %s chord %s: fingers %s" % (
                          side, size, cid, sorted(m.fingering for m in members)))
                if side == "right":
                    check(lo.fingering == 1 and hi.fingering == 5,
                          "right %s chord %s: low=%d high=%d" % (
                              size, cid, lo.fingering, hi.fingering))
                else:
                    check(lo.fingering == 5 and hi.fingering == 1,
                          "left %s chord %s: low=%d high=%d" % (
                              size, cid, lo.fingering, hi.fingering))
            print("   octave %s %s: {1,5} ok" % (side, size))


def test_triad_unchanged():
    """C-E-G triad (right, M) keeps (1, 3, 5) low..high."""
    h = solved("right", "M", [["C4", "E4", "G4"]])
    members = sorted([x for x in h.noteseq if x.isChord],
                     key=lambda m: m.chordnr)
    got = tuple(m.fingering for m in members)
    print("   triad C-E-G: %s" % (got,))
    check(got == (1, 3, 5), "triad fingering %s != (1, 3, 5)" % (got,))


def test_xxs_octave_does_not_crash():
    """XXS octave chords still get fingers 1..5 everywhere."""
    h = solved("right", "XXS", [["C4", "C5"]] * 6)
    bad = [m.fingering for m in h.noteseq
           if m.fingering not in (1, 2, 3, 4, 5)]
    check(not bad, "XXS octave: out-of-range fingers %s" % bad)
    print("   XXS octave: all fingers in 1..5")


def test_span_tables_symmetric():
    """MAX_SPAN/COMFORT_SPAN symmetric; COMFORT < MAX everywhere."""
    for a in (1, 2, 3, 4, 5):
        for b in (1, 2, 3, 4, 5):
            if a == b:
                continue
            check(HS.MAX_SPAN[a][b] == HS.MAX_SPAN[b][a],
                  "MAX_SPAN[%d][%d] != [%d][%d]" % (a, b, b, a))
            check(HS.COMFORT_SPAN[a][b] == HS.COMFORT_SPAN[b][a],
                  "COMFORT_SPAN[%d][%d] != [%d][%d]" % (a, b, b, a))
            check(HS.COMFORT_SPAN[a][b] < HS.MAX_SPAN[a][b],
                  "COMFORT >= MAX for (%d, %d)" % (a, b))
    print("   span tables symmetric, COMFORT < MAX")


def test_keypos_octave_is_octave():
    """Sanity: keypos C5 - keypos C4 == OCTAVE_CM."""
    from music21 import note
    dx = keypos(note.Note("C5")) - keypos(note.Note("C4"))
    check(abs(dx - OCTAVE_CM) < 1e-9, "keypos octave = %.6f" % dx)


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
