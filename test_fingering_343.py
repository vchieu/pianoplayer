"""3-4-3 weak-pair alternation tests (bug report 2026-10-16, Bug5-A).

`scores/50-nam-ve-sau.mxl` beam 1 (left, M), bars 31-32:
  B2(1) G#3(2) G#2(3) A2(4) E3(3) A3(1) B3(3)
(the leading E2 of bar 31 is outside the 7-note span). The G#2 valley is
only 1 semitone deep (A2 a semitone above), so VALLEY_MIN_DEPTH=2 never
fires, and the solver alternates the weak 3-4 pair: G#2(3) A2(4) E3(3).

Round-A3 resolution (supervisor-accepted DP result): with the
VALLEY_DROP_MIN=5 arrival gate the span solves to [1, 2, 5, 4, 3, 1, 3],
accepted per the A1 clause -- no (3,4,3)/(4,3,4)/(3,2,3) triple, no
consecutive same-finger reuse, G#2 valley-drop lands 5; the DP sees the
following bar (C4 above B3), which the hand-written 1-2-5-4-3-2-1 want
did not. The sterile-context isolated triple needs cap/velocity
rebalancing (out of scope); the synth test below is property-based.

    python test_fingering_343.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import converter, note, stream, tempo

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import score_tempo_map

FAILS = []

PIECE = os.path.join(HERE, "scores", "50-nam-ve-sau.mxl")

# 7-note span of bars 31-32 (leading E2 of bar 31 excluded).
# Supervisor-accepted DP result (A1 clause): valley-drop lands 5, no bad
# triple, no same-finger reuse; the DP sees bar 33 (C4 above B3).
SPAN_NAMES = ["B2", "G#3", "G#2", "A2", "E3", "A3", "B3"]
SPAN_ACCEPTED = [1, 2, 5, 4, 3, 1, 3]

# Forbidden adjacent triples: weak-pair alternations (3,4,3)/(4,3,4) and
# the unmusical dodge (3,2,3) the E4 penalty probe produced (ascending
# semitone on descending fingers = non-thumb crossing).
BAD_TRIPLES = ((3, 4, 3), (4, 3, 4), (3, 2, 3))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def solve_piece(beam=1, side="left", legacy=False):
    sf = converter.parse(PIECE)
    h = Hand(side, "M")
    h.verbose = False
    if legacy:
        h.legacy_chord_postpass = True
    h.noteseq = reader(sf, beam=beam, tempo_map=score_tempo_map(sf))
    h.generate()
    return h


def span_31_32(h):
    seq = [n for n in h.noteseq if n.measure in (31, 32)]
    names = ["%s%d" % (n.name, n.octave) for n in seq]
    # Leading E2 of bar 31 is outside the span; the span is the last 7.
    check(names[-7:] == SPAN_NAMES,
          "bars 31-32 notes changed: %s" % names)
    return seq[-7:]


def triples_bad(fingers):
    """Adjacent forbidden triples (indices into the finger list)."""
    return [i for i in range(len(fingers) - 2)
            if tuple(fingers[i:i + 3]) in BAD_TRIPLES]


def reuse_bad(notes, fingers):
    """Consecutive different-key pairs sharing one finger."""
    return [i for i, (a, b) in enumerate(zip(notes, notes[1:]))
            if fingers[i] == fingers[i + 1] and abs(b.x - a.x) > 1e-9]


# ------------------------------------------------------------------ tests --- #
def test_343_file_bars31_32():
    """File bars 31-32 LH span: accepted [1, 2, 5, 4, 3, 1, 3].

    Supervisor-accepted per the A1 clause: G#2 valley-drop (12 st arrival,
    1 st departure) lands outer 5, no bad triple, no same-finger reuse.
    """
    h = solve_piece()
    seq = span_31_32(h)
    got = [n.fingering for n in seq]
    names = ["%s%d" % (n.name, n.octave) for n in seq]
    print("   file bars31-32 span: %s -> %s" % (names, got))
    check(got == SPAN_ACCEPTED,
          "bars 31-32 span %s, accepted %s" % (got, SPAN_ACCEPTED))
    bad = triples_bad(got)
    check(not bad, "bad triple at %s: %s" % (bad, got))
    ruse = reuse_bad(seq, got)
    check(not ruse, "same-finger reuse at %s: %s" % (ruse, got))


def test_343_synth_triple():
    """Synth peak + valley-drop shape (property-based).

    C2 G2 E3(peak, +9 arrival) E2(valley, -12 arrival, +1 departure)
    F2 C3 F3 G3 @142bpm quarters: the E2 valley is the G#2 analog (deep
    arrival, shallow departure -- old min-side gate fails, arrival gate
    fires). Properties: valley lands outer 5, peak lands outer 1, no bad
    triple, no same-finger reuse. Solves to [5, 3, 1, 5, 4, 3, 1, 3].
    """
    names = ["C2", "G2", "E3", "E2", "F2", "C3", "F3", "G3"]
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=142))
    for i, m in enumerate(names):
        p.insert(i * 1.0, note.Note(m, quarterLength=1.0))
    s = stream.Score()
    s.insert(0, p)
    h = Hand("left", "M")
    h.verbose = False
    h.noteseq = reader(s, beam=0, tempo_map=score_tempo_map(s))
    h.generate()
    ns = h.noteseq
    got = [n.fingering for n in ns]
    midis = [n.midi for n in ns]
    print("   synth 343-shape: %s -> %s" % (names, got))
    check([n.name + str(n.octave) for n in ns] == names,
          "synth notes changed: %s" % midis)
    # Shape guard: idx 2 is a strict peak (+9 arrival), idx 3 a strict
    # valley (-12 arrival, +1 departure) -- the gate's target shape.
    check(midis[2] > midis[1] and midis[2] > midis[3]
          and midis[2] - midis[1] >= 5, "peak shape changed: %s" % midis)
    check(midis[3] < midis[2] and midis[3] < midis[4]
          and midis[2] - midis[3] >= 5, "valley shape changed: %s" % midis)
    check(got[3] == 5, "valley E2 on %d, want outer 5: %s" % (got[3], got))
    check(got[2] == 1, "peak E3 on %d, want outer 1: %s" % (got[2], got))
    bad = triples_bad(got)
    check(not bad, "synth bad triple at %s: %s" % (bad, got))
    ruse = reuse_bad(ns, got)
    check(not ruse, "synth same-finger reuse at %s: %s" % (ruse, got))


def test_343_legacy_agrees():
    """Legacy Viterbi path agrees with layered DP on bars 31-32 span."""
    h = solve_piece()
    hl = solve_piece(legacy=True)
    got = [n.fingering for n in span_31_32(h)]
    got_leg = [n.fingering for n in span_31_32(hl)]
    print("   layered %s vs legacy %s" % (got, got_leg))
    check(got_leg == got,
          "legacy %s disagrees with layered %s" % (got_leg, got))


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
