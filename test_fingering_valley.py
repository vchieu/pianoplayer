"""Broken-chord valley fingering tests (bug report 2026-10-16).

Left-hand ascending broken chords put inner fingers (3/4) on local pitch
minima and reuse one finger on different keys; a pianist wants the outer
finger (5 on minima, 1 on maxima for LH; mirrored for RH) and no
same-finger/different-key reuse.

Covers the exact passages from the report, twice:
  - synthetic snippets with piece-like context (ql=1.0 @120bpm; isolated
    3-note snippets already solve to 5-3-1, the failure needs the global
    Viterbi context);
  - the real file scores/50-nam-ve-sau.mxl beam 1 (left, M) at fixed bars:
    bar 10 F2 C3 A3 -> 5-3-1; bar 30 G#2 E3 G#3 -> 5-3-1;
    bars 4-5 G3 B3 + G#2 E3 -> no same-finger/different-key reuse;
    bars 11/14/18/20/22 low notes (G2 D2 C3 G2 C2) -> 5.

    python test_fingering_valley.py
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


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_score(names, bpm=120, ql=1.0):
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    for i, m in enumerate(names):
        p.insert(i * ql, note.Note(m, quarterLength=ql))
    s = stream.Score()
    s.insert(0, p)
    return s


def solve_names(names, side="left", bpm=120, ql=1.0, legacy=False):
    sf = make_score(names, bpm, ql)
    h = Hand(side, "M")
    h.verbose = False
    if legacy:
        h.legacy_chord_postpass = True
    h.noteseq = reader(sf, beam=0, tempo_map=score_tempo_map(sf))
    h.generate()
    return [n.fingering for n in h.noteseq]


def solve_piece(beam=1, side="left", legacy=False):
    sf = converter.parse(PIECE)
    h = Hand(side, "M")
    h.verbose = False
    if legacy:
        h.legacy_chord_postpass = True
    h.noteseq = reader(sf, beam=beam, tempo_map=score_tempo_map(sf))
    h.generate()
    return h


def _same_finger_diff_key(notes, fingers):
    """Consecutive different-key pairs sharing one finger (all slow here)."""
    bad = []
    for i, (a, b) in enumerate(zip(notes, notes[1:])):
        if fingers[i] == fingers[i + 1] and abs(b.x - a.x) > 1e-9:
            bad.append((i, a, b))
    return bad


# ------------------------------------------------------------------ tests --- #
def test_valley_synth_bar10():
    """Synth bar-10 shape: valley F2 -> 5, peak A3 -> 1, middle C3 -> 3."""
    got = solve_names(["E4", "B2", "G3", "B3", "G#2", "E3", "G#3"])
    print("   synth ctx bar10-shape: %s" % (got,))
    check(got[4] == 5, "valley G#2 on %d, want 5: %s" % (got[4], got))
    check(got[5] == 3, "middle E3 on %d, want 3: %s" % (got[5], got))
    check(got[6] == 1, "peak G#3 on %d, want 1: %s" % (got[6], got))


def test_valley_synth_noreuse():
    """Synth bars-4-5 shape: no same-finger/different-key consecutive pair."""
    names = ["E4", "B2", "G3", "B3", "G#2", "E3", "G#3"]
    sf = make_score(names)
    h = Hand("left", "M")
    h.verbose = False
    h.noteseq = reader(sf, beam=0, tempo_map=score_tempo_map(sf))
    h.generate()
    got = [n.fingering for n in h.noteseq]
    bad = _same_finger_diff_key(h.noteseq, got)
    print("   synth noreuse: %s bad=%d" % (got, len(bad)))
    check(not bad, "same-finger reuse: %s" % (
        [(i, a.name + str(a.octave), b.name + str(b.octave), got[i])
         for i, a, b in bad],))
    check(got[1] == 5, "valley B2 on %d, want 5: %s" % (got[1], got))


def test_valley_synth_legacy_agrees():
    """Legacy Viterbi path agrees with layered DP on the valley passage."""
    names = ["E4", "B2", "G3", "B3", "G#2", "E3", "G#3"]
    got = solve_names(names, legacy=True)
    print("   synth legacy-path: %s" % (got,))
    check(got[1] == 5, "legacy valley B2 on %d, want 5: %s" % (got[1], got))
    check(got[4:7] == [5, 3, 1],
          "legacy bar10-shape %s, want [5, 3, 1]" % (got[4:7],))


def test_valley_file_bar10():
    """File bar 10 LH: F2 C3 A3 -> 5-3-1."""
    h = solve_piece()
    seq = [(n, n.fingering) for n in h.noteseq if n.measure == 10]
    got = [f for _, f in seq]
    names = ["%s%d" % (n.name, n.octave) for n, _ in seq]
    print("   file bar10: %s -> %s" % (names, got))
    check(names == ["F2", "C3", "A3"], "bar 10 notes changed: %s" % names)
    check(got == [5, 3, 1], "bar 10 %s, want [5, 3, 1]" % (got,))


def test_valley_file_bar30():
    """File bar 30 LH: G#2 E3 G#3 -> 5-3-1."""
    h = solve_piece()
    seq = [(n, n.fingering) for n in h.noteseq if n.measure == 30]
    got = [f for _, f in seq]
    names = ["%s%d" % (n.name, n.octave) for n, _ in seq]
    print("   file bar30: %s -> %s" % (names, got))
    check(names == ["G#2", "E3", "G#3"], "bar 30 notes changed: %s" % names)
    check(got == [5, 3, 1], "bar 30 %s, want [5, 3, 1]" % (got,))


def test_valley_file_bars4_5():
    """File bars 4-5 LH: no same-finger/different-key consecutive pair."""
    h = solve_piece()
    seq = [n for n in h.noteseq if n.measure in (4, 5)]
    got = [n.fingering for n in seq]
    names = ["%s%d" % (n.name, n.octave) for n in seq]
    bad = _same_finger_diff_key(seq, got)
    print("   file bars4-5: %s -> %s bad=%d" % (names, got, len(bad)))
    check(names == ["B2", "G3", "B3", "G#2", "E3", "G#3"],
          "bars 4-5 notes changed: %s" % names)
    check(not bad, "bars 4-5 same-finger reuse: %s" % (
        [(names[i], names[i + 1], got[i]) for i, _, _ in bad],))


def test_valley_file_low_notes():
    """File bars 11/14/18/20/22 LH: low note on outer finger 5."""
    h = solve_piece()
    want = {11: ("G2", 0), 14: ("D2", 0), 18: ("C3", 0),
            20: ("G2", 0), 22: ("C2", 0)}
    for bar, (nm, _idx) in sorted(want.items()):
        seq = [n for n in h.noteseq if n.measure == bar]
        got = [n.fingering for n in seq]
        names = ["%s%d" % (n.name, n.octave) for n in seq]
        print("   file bar%d: %s -> %s" % (bar, names, got))
        check(names[0] == nm, "bar %d first note %s, want %s"
              % (bar, names[0], nm))
        check(got[0] == 5, "bar %d low %s on %d, want 5: %s"
              % (bar, names[0], got[0], got))
        bad = _same_finger_diff_key(seq, got)
        check(not bad, "bar %d same-finger reuse: %s" % (
            bar, [(names[i], names[i + 1], got[i]) for i, _, _ in bad]))


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
