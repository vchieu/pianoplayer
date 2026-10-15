"""Pinned-fingering tests (Phase 4, §6: F8).

Score lyric hints (1..5) pin fingers via the separate `INote.hint`
field (`fingering` keeps the solver output, so generate() stays
idempotent).

    python test_fingering_hints.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import chord, note, stream, tempo

from pianoplayer.hand import PIN_COST, Hand
from pianoplayer.scorereader import INote, reader

FAILS = []
OVERLAP_EPS = 1e-9


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def scale_score(pinned_idx=2, pinned_lyric="5", lyric_kind="note"):
    """C major scale, 8 quarter notes at 120 bpm, one lyric hint.

    `lyric_kind="note"`: the lyric goes on the pinned note itself
    (lyrics attach per-note in music21).
    """
    names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    for i, nm in enumerate(names):
        n = note.Note(nm, quarterLength=1.0)
        if i == pinned_idx and lyric_kind == "note":
            n.addLyric(pinned_lyric)
        p.insert(float(i), n)
    s = stream.Score()
    s.insert(0, p)
    return s


def solve(score, side="right", legacy=False):
    h = Hand(side, "M")
    h.verbose = False
    if legacy:
        h.legacy_chord_postpass = True
    h.noteseq = reader(score, beam=0)
    h.generate()
    return h


def overlap_violations(h):
    notes = sorted(h.noteseq, key=lambda n: (n.time_real, n.dur_real))
    viol = 0
    for i, a in enumerate(notes):
        if a.fingering not in (1, 2, 3, 4, 5):
            continue
        a_end = a.time_real + max(a.dur_real, 0.0)
        for b in notes[i + 1:]:
            if b.time_real >= a_end - OVERLAP_EPS:
                break
            if b.time_real + max(b.dur_real, 0.0) <= a.time_real + OVERLAP_EPS:
                continue
            if b.fingering != a.fingering:
                continue
            if abs(b.x - a.x) < 1e-9:
                continue  # same pitch: legal
            viol += 1
    return viol


def test_scale_pin_third_note_to_5():
    """§6(a): scale with the 3rd note pinned to finger 5 via lyric "5"."""
    check(PIN_COST == 1000.0, "PIN_COST changed: %r" % PIN_COST)
    for legacy in (False, True):
        h = solve(scale_score(), legacy=legacy)
        check(h.noteseq[2].hint == 5,
              "legacy=%s: hint not parsed: %s"
              % (legacy, [n.hint for n in h.noteseq]))
        check(all(n.hint == 0 for i, n in enumerate(h.noteseq) if i != 2),
              "legacy=%s: hint leaked to other notes" % legacy)
        got = [n.fingering for n in h.noteseq]
        check(got[2] == 5,
              "legacy=%s: pinned note got finger %s, full %s"
              % (legacy, got[2], got))
        check(all(f in (1, 2, 3, 4, 5) for f in got),
              "legacy=%s: fingers out of range: %s" % (legacy, got))
        check(overlap_violations(h) == 0,
              "legacy=%s: new overlap violations" % legacy)
    print("   scale pin: 3rd note -> 5 on both paths, legal elsewhere")


def test_double_generate_identical():
    """§6(b): a second generate() run changes nothing (no hint leakage)."""
    for legacy in (False, True):
        h = solve(scale_score(), legacy=legacy)
        first = [n.fingering for n in h.noteseq]
        hints_before = [n.hint for n in h.noteseq]
        h.generate()
        second = [n.fingering for n in h.noteseq]
        check(first == second,
              "legacy=%s: 2nd generate() differs: %s vs %s"
              % (legacy, first, second))
        check([n.hint for n in h.noteseq] == hints_before,
              "legacy=%s: hints mutated by generate()" % legacy)
    print("   double generate: identical on both paths")


def test_chord_member_hints():
    """Chord members keep their own hints (one lyric per member)."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    c = chord.Chord(["C4", "E4", "G4"], quarterLength=1.0)
    # generate() writes member `chordnr` under lyric number N - chordnr,
    # so the reader matches lyrics to members in number-descending order.
    c.addLyric("1", 3)
    c.addLyric("3", 2)
    c.addLyric("5", 1)
    p.insert(0, c)
    s = stream.Score()
    s.insert(0, p)
    for legacy in (False, True):
        h = Hand("right", "M")
        h.verbose = False
        if legacy:
            h.legacy_chord_postpass = True
        h.noteseq = reader(s, beam=0)
        check([n.hint for n in h.noteseq] == [1, 3, 5],
              "legacy=%s: chord hints %s"
              % (legacy, [n.hint for n in h.noteseq]))
        h.generate()
        got = [n.fingering for n in h.noteseq]
        check(got == [1, 3, 5], "legacy=%s: chord fingers %s" % (legacy, got))
    print("   chord pins [1,3,5] on both paths")


def test_chord_single_shared_lyric_pins_lowest_only():
    """One shared lyric pins only the lowest pitch (documented)."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    c = chord.Chord(["C4", "E4", "G4"], quarterLength=1.0)
    c.addLyric("1")
    p.insert(0, c)
    s = stream.Score()
    s.insert(0, p)
    h = Hand("right", "M")
    h.verbose = False
    h.noteseq = reader(s, beam=0)
    check([n.hint for n in h.noteseq] == [1, 0, 0],
          "shared-lyric chord hints %s" % [n.hint for n in h.noteseq])
    h.generate()
    got = [n.fingering for n in h.noteseq]
    check(got[0] == 1, "lowest chord member not pinned: %s" % got)
    check(all(f in (1, 2, 3, 4, 5) for f in got), "fingers %s" % got)
    print("   shared lyric pins lowest pitch only: %s" % got)


def test_lone_pinned_note():
    """A single hinted note solves to its hint (both DP entry points)."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    n = note.Note("C4", quarterLength=1.0)
    n.addLyric("4")
    p.insert(0, n)
    s = stream.Score()
    s.insert(0, p)
    h = Hand("right", "M")
    h.verbose = False
    h.noteseq = reader(s, beam=0)
    check(h.noteseq[0].hint == 4, "lone hint %s" % h.noteseq[0].hint)
    check(h.solve_viterbi(h.noteseq) == [4], "layered lone pin")
    h.legacy_chord_postpass = True
    check(h.solve_viterbi(h.noteseq) == [4], "legacy lone pin")
    h.legacy_chord_postpass = False
    h.generate()
    check(h.noteseq[0].fingering == 4, "generate lone pin")
    print("   lone pinned note -> hint on all entry points")


def _held_pair(pin_second=None):
    """Two overlapping notes on different keys, both wanting finger 2."""
    a = INote()
    a.name, a.octave = "C", 4
    a.x, a.time, a.time_real = 0.0, 0.0, 0.0
    a.duration, a.dur_real = 2.0, 2.0
    a.isChord, a.chordID, a.isBlack = False, 0, False
    a.t_sec, a.midi, a.hint = None, None, 0
    b = INote()
    b.name, b.octave = "D", 4
    b.x, b.time, b.time_real = 2.0, 0.5, 0.5
    b.duration, b.dur_real = 0.5, 0.5
    b.isChord, b.chordID, b.isBlack = False, 0, False
    b.t_sec, b.midi = None, None
    b.hint = pin_second or 0
    return [a, b]


def test_repair_held_never_moves_pinned():
    """_repair_held (and reference) never re-fingers a pinned note."""
    for fn in ("_repair_held", "_repair_held_reference"):
        # pinned conflicting note stays put ...
        h = Hand("right", "M")
        h.verbose = False
        out = getattr(h, fn)(_held_pair(pin_second=2), [2, 2])
        check(list(out) == [2, 2], "%s moved a pinned note: %s" % (fn, out))
        # ... while the unpinned twin is repaired identically by both.
        h2 = Hand("right", "M")
        h2.verbose = False
        out2 = getattr(h2, fn)(_held_pair(), [2, 2])
        check(out2[0] == 2 and out2[1] != 2,
              "%s did not repair unpinned conflict: %s" % (fn, out2))
    h3 = Hand("right", "M")
    h3.verbose = False
    check(list(h3._repair_held(_held_pair(pin_second=2), [2, 2]))
          == list(h3._repair_held_reference(_held_pair(pin_second=2), [2, 2])),
          "linear/reference disagree on pinned input")
    print("   repair keeps pins, still fixes unpinned conflicts")


def test_unpinned_hint_plumbing_is_identity():
    """Notes with hint == 0 solve exactly like notes without the field."""
    base = scale_score(pinned_idx=-1)  # no lyric at all
    h1 = solve(base)
    a = [n.fingering for n in h1.noteseq]
    h2 = Hand("right", "M")
    h2.verbose = False
    h2.noteseq = reader(base, beam=0)
    for n in h2.noteseq:
        del n.hint  # pre-Phase-4 shape: _hint_of falls back to 0
    h2.generate()
    b = [n.fingering for n in h2.noteseq]
    check(a == b, "hint==0 differs from no-hint-field: %s vs %s" % (a, b))
    print("   hint==0 identical to missing hint field")


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
