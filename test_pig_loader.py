"""Tests for pianoplayer.pigloader.load_pig (Phase 1, step 1.3).

    python test_pig_loader.py
"""
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import note as m21note

from pianoplayer.hand import Hand
from pianoplayer.pigloader import load_pig
from pianoplayer.utils import keypos

FAILS = []

# noteID onset offset name onVel offVel channel finger
PIG_TEXT = """// synthetic PIG: RH chord + singles, LH note, unknown, zero-length
0 0.0 0.5 C4 80 0 0 1
1 0.0 0.5 E4 80 0 0 3
2 0.0 0.5 G4 80 0 0 5
3 1.0 1.5 A4 80 0 0 2_1
4 1.0 1.5 D4 80 0 1 -4
5 2.0 2.5 Bb4 80 0 0 _
6 3.0 3.0 F4 80 0 0 1
"""


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def loaded():
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        tmp.write(PIG_TEXT)
        tmp.close()
        return load_pig(tmp.name)
    finally:
        os.unlink(tmp.name)


def test_chord_is_one_group():
    """The 3 same-onset RH lines form one 3-member ascending chord."""
    data = loaded()
    inotes, experts = data[0]
    members = [n for n in inotes if n.isChord]
    check(len(members) == 3, "RH chord members: %d" % len(members))
    check(all(m.NinChord == 3 for m in members), "NinChord != 3")
    check(len({m.chordID for m in members}) == 1, "chord split across IDs")
    midis = [m.midi for m in members]
    check(midis == sorted(midis), "chord not pitch-ascending: %s" % midis)
    check([m.name for m in members] == ["C", "E", "G"],
          "chord pitches wrong")
    check(len(inotes) == 5, "RH INote count %d (want 5: chord + A4 + Bb4)"
          % len(inotes))
    print("   chord: one group of 3, ascending, NinChord == 3")


def test_expert_fingers_parsed():
    """-4 -> 4, 2_1 -> 2, _ -> None; zero-length line skipped."""
    data = loaded()
    inotes_rh, experts_rh = data[0]
    inotes_lh, experts_lh = data[1]
    check(experts_rh == [1, 3, 5, 2, None],
          "RH experts %s" % (experts_rh,))
    check(experts_lh == [4], "LH experts %s" % (experts_lh,))
    check(len(inotes_lh) == 1, "LH INote count %d" % len(inotes_lh))
    check(sum(len(v[0]) for v in data.values()) == 6,
          "total INotes != 6 (duplicated or missing notes)")
    print("   experts: %s / %s" % (experts_rh, experts_lh))


def test_positions_and_midi():
    """x equals keypos(Note(name)); midi set; Bb parsed as flat."""
    data = loaded()
    for channel in (0, 1):
        for n in data[channel][0]:
            want_x = keypos(m21note.Note(n.name + str(n.octave)))
            check(abs(n.x - want_x) < 1e-9,
                  "x %.4f != keypos %.4f for %s%s"
                  % (n.x, want_x, n.name, n.octave))
            check(n.midi is not None, "midi unset")
    bb = [n for n in data[0][0] if n.octave == 4 and n.name == "B-"]
    check(len(bb) == 1 and bb[0].isBlack, "Bb4 flat not parsed")
    print("   positions/midi ok (incl. Bb4 flat)")


def test_generate_runs():
    """Hand.generate() runs on loader output; fingers all 1..5."""
    data = loaded()
    for channel, side in ((0, "right"), (1, "left")):
        inotes, _experts = data[channel]
        h = Hand(side, "M")
        h.verbose = False
        h.noteseq = inotes
        h.generate()
        bad = [n.fingering for n in h.noteseq
               if n.fingering not in (1, 2, 3, 4, 5)]
        check(not bad, "%s fingers out of range: %s" % (side, bad))
    print("   generate ok on loader output, fingers 1..5")


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
