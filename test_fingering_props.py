"""Property tests for the fingering solver (Phase 1, step 1.4).

Seeded random scores (random.Random(seed), never the global RNG), 20
seeds, 120 notes each, both hands.

    python test_fingering_props.py
"""
import os
import random
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import chord, note, stream, tempo

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader

FAILS = []

N_SEEDS = 20
N_NOTES = 120
DURS = (0.25, 0.5, 1.0)
OVERLAP_EPS = 1e-9


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def random_score(seed):
    """120-note seeded walk: steps [-5,+5] semitones from C4, clamped C3..C6."""
    rng = random.Random(seed)
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    t, midi = 0.0, 60
    for _ in range(N_NOTES):
        ql = DURS[rng.randint(0, len(DURS) - 1)]
        p.insert(t, note.Note(midi, quarterLength=ql))
        t += ql
        midi = min(84, max(48, midi + rng.randint(-5, 5)))
    s = stream.Score()
    s.insert(0, p)
    return s


def solve_seed(seed, side):
    h = Hand(side, "M")
    h.verbose = False
    h.noteseq = reader(random_score(seed), beam=0)
    h.generate()
    return h


def test_all_fingers_in_range():
    bad = 0
    for seed in range(N_SEEDS):
        for side in ("right", "left"):
            h = solve_seed(seed, side)
            bad += sum(1 for n in h.noteseq
                       if n.fingering not in (1, 2, 3, 4, 5))
    print("   props out-of-range fingerings: %d" % bad)
    check(bad == 0, "%d notes with fingering outside 1..5" % bad)


def test_deterministic():
    for seed in range(N_SEEDS):
        for side in ("right", "left"):
            a = [n.fingering for n in solve_seed(seed, side).noteseq]
            b = [n.fingering for n in solve_seed(seed, side).noteseq]
            check(a == b, "seed %d %s not deterministic" % (seed, side))
    print("   props deterministic: %d seeds x both hands identical" % N_SEEDS)


def test_idempotent():
    for seed in range(N_SEEDS):
        for side in ("right", "left"):
            h = solve_seed(seed, side)
            first = [n.fingering for n in h.noteseq]
            h.generate()
            second = [n.fingering for n in h.noteseq]
            check(first == second,
                  "seed %d %s generate() not idempotent" % (seed, side))
    print("   props idempotent: 2nd generate() identical")


def test_no_overlap_same_finger():
    viol_total = 0
    for seed in range(N_SEEDS):
        for side in ("right", "left"):
            h = solve_seed(seed, side)
            notes = sorted(h.noteseq,
                           key=lambda n: (n.time_real, n.dur_real))
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
                    viol_total += 1
    print("   props overlapping same-finger violations: %d" % viol_total)
    check(viol_total == 0, "%d overlap violations" % viol_total)


def _solve_chord(pitches, side):
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    p.insert(0, chord.Chord(pitches, quarterLength=1.0))
    s = stream.Score()
    s.insert(0, p)
    h = Hand(side, "M")
    h.verbose = False
    h.noteseq = reader(s, beam=0)
    h.generate()
    return h


def test_five_note_chord_uses_all_fingers():
    """A 5-note chord within one octave gets exactly {1,2,3,4,5}."""
    for side in ("right", "left"):
        h = _solve_chord(["C4", "E4", "G4", "B4", "C5"], side)
        got = sorted(n.fingering for n in h.noteseq)
        print("   5-note chord %s: %s" % (side, got))
        check(got == [1, 2, 3, 4, 5],
              "%s 5-note chord fingers %s" % (side, got))


def test_six_note_chord_no_crash():
    """A 6-note chord (F9 edge): no raise, every finger still 1..5."""
    for side in ("right", "left"):
        h = _solve_chord(["C4", "D4", "E4", "F4", "G4", "A4"], side)
        got = [n.fingering for n in h.noteseq]
        check(len(got) == 6, "%s 6-note chord lost notes" % side)
        check(all(f in (1, 2, 3, 4, 5) for f in got),
              "%s 6-note chord fingers out of range: %s" % (side, got))
    print("   6-note chord: no crash, fingers in 1..5")


def _overlap_score(seed, n=200):
    """Seeded score with long overlapping holds and chords (F7 test bed)."""
    rng = random.Random(1000 + seed)
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    t, midi = 0.0, 60
    k = 0
    while k < n:
        if rng.randint(0, 6) == 0:
            # 2-3 note chord, sometimes with a long duration (held)
            span = rng.randint(0, 1)
            pitches = [midi + rng.randint(0, 7) + 2 * m for m in range(2 + span)]
            pitches = [min(84, max(48, x)) for x in pitches]
            ql = 4.0 if rng.randint(0, 3) == 0 else 0.5
            p.insert(t, chord.Chord(list(dict.fromkeys(pitches)),
                                    quarterLength=ql))
            k += len(pitches)
            t += 0.5
        else:
            ql = 4.0 if rng.randint(0, 4) == 0 else 0.5
            p.insert(t, note.Note(midi, quarterLength=ql))
            k += 1
            t += 0.5
        midi = min(84, max(48, midi + rng.randint(-5, 5)))
    s = stream.Score()
    s.insert(0, p)
    return s


def test_repair_held_equivalence():
    """F7: linear _repair_held equals _repair_held_reference on 30 seeded
    overlapping/chordal scores (raw solve_viterbi output as input)."""
    for seed in range(30):
        for side in ("right", "left"):
            h = Hand(side, "M")
            h.verbose = False
            h.noteseq = reader(_overlap_score(seed), beam=0)
            notes = h.noteseq
            vit = h.solve_viterbi(notes)
            got = h._repair_held(notes, vit)
            want = h._repair_held_reference(notes, vit)
            check(list(got) == list(want),
                  "seed %d %s: linear differs from reference" % (seed, side))
    print("   repair_held equivalence: 30 seeds x both hands identical")


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
