"""Fingering audit (Phase 0, step 0.1 of PIANOPLAYER_FINGERING_PLAN.md).

Diagnostic script (not a test; never asserts).
Usage: python fingering_audit.py > docs/fingering_baseline.txt

One row per (source, side, size):
  source | side | size | notes | solve_s | violations | overlaps |
  chords | postpass_changed | total_cost | finger histogram
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import chord, converter, note, stream, tempo

import pianoplayer.hand as handmod
from pianoplayer.hand import HARD, Hand
from pianoplayer.scorereader import reader

OVERLAP_EPS = 1e-9


def make_hand_for(sf, side="right", size="M"):
    h = Hand(side, size)
    h.verbose = False
    h.noteseq = reader(sf, beam=0)
    return h


# ------------------------------------------------------------------ sources --
def _synth_multivoice():
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    p.insert(0, note.Note("C4", quarterLength=4.0))
    for i, m in enumerate(["E4", "G4", "A4", "G4", "E4", "F4", "D4", "B3"]):
        p.insert(i * 0.5, note.Note(m, quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_chords():
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    roots = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, r in enumerate(roots):
        p.insert(i * 1.0, chord.Chord([r, r + 4, r + 7], quarterLength=1.0))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_repeated():
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    seq = ["C4", "C4", "C4", "D4", "E4", "E4", "D4", "C4",
           "G4", "G4", "A4", "G4", "F4", "E4", "D4", "C4"]
    for i, m in enumerate(seq):
        p.insert(i * 0.5, note.Note(m, quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_octaves():
    """8 consecutive octave chords [C4, C5] walking up by a tone, qn @100bpm."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=100))
    for i in range(8):
        r = 60 + 2 * i
        p.insert(i * 1.0, chord.Chord([r, r + 12], quarterLength=1.0))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_triads():
    """8 root-position triads walking up by a tone, qn @100bpm."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=100))
    for i in range(8):
        r = 60 + 2 * i
        p.insert(i * 1.0, chord.Chord([r, r + 4, r + 7], quarterLength=1.0))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_alberti():
    """64 eighth notes C-G-E-G pattern at 120 bpm."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    pat = ["C4", "G4", "E4", "G4"]
    for i in range(64):
        p.insert(i * 0.5, note.Note(pat[i % 4], quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_arpeggio():
    """C arpeggio from eval_fingering.GOLD, plus 2 more octaves."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    base = [60, 64, 67, 72, 76, 79, 84, 88]  # C4..E6
    extra = [91, 96, 100, 103, 108]          # G6 C7 E7 G7 C8
    for i, m in enumerate(base + extra):
        p.insert(i * 0.5, note.Note(m, quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_long(n=3000, seed=7):
    """Seeded random walk, steps [-4,+4] semitones from C4, clamped C3..C6."""
    rng = random.Random(seed)
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    midi = 60
    for i in range(n):
        p.insert(i * 0.5, note.Note(midi, quarterLength=0.5))
        midi = min(84, max(48, midi + rng.randint(-4, 4)))
    s = stream.Score()
    s.insert(0, p)
    return s


def sources():
    out = []
    for fn in ("bach_invention4.xml", "test_chords.xml", "test_scales.xml"):
        path = os.path.join(HERE, "scores", fn)
        if os.path.isfile(path):
            out.append(("file:" + fn, lambda p=path: converter.parse(p), False))
        else:
            print("   (missing scores/%s -- skipped)" % fn)
    out.append(("synth:multivoice", _synth_multivoice, False))
    out.append(("synth:chords", _synth_chords, False))
    out.append(("synth:repeated", _synth_repeated, False))
    out.append(("synth:octaves", _synth_octaves, True))
    out.append(("synth:triads", _synth_triads, True))
    out.append(("synth:alberti", _synth_alberti, False))
    out.append(("synth:arpeggio", _synth_arpeggio, False))
    out.append(("synth:long", _synth_long, False))
    return out


# ------------------------------------------------------------------ metrics --
def count_overlaps(h):
    """A1.2 logic of test_fingering.py::test_no_overlap_same_finger (copy)."""
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


def audit_row(label, side, size):
    builder = dict((l, b) for l, b, _ in sources())[label]
    sf = builder()
    h = make_hand_for(sf, side, size)
    t0 = time.perf_counter()
    h.generate(1, 1000)
    solve_s = time.perf_counter() - t0
    notes = h.noteseq
    n = len(notes)
    viols = sum(
        1 for a, fa, b, fb in zip(notes, [x.fingering for x in notes],
                                  notes[1:], [x.fingering for x in notes][1:])
        if h.pair_cost(a, fa, b, fb) >= handmod.HARD)
    overlaps = count_overlaps(h)
    by_chord = {}
    for x in notes:
        if x.isChord:
            by_chord.setdefault(x.chordID, []).append(x)
    n_chords = sum(1 for members in by_chord.values() if len(members) >= 2)
    vit = h.solve_viterbi(notes)
    post = h._apply_chord_candidates(notes, vit)
    postpass_changed = sum(1 for a, b in zip(vit, post) if a != b)
    total = h.total_cost(notes, [x.fingering for x in notes])
    hist = {f: sum(1 for x in notes if x.fingering == f) for f in (1, 2, 3, 4, 5)}
    hist_s = ",".join("%d:%d" % (f, hist[f]) for f in (1, 2, 3, 4, 5))
    return (label, side, size, n, solve_s, viols, overlaps,
            n_chords, postpass_changed, total, hist_s)


def main():
    print("# fingering audit (Phase 0 step 0.1; diagnostic, never asserts)")
    print("# columns: source | side | size | notes | solve_s | violations | "
          "overlaps | chords | postpass_changed | total_cost | hist(1..5)")
    for label, builder, multi_size in sources():
        sizes = ("XXS", "M", "XXL") if multi_size else ("M",)
        for side in ("right", "left"):
            for size in sizes:
                (l, s, z, n, solve_s, viols, overlaps,
                 n_chords, postpass_changed, total, hist_s) = audit_row(
                    label, side, size)
                print("%s | %s | %s | %d | %.3f | %d | %d | %d | %d | %.2f | %s"
                      % (l, s, z, n, solve_s, viols, overlaps,
                         n_chords, postpass_changed, total, hist_s))


if __name__ == "__main__":
    main()
