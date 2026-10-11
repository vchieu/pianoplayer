"""
Fingering quality tests (Phase A1 of PIANOPLAYER_FIX_PLAN.md).

Mirrors test_regression.py structure (check(), main() collecting test_*).

For scores/bach_invention4.xml, scores/test_chords.xml, scores/test_scales.xml
plus 3 synthetic scores (multi-voice, consecutive chords, repeated notes),
solved for both right and left hands, checks:

1. every note has fingering in {1..5};
2. no two time-overlapping notes share a finger, unless same pitch;
3. consecutive different-pitch notes on the same finger < 0.25 s apart:
   0 violations total;
4. within a chord, fingers increase with pitch (right) / decrease (left);
5. a 500-note score solves in < 2 s.

    python test_fingering.py
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import chord, converter, note, stream, tempo

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import score_tempo

FAILS = []

REUSE_LIMIT_S = 0.25
OVERLAP_EPS = 1e-9
HELD_EPS = 1e-3


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_hand(sf, side="right", start=1, nmeas=1000):
    h = Hand(side, "M")
    h.verbose = False
    h.noteseq = reader(sf, beam=0)
    h.generate(start, nmeas)
    return h


# ---------------------------------------------------------------- sources --- #
def _file_builder(path):
    def build():
        return converter.parse(path)
    return build


def _synth_multivoice():
    """One long held note with moving notes on top (multi-voice overlap)."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    p.insert(0, note.Note("C4", quarterLength=4.0))
    for i, m in enumerate(["E4", "G4", "A4", "G4", "E4", "F4", "D4", "B3"]):
        p.insert(i * 0.5, note.Note(m, quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_chords():
    """Consecutive 3-note chords walking up."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    roots = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, r in enumerate(roots):
        p.insert(i * 1.0, chord.Chord([r, r + 4, r + 7], quarterLength=1.0))
    s = stream.Score()
    s.insert(0, p)
    return s


def _synth_repeated():
    """Repeated notes (same pitch reuse of one finger is legal) mixed with
    stepwise motion."""
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    seq = ["C4", "C4", "C4", "D4", "E4", "E4", "D4", "C4",
           "G4", "G4", "A4", "G4", "F4", "E4", "D4", "C4"]
    for i, m in enumerate(seq):
        p.insert(i * 0.5, note.Note(m, quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    return s


def sources():
    """[(label, builder, bpm)] — builder() returns a fresh score each call."""
    out = []
    for fn in ("bach_invention4.xml", "test_chords.xml", "test_scales.xml"):
        path = os.path.join(HERE, "scores", fn)
        if os.path.isfile(path):
            out.append(("file:" + fn, _file_builder(path), None))
        else:
            print("   (missing scores/%s — skipped)" % fn)
    out.append(("synth:multivoice", _synth_multivoice, 120.0))
    out.append(("synth:chords", _synth_chords, 120.0))
    out.append(("synth:repeated", _synth_repeated, 120.0))
    return out


def solve_all():
    """Solve every source for both hands. Returns [(label, side, hand, bpm)]."""
    solved = []
    for label, builder, bpm in sources():
        for side in ("right", "left"):
            sf = builder()
            if bpm is None:
                try:
                    bpm = float(score_tempo(sf))
                except Exception:
                    bpm = 120.0
            h = make_hand(sf, side)
            solved.append((label, side, h, bpm))
    return solved


SOLVED = None


def solved():
    global SOLVED
    if SOLVED is None:
        SOLVED = solve_all()
        n = sum(len(h.noteseq) for _, _, h, _ in SOLVED)
        print("   solved %d (source, hand) pairs, %d notes total"
              % (len(SOLVED), n))
    return SOLVED


def _pitch_key(n, side):
    # left-hand x is mirrored (negated); un-mirror so the key rises w/ pitch
    return float(n.x) if side == "right" else -float(n.x)


# ------------------------------------------------------------------ tests --- #
def test_all_fingers_in_range():
    """A1.1: every note has fingering in {1..5}."""
    bad_total = 0
    for label, side, h, _bpm in solved():
        bad = [i for i, x in enumerate(h.noteseq)
               if x.fingering not in (1, 2, 3, 4, 5)]
        bad_total += len(bad)
        if bad:
            print("   %s %s: %d notes without finger 1..5, e.g. %s"
                  % (label, side, len(bad), bad[:5]))
    print("   A1.1 out-of-range fingerings: %d" % bad_total)
    check(bad_total == 0, "A1.1: %d notes with fingering outside 1..5"
          % bad_total)


def test_no_overlap_same_finger():
    """A1.2: no two time-overlapping notes share a finger (same pitch ok)."""
    viol_total = 0
    for label, side, h, _bpm in solved():
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
                    continue          # same pitch: legal
                viol += 1
        viol_total += viol
        if viol:
            print("   %s %s: %d overlapping same-finger pairs" % (label, side, viol))
    print("   A1.2 overlapping same-finger violations: %d" % viol_total)
    check(viol_total == 0, "A1.2: %d overlapping same-finger violations"
          % viol_total)


def test_no_fast_same_finger_reuse():
    """A1.3: consecutive different-pitch notes on one finger < 0.25 s: 0."""
    viol_total = 0
    for label, side, h, bpm in solved():
        sec_per_q = 60.0 / max(float(bpm), 1e-3)
        viol = 0
        seq = h.noteseq
        for a, b in zip(seq, seq[1:]):
            if a.fingering not in (1, 2, 3, 4, 5):
                continue
            if b.fingering != a.fingering:
                continue
            if abs(b.x - a.x) < 1e-9:
                continue              # same pitch: legal (repeated notes)
            dt = abs(b.time_real - a.time_real) * sec_per_q
            if dt < REUSE_LIMIT_S:
                viol += 1
        viol_total += viol
        if viol:
            print("   %s %s: %d fast same-finger reuses" % (label, side, viol))
    print("   A1.3 fast same-finger violations: %d" % viol_total)
    check(viol_total == 0, "A1.3: %d fast same-finger violations" % viol_total)


def test_chord_finger_order():
    """A1.4: chord fingers rise with pitch (right) / fall (left)."""
    viol_total = n_chords = 0
    for label, side, h, _bpm in solved():
        by_chord = {}
        for x in h.noteseq:
            if x.isChord:
                by_chord.setdefault(x.chordID, []).append(x)
        for cid, members in sorted(by_chord.items()):
            if len(members) < 2:
                continue
            if any(m.fingering not in (1, 2, 3, 4, 5) for m in members):
                continue              # covered by A1.1
            n_chords += 1
            members = sorted(members, key=lambda m: _pitch_key(m, side))
            fings = [m.fingering for m in members]
            if side == "right":
                ok = all(b > a for a, b in zip(fings, fings[1:]))
            else:
                ok = all(b < a for a, b in zip(fings, fings[1:]))
            if not ok:
                viol_total += 1
                if viol_total <= 5:
                    print("   %s %s chord %s: pitch-ordered fingers %s"
                          % (label, side, cid, fings))
    print("   A1.4 chord-order violations: %d / %d chords" % (viol_total, n_chords))
    check(viol_total == 0, "A1.4: %d chords with wrong finger order"
          % viol_total)


def test_solve_500_notes_fast():
    """A1.5: a 500-note score solves in < 2 s."""
    scale = ["C", "D", "E", "F", "G", "A", "B"]
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=120))
    for i in range(500):
        name = "%s%d" % (scale[i % 7], 4 + (i // 7) % 2)
        p.insert(i * 0.25, note.Note(name, quarterLength=0.5))
    s = stream.Score()
    s.insert(0, p)
    t0 = time.perf_counter()
    h = make_hand(s)
    dt = time.perf_counter() - t0
    print("   A1.5 solved 500 notes in %.3f s" % dt)
    check(len(h.noteseq) == 500, "expected 500 notes, got %d" % len(h.noteseq))
    check(dt < 2.0, "500-note solve took %.3f s (>= 2 s)" % dt)


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
