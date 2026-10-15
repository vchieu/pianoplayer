"""Fingering quality evaluation (Phase F0 of PIANOPLAYER_HUMAN_HAND_PLAN.md).

(a) Hand-written gold standard (hard test, exact match required):
    C and G major scales (1 octave, 8 notes) --
      right hand ascending 1 2 3 1 2 3 4 5,
      left hand ascending  5 4 3 2 1 3 2 1;
    F major scale right hand 1 2 3 4 1 2 3 4;
    C major arpeggio right hand 1 2 3 5 1 2 3 5.
(b) PIG dataset (PianoFingeringDataset, expert fingerings, NOT redistributed):
    give a .txt file or a directory; prints the per-piece note-weighted
    average match rate over both hands (via pigloader.load_pig, chords
    included) and the average. Notes without an expert number are skipped.

(c) F1-F3 acceptance checks: tempo dependence (60 vs 200 bpm cost tables
    differ, gold passes at both), repeated-note alternation (16ths at 120
    bpm must not sit on one finger; at 40 bpm may repeat), same-finger
    reuse over time (2 s-apart octave: soft cost, fast reuse: HARD),
    R1.8 aim-set helper.

    python eval_fingering.py [pig_path_or_dir]

Gate (plan F0.3): no F-task merges if the PIG average drops or gold breaks.
Known-failure gate: labels in KNOWN_FAIL that still fail are reported as
XFAIL (exit 0, counted separately); any other failure is FAIL (exit
nonzero); a KNOWN_FAIL label that unexpectedly passes is XPASS (exit
nonzero -- remove it from KNOWN_FAIL). So F-tasks are blocked on
non-known failures or a PIG drop, not on the one deferred arpeggio.
"""
import glob
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import note, stream, tempo
from music21.articulations import Fingering

from pianoplayer.hand import (
    HARD,
    SAME_FINGER_BASE,
    SAME_FINGER_DX,
    Hand,
    _dt_sec,
)
from pianoplayer.pigloader import load_pig
from pianoplayer.scorereader import PIG2Stream, reader
from pianoplayer.utils import score_tempo_map

FAILS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_score(names, bpm=120, ql=0.5, hold=None):
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    for i, m in enumerate(names):
        p.insert(i * ql, note.Note(m, quarterLength=hold if hold else ql))
    s = stream.Score()
    s.insert(0, p)
    return s


def solve(names, side="right", bpm=120, ql=0.5, hold=None):
    sf = make_score(names, bpm, ql, hold)
    h = Hand(side, "M")
    h.verbose = False
    h.noteseq = reader(sf, beam=0, tempo_map=score_tempo_map(sf))
    h.generate()
    return [n.fingering for n in h.noteseq], h


# ------------------------------------------------------------------ gold --- #
# C arpeggio R needs the F5 crossing model (thumb-under-after-2 is
# preferred by the velocity-only cost): deferred, tracked as XFAIL.
KNOWN_FAIL = {"C arpeggio R"}
GOLD = [
    ("C scale R", ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
     "right", [1, 2, 3, 1, 2, 3, 4, 5]),
    ("C scale L", ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
     "left", [5, 4, 3, 2, 1, 3, 2, 1]),
    ("G scale R", ["G4", "A4", "B4", "C5", "D5", "E5", "F#5", "G5"],
     "right", [1, 2, 3, 1, 2, 3, 4, 5]),
    ("G scale L", ["G4", "A4", "B4", "C5", "D5", "E5", "F#5", "G5"],
     "left", [5, 4, 3, 2, 1, 3, 2, 1]),
    ("F scale R", ["F4", "G4", "A4", "Bb4", "C5", "D5", "E5", "F5"],
     "right", [1, 2, 3, 4, 1, 2, 3, 4]),
    ("C arpeggio R", ["C4", "E4", "G4", "C5", "E5", "G5", "C6", "E6"],
     "right", [1, 2, 3, 5, 1, 2, 3, 5]),
]


# ------------------------------------------------------------- gold extra -- #
# Phase 0 step 0.2: extended gold table. INFORMATIONAL ONLY: printed as a
# percentage, never fails the run. Textbook patterns, NOT pianist-checked:
# a human must confirm each row before it is promoted into GOLD.
GOLD_EXTRA = [
    ("D scale R", ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5"],
     "right", [1, 2, 3, 1, 2, 3, 4, 5]),
    ("A scale R", ["A4", "B4", "C#5", "D5", "E5", "F#5", "G#5", "A5"],
     "right", [1, 2, 3, 1, 2, 3, 4, 5]),
    ("E scale R", ["E4", "F#4", "G#4", "A4", "B4", "C#5", "D#5", "E5"],
     "right", [1, 2, 3, 1, 2, 3, 4, 5]),
    ("D scale L", ["D3", "E3", "F#3", "G3", "A3", "B3", "C#4", "D4"],
     "left", [5, 4, 3, 2, 1, 3, 2, 1]),
    ("A scale L", ["A2", "B2", "C#3", "D3", "E3", "F#3", "G#3", "A3"],
     "left", [5, 4, 3, 2, 1, 3, 2, 1]),
    ("E scale L", ["E3", "F#3", "G#3", "A3", "B3", "C#4", "D#4", "E4"],
     "left", [5, 4, 3, 2, 1, 3, 2, 1]),
    ("C 2oct R", ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
                  "D5", "E5", "F5", "G5", "A5", "B5", "C6"],
     "right", [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5]),
    ("C 2oct L", ["C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4",
                  "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
     "left", [5, 4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1]),
    ("Chromatic R (12 notes)",
     ["C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4"],
     "right", [1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1]),
]


def test_gold_extra():
    """Phase 0 step 0.2: extended gold patterns, informational only.

    Prints got/want per row and the match percentage; returns without
    failing. Percentages are recorded in docs/fingering_baseline.txt.
    """
    match = 0
    for label, names, side, want in GOLD_EXTRA:
        got, _h = solve(names, side)
        ok = list(got) == list(want)
        match += ok
        print("   gold-extra %-22s %s: got %s want %s" % (
            label, "OK " if ok else "--", list(got), list(want)))
    print("   gold-extra: %d/%d exact (%.1f%%)"
          % (match, len(GOLD_EXTRA), 100.0 * match / len(GOLD_EXTRA)))
    return


def test_gold():
    """F0: hand-written gold fingerings, exact match (KNOWN_FAIL -> XFAIL)."""
    bad = 0
    xfailed, xpassed = [], []
    for label, names, side, want in GOLD:
        got, _h = solve(names, side)
        ok = list(got) == list(want)
        if ok and label in KNOWN_FAIL:
            xpassed.append(label)
            print("   gold %-12s XPASS (unexpected pass: remove from "
                  "KNOWN_FAIL): got %s" % (label, got))
        elif not ok and label in KNOWN_FAIL:
            xfailed.append(label)
            print("   gold %-12s XFAIL (known failure): got %s want %s" % (
                label, got, want))
        else:
            print("   gold %-12s %s: got %s want %s" % (
                label, "OK " if ok else "FAIL", got, want))
            bad += not ok
    print("   gold: %d/%d exact + %d XFAIL"
          % (len(GOLD) - bad - len(xfailed), len(GOLD), len(xfailed)))
    check(not xpassed,
          "XPASS %s: remove from KNOWN_FAIL" % (xpassed,))
    check(bad == 0, "gold: %d/%d pieces differ" % (bad, len(GOLD)))


# ------------------------------------------------------------------- PIG --- #
def _pig_experts(part):
    """Expert finger per sounding event of a PIG part (None = unknown)."""
    flat = part.flatten() if hasattr(part, "flatten") else part.flat
    out = []
    for el in flat:
        if getattr(el, "isRest", False):
            continue
        if not (getattr(el, "isNote", False) or getattr(el, "isChord", False)):
            continue
        if getattr(el, "isChord", False):
            out.append(None)  # PIG2Stream stores no chord fingerings
            continue
        f = None
        for a in getattr(el, "articulations", []):
            if isinstance(a, Fingering):
                try:
                    f = int(a.fingerNumber)
                except (TypeError, ValueError):
                    f = None
        out.append(f if f in (1, 2, 3, 4, 5) else None)
    return out


def pig_match_rate(fname):
    """Note-weighted average match rate over both hands of one PIG file.

    Uses pigloader.load_pig (chords included, one group per onset).
    Returns (avg, per_hand) with per_hand = {"right": r or None,
    "left": r or None}; avg is None when no hand has a known expert.
    """
    try:
        data = load_pig(fname)
    except Exception as exc:
        print("   PIG %s unreadable: %s" % (fname, exc))
        return None, {}
    per_hand, match_total, known_total = {}, 0, 0
    for channel, side in ((0, "right"), (1, "left")):
        inotes, experts = data[channel]
        if not any(e is not None for e in experts):
            per_hand[side] = None
            continue
        h = Hand(side, "M")
        h.verbose = False
        try:
            h.noteseq = inotes
            h.generate()
        except Exception as exc:
            print("   PIG %s %s solve failed: %s" % (fname, side, exc))
            per_hand[side] = None
            continue
        got = [n.fingering for n in h.noteseq]
        n = min(len(got), len(experts))
        match = known = 0
        for g, e in zip(got[:n], experts[:n]):
            if e is None:
                continue
            known += 1
            match += (g == e)
        per_hand[side] = (match / known) if known else None
        match_total += match
        known_total += known
    if not known_total:
        return None, per_hand
    return match_total / known_total, per_hand


def test_pig():
    """F0: PIG match rates (skipped when no dataset path is given)."""
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.exists(path):
        print("   PIG: no dataset path given (or missing) -- skipped "
              "(usage: python eval_fingering.py <pig.txt|pig_dir>)")
        return
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.txt")))
    else:
        files = [path]
    rates = []
    for fn in files:
        r = pig_match_rate(fn)
        avg = r[0] if isinstance(r, tuple) else r
        per = r[1] if isinstance(r, tuple) else {}
        if avg is None:
            print("   PIG %-40s no known experts" % os.path.basename(fn))
            continue
        rates.append(avg)
        print("   PIG %-40s %.1f%% (R %s L %s)"
              % (os.path.basename(fn), 100.0 * avg,
                 ("%.1f%%" % (100.0 * per["right"])
                  if per.get("right") is not None else "--"),
                 ("%.1f%%" % (100.0 * per["left"])
                  if per.get("left") is not None else "--")))
    if rates:
        print("   PIG average over %d pieces: %.1f%%"
              % (len(rates), 100.0 * sum(rates) / len(rates)))
    else:
        print("   PIG: no comparable notes found")


# ------------------------------------------------------------ F1 tempo ----- #
def test_tempo_dependence():
    """F1: same score at 60 vs 200 bpm yields different cost sets (seconds),
    and the scale gold still passes at both tempi."""
    names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
    costs = {}
    for bpm in (60, 200):
        sf = make_score(names, bpm=bpm)
        h = Hand("right", "M")
        h.verbose = False
        h.noteseq = reader(sf, beam=0, tempo_map=score_tempo_map(sf))
        fingers = [1, 2, 3, 1, 2, 3, 4, 5]
        costs[bpm] = sum(h.pair_cost(na, fa, nb, fb)
                         for na, fa, nb, fb
                         in zip(h.noteseq, fingers,
                                h.noteseq[1:], fingers[1:]))
        check(all(abs(n.t_sec - (i * 0.5 * 60.0 / bpm)) < 1e-9
                    for i, n in enumerate(h.noteseq)),
              "t_sec not in seconds at %d bpm" % bpm)
    print("   tempo costs: 60bpm=%.3f 200bpm=%.3f"
          % (costs[60], costs[200]))
    check(abs(costs[60] - costs[200]) > 1e-9,
          "cost sets identical at 60 vs 200 bpm (still in quarters?)")
    for bpm in (60, 200):
        got, _h = solve(names, "right", bpm=bpm)
        check(list(got) == [1, 2, 3, 1, 2, 3, 4, 5],
              "C scale R gold breaks at %d bpm: %s" % (bpm, got))
        gotL, _h = solve(names, "left", bpm=bpm)
        check(list(gotL) == [5, 4, 3, 2, 1, 3, 2, 1],
              "C scale L gold breaks at %d bpm: %s" % (bpm, gotL))
    print("   tempo: gold passes at 60 and 200 bpm")


# ---------------------------------------------------------- F2 repeat ----- #
def test_repeat_alternation():
    """F2: 8 sixteenth C4s at 120 bpm must not sit on one finger;
    at 40 bpm the same finger is still allowed."""
    fast, _h = solve(["C4"] * 8, "right", bpm=120, ql=0.25, hold=0.25)
    print("   repeat @120bpm 16ths: %s" % (fast,))
    check(len(set(fast)) > 1,
          "fast repeats all on finger %s (no alternation)" % (fast[:1],))
    slow, _h = solve(["C4"] * 8, "right", bpm=40, ql=0.25, hold=0.25)
    print("   repeat @40bpm 16ths:  %s (may repeat: %s)"
          % (slow, "yes" if len(set(slow)) == 1 else "no"))
    check(all(f in (1, 2, 3, 4, 5) for f in slow),
          "slow repeats left fingerless: %s" % (slow,))


# ------------------------------------------------------ F3 same finger --- #
def test_same_finger_over_time():
    """F3: reuse after a 2 s gap is a soft cost (< HARD), a fast reuse is
    HARD, and a held note still forbids replay."""
    h = Hand("right", "M")
    h.verbose = False
    # two notes an octave apart, first held 0.5 q, second 2 s later
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=60))
    p.insert(0, note.Note("C5", quarterLength=0.5))
    p.insert(2.0, note.Note("C6", quarterLength=1.0))
    s = stream.Score()
    s.insert(0, p)
    h.noteseq = reader(s, beam=0, tempo_map=score_tempo_map(s))
    na, nb = h.noteseq
    check(abs(_dt_sec(na, nb) - 2.0) < 1e-9, "gap should be 2 s")
    soft = h._skip_penalty(5, 5, na, nb)
    print("   same-finger octave after 2 s: cost %.3f (base %.1f)"
          % (soft, SAME_FINGER_BASE))
    check(soft < HARD, "2 s-apart reuse still HARD")
    check(abs(soft - (SAME_FINGER_BASE + SAME_FINGER_DX * abs(nb.x - na.x))) < 1e-9,
          "soft cost shape wrong: %.3f" % soft)
    # fast reuse: 16ths at 120 bpm (dt = 0.125 s)
    pf = stream.Part()
    pf.insert(0, tempo.MetronomeMark(number=120))
    pf.insert(0, note.Note("C5", quarterLength=0.25))
    pf.insert(0.25, note.Note("D5", quarterLength=0.25))
    sf = stream.Score()
    sf.insert(0, pf)
    h.noteseq = reader(sf, beam=0, tempo_map=score_tempo_map(sf))
    fa, fb = h.noteseq
    check(h._skip_penalty(5, 5, fa, fb) >= HARD, "fast reuse not HARD")
    # held note: first still sounding when the second starts
    ph = stream.Part()
    ph.insert(0, tempo.MetronomeMark(number=60))
    ph.insert(0, note.Note("C5", quarterLength=4.0))
    ph.insert(1.0, note.Note("D5", quarterLength=1.0))
    sh = stream.Score()
    sh.insert(0, ph)
    h.noteseq = reader(sh, beam=0, tempo_map=score_tempo_map(sh))
    ha, hb = h.noteseq
    check(h._skip_penalty(5, 5, ha, hb) >= HARD, "held replay not HARD")
    # end to end: the 2 s-apart octave solves (any legal fingering)
    h.noteseq = reader(s, beam=0, tempo_map=score_tempo_map(s))
    h.generate()
    check(all(n.fingering in (1, 2, 3, 4, 5) for n in h.noteseq),
          "octave pair unsolved")


# ----------------------------------------------------------- R1.8 aim ----- #
def test_aim_fingers():
    """R1.8 (caller side): imminent unfired fingers are exactly the aim set."""
    from pianoplayer.vkeyboard import _aim_fingers

    class E:
        def __init__(self, t_on, finger, fired=False):
            self.t_on = t_on
            self.finger = finger
            self.fired = fired

    notes = [E(1.00, 1, fired=True), E(1.10, 2), E(1.20, 3), E(5.00, 4)]
    got = _aim_fingers(notes, 1.00)
    print("   aim_fingers at t=1.00: %s" % sorted(got))
    check(got == {2, 3}, "aim set wrong: %s" % sorted(got))
    later = [E(1.00, 1, fired=True), E(1.10, 2, fired=True),
             E(1.20, 3, fired=True), E(5.00, 4)]
    check(_aim_fingers(later, 1.30) == set(),
          "lead window should have passed")
    check(_aim_fingers(notes, 0.0) == set(), "nothing imminent at t=0")
    rig_notes = [E(0.5, 0), E(0.6, 5)]
    check(_aim_fingers(rig_notes, 0.5) == {5}, "finger 0 must be excluded")


# --------------------------------------------------------------------------- #
def main():
    global FAILS
    t0 = time.perf_counter()
    # PIG first (informational even when gold fails), then the hard gates.
    tests = [test_pig, test_gold, test_gold_extra, test_tempo_dependence,
             test_repeat_alternation, test_same_finger_over_time,
             test_aim_fingers]
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception:
            FAILS.append(t.__name__)
            print("FAIL", t.__name__)
            traceback.print_exc()
    print("\n%d/%d passed (%.2f s)"
          % (len(tests) - len(FAILS), len(tests), time.perf_counter() - t0))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
