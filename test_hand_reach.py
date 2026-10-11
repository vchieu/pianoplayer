"""
Hand-reach tests (Phase A2 of PIANOPLAYER_FIX_PLAN.md).

For every hand size XXS..XXL and every finger 1..5, uses HandRig.settle
to place the finger on white key C4 and on black key C#4, then checks:

    tip_err  < 0.05 cm   (true q[3] vs target)
    max_bend < 1.3  rad  (PIP bend angle b1)

Needs the A2 hooks: solve_chain returns (pts, b1); HandRig.update stores
self.tip_err[f] / self.max_bend[f].

NOTE: the thresholds may not hold for all sizes yet (fix D1 scales the
root per hand size). The test prints a per-size table either way and
asserts strictly — do NOT weaken the thresholds; record failures in the
plan's Deviations instead.

    python test_hand_reach.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import note

from pianoplayer.hand import Hand
from pianoplayer.hand3d import FINGERS, HandRig
from pianoplayer.utils import keypos

FAILS = []

TIP_ERR_MAX = 0.05    # cm
BEND_MAX = 1.3        # rad
SIZES = ("XXS", "XS", "S", "M", "L", "XL", "XXL")

X_WHITE = float(keypos(note.Note("C4")))
X_BLACK = float(keypos(note.Note("C#4")))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


_FREST = {}    # size -> Hand.frest (rest-shape fan offsets)


def _frest(size):
    h = _FREST.get(size)
    if h is None:
        h = Hand("right", size)
        h.verbose = False
        _FREST[size] = h
    return h.frest


def measure(size, finger, x):
    """Settle `finger` (fully pressed) onto key x and return (tip_err, bend).

    The other four fingers ride along in their rest-shape offsets, exactly
    as in real playback where settle() gets a whole fan (notes[0].fx):
    the rig root chases the mean of all five targets, so a lone target
    would park the root in the middle of nowhere and prove nothing.
    """
    frest = _frest(size)
    fan = {f: x + (frest[f] - frest[finger]) for f in FINGERS}
    rig = HandRig("right", size)
    rig.settle(fan, {finger: 1.0})
    check(hasattr(rig, "tip_err") and hasattr(rig, "max_bend"),
          "A2 hooks missing: HandRig has no tip_err/max_bend")
    check(finger in rig.tip_err and finger in rig.max_bend,
          "size %s finger %d: no tip_err/max_bend recorded" % (size, finger))
    return float(rig.tip_err[finger]), float(rig.max_bend[finger])


def test_hooks_present():
    """The A2 hooks exist and return sane values."""
    pts_b1 = None
    from pianoplayer.hand3d import solve_chain
    pts_b1 = solve_chain((0, 0, 0), (0, 5, 0), [4.3, 2.7, 2.1], (0, 0.35, 1.0))
    check(isinstance(pts_b1, tuple) and len(pts_b1) == 2,
          "solve_chain must return (pts, b1), got %r" % type(pts_b1))
    pts, b1 = pts_b1
    check(len(pts) == 4, "solve_chain pts must have 4 joints")
    check(all(v == v and abs(v) != float("inf") for v in (b1,)),
          "b1 must be finite")
    rig = HandRig("right", "M")
    rig.settle({3: X_WHITE}, {3: 1.0})
    for f in FINGERS:
        check(f in rig.tip_err and f in rig.max_bend,
              "finger %d missing from tip_err/max_bend" % f)
        check(rig.tip_err[f] >= 0.0 and rig.tip_err[f] == rig.tip_err[f],
              "tip_err[%d] not a sane number" % f)
    print("   hooks ok: solve_chain -> (pts[4], b1=%.3f); "
          "tip_err/max_bend populated for all fingers" % b1)


def test_hand_reach():
    """tip_err < 0.05 cm and max_bend < 1.3 rad for all sizes/fingers/keys."""
    rows = []       # (size, finger, key, tip_err, bend, ok)
    for size in SIZES:
        for finger in FINGERS:
            for key, x in (("C4-white", X_WHITE), ("C#4-black", X_BLACK)):
                err, bend = measure(size, finger, x)
                ok = err < TIP_ERR_MAX and bend < BEND_MAX
                rows.append((size, finger, key, err, bend, ok))

    print("   %8s %6s %-9s %10s %10s  %s"
          % ("size", "finger", "key", "tip_err", "max_bend", "ok"))
    bad = []
    for size, finger, key, err, bend, ok in rows:
        print("   %8s %6d %-9s %10.4f %10.4f  %s"
              % (size, finger, key, err, bend, "ok" if ok else "FAIL"))
        if not ok:
            bad.append((size, finger, key))
    n_bad = len(bad)
    print("   reach: %d/%d cells pass (tip_err<%.2f, max_bend<%.2f)"
          % (len(rows) - n_bad, len(rows), TIP_ERR_MAX, BEND_MAX))
    if bad:
        seen = {}
        for size, finger, key in bad:
            seen.setdefault(size, []).append("%d/%s" % (finger, key))
        for size in SIZES:
            if size in seen:
                print("   failing %s: %s" % (size, ", ".join(seen[size])))
    check(n_bad == 0, "%d/%d size/finger/key cells exceed tip_err/max_bend "
          "(see table above; fix D1)" % (n_bad, len(rows)))


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
