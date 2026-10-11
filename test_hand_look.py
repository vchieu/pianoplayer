"""
Hand-look acceptance test (Tier A H0 table).

    python test_hand_look.py

Measures the fixed mid/ends poses (see diag_hand_look.py) and asserts the
H0 thresholds. The table is printed whether it passes or not -- on the
unmodified tree the tilt/gap/ratio/bend assertions are EXPECTED to fail;
that failure is the H0 baseline, not a regression. Do NOT weaken the
thresholds; record misses as Deviations instead.
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from diag_hand_look import TARGETS, measure_all, table_text

FAILS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_hand_look():
    m = measure_all(write_files=False)
    print(table_text(m))
    check(m["tilt_mid_max"] <= TARGETS["tilt_mid_max"],
          "tilt_mid_max %.2f > %.1f"
          % (m["tilt_mid_max"], TARGETS["tilt_mid_max"]))
    check(m["tilt_ends_max"] <= TARGETS["tilt_ends_max"],
          "tilt_ends_max %.2f > %.1f"
          % (m["tilt_ends_max"], TARGETS["tilt_ends_max"]))
    check(m["gap_max"] == TARGETS["gap_components"],
          "gap_max %d != %d" % (m["gap_max"], TARGETS["gap_components"]))
    check(m["ratio_min"] >= TARGETS["finger_proj_ratio_min"],
          "proj_ratio_min %.3f < %.2f"
          % (m["ratio_min"], TARGETS["finger_proj_ratio_min"]))
    check(m["bend_max"] <= TARGETS["idle_max_bend_max"],
          "idle_bend_max %.3f > %.2f"
          % (m["bend_max"], TARGETS["idle_max_bend_max"]))


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
