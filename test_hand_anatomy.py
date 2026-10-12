"""Anatomical proportion tests (Phase R, R0/R2).

    python test_hand_anatomy.py

On the unmodified tree these assertions are EXPECTED to fail (BUG-1/2/3
baseline); do NOT weaken the thresholds, record misses as Deviations.
Post-R2 all must pass.  Set PIANOPLAYER_SKIP_SDF=1 to skip the (slower)
SDF-level metrics.
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from diag_hand_anatomy import TARGETS, measure_all, table_text

FAILS = []
SKIP_SDF = bool(os.environ.get("PIANOPLAYER_SKIP_SDF"))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


_M = None


def metrics():
    global _M
    if _M is None:
        _M = measure_all(with_sdf=not SKIP_SDF)
        print(table_text(_M))
    return _M


def test_wrist_profile():
    m = metrics()
    check(m["mono_ok"], "BUG-1: width profile not monotonic: %s"
          % m["mono_msg"])
    check(TARGETS["wrist_palm_min"] <= m["wrist_palm"]
          <= TARGETS["wrist_palm_max"],
          "BUG-1: wrist/palm %.3f outside [%.2f, %.2f]"
          % (m["wrist_palm"], TARGETS["wrist_palm_min"],
             TARGETS["wrist_palm_max"]))


def test_thumb_proportion():
    m = metrics()
    check(TARGETS["thumb_index_min"] <= m["thumb_index"]
          <= TARGETS["thumb_index_max"],
          "BUG-2: thumb/index %.3f outside [%.3f, %.2f]"
          % (m["thumb_index"], TARGETS["thumb_index_min"],
             TARGETS["thumb_index_max"]))


def test_finger_spacing():
    m = metrics()
    for v in m["pip_spacing"]:
        check(TARGETS["pip_spacing_min"] <= v <= TARGETS["pip_spacing_max"],
              "BUG-3: PIP diam/spacing %.3f outside [%.2f, %.2f]"
              % (v, TARGETS["pip_spacing_min"], TARGETS["pip_spacing_max"]))
    check(TARGETS["gap50_min"] <= m["gap50_min"] <= TARGETS["gap50_max"],
          "BUG-3: gap@50%% %.3f outside [%.2f, %.2f]"
          % (m["gap50_min"], TARGETS["gap50_min"], TARGETS["gap50_max"]))


def test_webbing_components():
    if SKIP_SDF:
        print("   (SDF metrics skipped via PIANOPLAYER_SKIP_SDF)")
        return
    m = metrics()
    check(m["cc50"] == TARGETS["cc50"],
          "webbing: CC@50%% = %d, want %d" % (m["cc50"], TARGETS["cc50"]))
    check(m["cc85"] >= TARGETS["cc85_min"],
          "BUG-3: CC@85%% = %d, want >= %d (tips fused)"
          % (m["cc85"], TARGETS["cc85_min"]))
    check(m["profile_max_err"] <= TARGETS["meas_profile_tol"],
          "SDF half-width deviates %.3f from spec (tol %.2f)"
          % (m["profile_max_err"], TARGETS["meas_profile_tol"]))
    check(m["profile_mono_ok"], "measured SDF profile not monotonic")


def test_skinning_leakage():
    """R2.7: verts near finger i get <=5% weight from finger j != i."""
    import numpy as np
    from pianoplayer.handmesh import get_rest_data
    d = get_rest_data(0.82, "right")
    names = d["names"]
    idx = np.asarray(d["idx"])
    w = np.asarray(d["w"])
    pts = np.asarray(d["pts"])

    def finger_of(nm):
        if len(nm) == 4 and nm[0] == "f" and nm[2] == "_" and \
                nm[1] in "12345" and nm[3] in "012":
            return int(nm[1])
        return None

    f_of_bone = [finger_of(nm) for nm in names]
    y_kn = 8.2 * 0.82
    in_web = (pts[:, 1] >= y_kn - 1.6 * 0.82) & \
        (pts[:, 1] <= y_kn + 1.4 * 0.82)
    worst = 0.0
    n_checked = 0
    for i in range(len(pts)):
        bi = int(idx[i, 0])
        fi = f_of_bone[bi]
        if fi is None or in_web[i]:
            continue
        n_checked += 1
        leak = 0.0
        for k in range(4):
            fj = f_of_bone[int(idx[i, k])]
            if fj is not None and fj != fi:
                leak += float(w[i, k])
        worst = max(worst, leak)
    check(n_checked > 100, "too few finger verts to grade (%d)" % n_checked)
    check(worst <= 0.05,
          "skinning leaks %.1f%% across fingers (target <= 5%%)"
          % (100.0 * worst))
    print("   leakage ok: worst %.2f%% over %d verts" % (100 * worst,
                                                        n_checked))


def test_finger_smoothing():
    """R1.8: idle fingers glide (<=1.2 cm/frame @60fps), held fingers exact."""
    from pianoplayer.hand3d import FINGERS, HandRig
    rig = HandRig("right", "M", mesh="tubes")
    fan = rig._neutral_x()
    rig.settle(dict(fan), {})
    tip0 = {f: rig._geom[f]["pts"][-1].copy() for f in FINGERS}
    far = dict(fan)
    far[2] = float(fan[2] + 6.0)
    rig.update(1.0 / 60.0, far, {})
    tip1 = rig._geom[2]["pts"][-1]
    dx = abs(float(tip1[0]) - float(tip0[2][0]))
    check(dx <= 1.2, "idle finger teleports %.2f cm in one frame" % dx)
    # convergence on a reachable jump (+6 cm lateral exceeds one finger's
    # IK range no matter the smoothing, so use +1.5 cm here)
    rig.settle(dict(fan), {})
    near = dict(fan)
    near[2] = float(fan[2] + 1.5)
    rig.settle(near, {2: 1.0})
    tip2 = rig._geom[2]["pts"][-1]
    check(abs(float(tip2[0]) - near[2]) <= 0.10,
          "pressed finger never converges (dx=%.2f)" % abs(tip2[0] - near[2]))
    check(rig.tip_err[2] < 0.05, "pressed tip_err %.3f" % rig.tip_err[2])
    # held finger keeps exactly its key x even when the fan jumps
    rig.settle(dict(fan), {})
    rig.update(1.0 / 60.0, far, {3: 1.0}, held={3})
    tip3 = rig._geom[3]["pts"][-1]
    check(abs(float(tip3[0]) - float(fan[3])) <= 0.05,
          "held finger drifted %.3f off its key" % abs(tip3[0] - fan[3]))
    print("   smoothing ok: 1-frame dx=%.2f, held exact" % dx)


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
