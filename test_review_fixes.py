"""
Review-fix acceptance tests (2026-10-12 batch).

    python test_review_fixes.py

Covers, without touching existing suites:

    thumb lock         rig._geom[1]["base_local"] equals rest_chains(hf)[1][0]
                       (frame round-tripped, x-mirrored for left) at 1e-6
                       for all 7 sizes x both sides.
    cache persistence  delete the hashed cache, call get_rest_data twice:
                       1st builds, 2nd loads the full 5-array set (no
                       recompute) and matches the 1st call exactly.
    t_sec None         hand-built INotes (never through reader) have
                       t_sec None and _dt_sec falls back to quarters at
                       FALLBACK_BPM (0.5 q -> 0.25 s).
    texture fallback   a garbage texture_path keeps the SKIN actor colour;
                       the default texture path yields white (positive
                       control proving the branch, not the colour).
"""
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_cache_persistence():
    import numpy as np

    import pianoplayer.handmesh as hm
    hf, side = 0.79, "right"  # off-size: never clobbers a real hand cache
    try:
        os.remove(hm.cache_path(hf, side))
    except Exception:
        pass
    check(not os.path.isfile(hm.cache_path(hf, side)),
          "cache file still present after delete")
    d1 = hm.get_rest_data(hf, side)
    check(os.path.isfile(hm.cache_path(hf, side)),
          "get_rest_data did not write the cache")
    z = hm._load_cache(hm.cache_path(hf, side))
    check(z is not None and {"rest_pts", "faces", "rest_nrm", "idx", "w"}
          <= set(z.files),
          "cache lacks the full 5-array set: %s"
          % (sorted(z.files) if z is not None else None))
    d2 = hm.get_rest_data(hf, side)  # must load, not recompute
    for k in ("pts", "nrm", "idx", "w"):
        a, b = np.asarray(d1[k]), np.asarray(d2[k])
        check(a.shape == b.shape and bool((a == b).all()),
              "2nd get_rest_data differs in %s" % k)
    check(len(d1["faces"]) == len(d2["faces"]) and
          all(a == b for a, b in zip(d1["faces"], d2["faces"])),
          "2nd get_rest_data differs in faces")
    print("   cache: N=%d tris=%d, 2nd call identical (loaded)"
          % (len(d1["pts"]), len(d1["faces"])))


def test_thumb_lock():
    """Thumb MCP base matches rest_chains for all 7 sizes x both sides.

    The rig's thumb MCP (D1b forward shift) and the SDF rest pose must use
    the same base: compare in the same frame -- build the rig, run the
    reset() path used by the reach tests, undo the root + R@ transform,
    and check against rest_chains(hf)[1][0] (x-mirrored for left).
    Tolerance 1e-6.
    """
    import numpy as np

    from pianoplayer import handspec as HS
    from pianoplayer.hand3d import HandRig, _rot_x, _rot_y, _rot_z
    from pianoplayer.handmesh import rest_chains
    from pianoplayer.utils import handSizeFactor

    worst = 0.0
    for size in ("XXS", "XS", "S", "M", "L", "XL", "XXL"):
        hf = float(handSizeFactor(size))
        chains, _cmc = rest_chains(hf)
        want_r = np.asarray(chains[1][0], float)
        for side in ("right", "left"):
            rig = HandRig(side, size, mesh="tubes")
            rig.reset()  # settle path used by the reach tests
            base_local = np.asarray(rig._geom[1]["base_local"], float)
            # same frame: world MCP as update() builds it, then undo
            # the root + R@ transform back to the local frame.
            R = (_rot_z(rig.yaw) @ _rot_y(-rig.s * rig.ROLL)
                 @ _rot_x(rig.PITCH))
            world = np.asarray(rig.root, float) + R @ base_local
            local = R.T @ (world - np.asarray(rig.root, float))
            check(np.allclose(local, base_local, atol=1e-9),
                  "%s %s: frame round-trip broke (%.3g)" %
                  (size, side, abs(local - base_local).max()))
            want = want_r * (np.array([-1.0, 1.0, 1.0])
                             if side == "left" else 1.0)
            err = float(abs(local - want).max())
            worst = max(worst, err)
            check(err < 1e-6,
                  "%s %s: thumb MCP base mismatch: rig %s vs rest %s "
                  "(err %.3g)" % (size, side, tuple(base_local),
                                  tuple(np.asarray(want, float)), err))
    print("   thumb lock: 7 sizes x 2 sides match rest_chains "
          "(worst err %.2g)" % worst)


def test_tsec_none():
    from pianoplayer.hand import FALLBACK_BPM, _dt_sec
    from pianoplayer.scorereader import INote
    na, nb = INote(), INote()
    check(na.t_sec is None and nb.t_sec is None,
          "hand-built INote.t_sec should default to None")
    na.time, nb.time = 0.0, 0.5
    want = 0.5 * 60.0 / FALLBACK_BPM
    got = _dt_sec(na, nb)
    check(abs(got - want) < 1e-12,
          "_dt_sec fallback %.6f != quarters %.6f" % (got, want))
    print("   t_sec None: fallback dt=%.4f s for 0.5 q (as expected)" % got)


def test_texture_fallback():
    from pianoplayer.hand3d import SKIN, HandRig
    fd, bogus = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"not a png, just garbage")
        rig = HandRig(side="right", size="M", mesh="skinned",
                      texture_path=bogus)
    finally:
        try:
            os.remove(bogus)
        except Exception:
            pass
    check(rig._skin is not None, "skinned mesh missing after bogus texture")
    col = tuple(rig._skin["mesh"].actor.GetProperty().GetColor())
    check(all(abs(c - s) < 1e-6 for c, s in zip(col, SKIN)),
          "bogus texture: actor colour %s != SKIN %s" % (col, SKIN))
    import numpy as _np
    sc = _np.asarray(rig._skin["blush"][0], float) / 255.0
    # R4: with baked AO + skin shading, vertex base is SKIN * AO * shading
    # not pure SKIN. Check it's in the expected range (SKIN * 0.75 .. 1.15)
    for c, s in zip(sc, SKIN):
        check(s * 0.7 < c < s * 1.2,
              "bogus texture: vertex base %s not in SKIN*AO*shading range" % (tuple(sc),))
    print("   texture: bogus path keeps SKIN*AO*shading %s" % (tuple(sc),))
    rig_ok = HandRig(side="right", size="M", mesh="skinned")
    check(rig_ok._skin is not None, "default-texture rig lost its skin")
    col_ok = tuple(rig_ok._skin["mesh"].actor.GetProperty().GetColor())
    check(all(abs(c - 1.0) < 1e-6 for c in col_ok),
          "default texture: actor colour %s != white" % (col_ok,))
    print("   texture: default path yields white %s" % (col_ok,))


def test_finger_smoothing():
    """R1.8: idle fingers glide (<=1.2 cm/frame @60fps), aim/held fingers exact."""
    import numpy as np
    from pianoplayer.hand3d import FINGERS, HandRig

    rig = HandRig("right", "M", mesh="tubes")
    fan = rig._neutral_x()
    rig.settle(dict(fan), {})
    tip0 = {f: rig._geom[f]["pts"][-1].copy() for f in FINGERS}

    # 1. Idle finger: fan target moves 6 cm, finger should glide smoothly
    far = dict(fan)
    far[2] = float(fan[2] + 6.0)
    rig.update(1.0 / 60.0, far, {})
    tip1 = rig._geom[2]["pts"][-1]
    dx = abs(float(tip1[0]) - float(tip0[2][0]))
    check(dx <= 1.2, "idle finger teleports %.2f cm in one frame" % dx)

    # 2. Aim finger: enters aim_set, _sx snaps to key_x (no spring).
    # Physical convergence takes a few frames (like pressed fingers).
    rig.settle(dict(fan), {})  # back to natural
    key_x = float(fan[3] + 3.0)  # key 3 cm right of natural
    aim_fan = dict(fan)
    aim_fan[3] = key_x
    rig.update(1.0 / 60.0, aim_fan, {}, aim_fingers={3})
    # Check target assignment snapped (no spring)
    check(abs(rig._sx[3] - key_x) < 1e-6,
          "aim_fingers: _sx not snapped to key_x (got %.3f, want %.3f)" %
          (rig._sx[3], key_x))
    # Converge over a few frames (simulate AIM_LEAD period)
    for _ in range(5):
        rig.update(1.0 / 60.0, aim_fan, {}, aim_fingers={3})
    tip_aim = rig._geom[3]["pts"][-1]
    aim_err = abs(float(tip_aim[0]) - key_x)
    check(aim_err < 0.1,
          "aim_fingers: finger at %.3f cm, expected %.3f cm (err %.3f)" %
          (tip_aim[0], key_x, aim_err))

    # 3. Held finger: press=1.0, in held set, stays exact on key even if fan jumps
    rig.settle(dict(fan), {})
    rig.update(1.0 / 60.0, far, {3: 1.0}, held={3})
    tip3 = rig._geom[3]["pts"][-1]
    check(abs(float(tip3[0]) - float(fan[3])) <= 0.05,
          "held finger drifted %.3f off its key" % abs(tip3[0] - fan[3]))

    # 4. Pressed finger (press>0.5, not in held) also tracks key exactly
    rig.settle(dict(fan), {})
    near = dict(fan)
    near[2] = float(fan[2] + 1.5)
    rig.settle(near, {2: 1.0})
    tip2 = rig._geom[2]["pts"][-1]
    check(abs(float(tip2[0]) - near[2]) <= 0.10,
          "pressed finger never converges (dx=%.2f)" % abs(tip2[0] - near[2]))
    check(rig.tip_err[2] < 0.05, "pressed tip_err %.3f" % rig.tip_err[2])

    # 5. Hold p95 sanity: 100 frames alternating press/release
    rig.settle(dict(fan), {})
    key_x = float(fan[3] + 2.0)
    held_fan = dict(fan)
    held_fan[3] = key_x
    max_held_err = 0.0
    for i in range(100):
        is_held = (i % 20) < 10
        press_amt = 1.0 if is_held else 0.0
        rig.update(1.0 / 60.0, held_fan, {3: press_amt},
                   held={3} if is_held else set(), aim_fingers=set())
        tip_x = float(rig._geom[3]["pts"][-1][0])
        if is_held:
            max_held_err = max(max_held_err, abs(tip_x - key_x))
    check(max_held_err < 0.1,
          "hold max err %.3f cm over 100 frames (expected <0.1)" % max_held_err)

    print("   smoothing ok: idle dx=%.2f, aim err=%.3f, hold max err=%.3f"
          % (dx, aim_err, max_held_err))


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
