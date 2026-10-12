"""
Review-fix acceptance tests (2026-10-13 batch).

    python test_review_fixes.py

Covers, without touching existing suites:

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
    check(all(abs(c - s) < 2e-3 for c, s in zip(sc, SKIN)),
          "bogus texture: vertex base %s != SKIN %s" % (tuple(sc), SKIN))
    print("   texture: bogus path keeps SKIN %s" % (col,))
    rig_ok = HandRig(side="right", size="M", mesh="skinned")
    check(rig_ok._skin is not None, "default-texture rig lost its skin")
    col_ok = tuple(rig_ok._skin["mesh"].actor.GetProperty().GetColor())
    check(all(abs(c - 1.0) < 1e-6 for c in col_ok),
          "default texture: actor colour %s != white" % (col_ok,))
    print("   texture: default path yields white %s" % (col_ok,))


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
