"""R0 anatomical measurement tools (Phase R).

Measures, at size M (hf=0.82) unless noted:

- width profile along y (monotonic both ways from the wrist + wrist/palm);
- finger PIP diameter / MCP spacing (no overlap);
- gap at 50% of the proximal phalanx (target [0, 0.15] cm);
- thumb (MCP->tip) / index ratio;
- connected components of the SDF cross-section at 50% proximal (=1,
  webbing bridges) and at 85% finger length (>=2, tips separate);
- measured SDF half-widths at the profile stations vs handspec.

    python diag_hand_anatomy.py [--hf 0.82] [--no-sdf]

Spec-level metrics are pure numpy (fast).  SDF-level metrics evaluate the
handmesh field on two y slices (~seconds, no mesh extraction).

On the unmodified tree the assertions are EXPECTED to fail (BUG-1/2/3
baseline); do NOT weaken the thresholds, record misses as Deviations.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HF_M = 0.82

TARGETS = {
    "wrist_palm_min": 0.68,
    "wrist_palm_max": 0.80,
    "thumb_index_min": 0.585,   # 0.65 - 10%
    "thumb_index_max": 0.88,    # 0.80 + 10%
    "pip_spacing_min": 0.85,
    "pip_spacing_max": 1.00,
    "gap50_min": 0.0,
    "gap50_max": 0.15,
    "cc50": 1,
    "cc85_min": 2,
    "meas_profile_tol": 0.35,   # |measured - spec| half-width [cm]
}


def spec_metrics(hf=HF_M):
    """Fast spec-level metrics from handspec (no SDF evaluation)."""
    from pianoplayer import handspec as H

    mono_ok, mono_msg = H.check_palm_monotonic()
    wp = H.wrist_palm_ratio()
    tr = H.thumb_index_ratio()
    spac = H.mcp_spacing()
    pip = []
    gaps = []
    pairs = [(2, 3), (3, 4), (4, 5)]
    for fa, fb in pairs:
        xa = H.mcp_base(fa)[0]
        xb = H.mcp_base(fb)[0]
        sp = abs(xb - xa)
        # radius halfway along the proximal phalanx (MCP->PIP midpoint)
        ra = 0.5 * (H.finger_radius(fa, 0) + H.finger_radius(fa, 1))
        rb = 0.5 * (H.finger_radius(fb, 0) + H.finger_radius(fb, 1))
        pip.append(0.5 * (2.0 * ra / sp + 2.0 * rb / sp))
        gaps.append(sp - (ra + rb))
    return {
        "mono_ok": bool(mono_ok),
        "mono_msg": mono_msg,
        "wrist_palm": float(wp),
        "thumb_index": float(tr),
        "pip_spacing_max": float(max(pip)),
        "pip_spacing": [float(v) for v in pip],
        "gap50_min": float(min(gaps)),
        "gap50": [float(v) for v in gaps],
    }


def _label(mask):
    """4-connected components via BFS (no scipy). Returns (n, sizes)."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    sizes = []
    cur = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not lab[r, c]:
                cur += 1
                n = 0
                stack = [(r, c)]
                lab[r, c] = cur
                while stack:
                    y, x = stack.pop()
                    n += 1
                    if y > 0 and mask[y - 1, x] and not lab[y - 1, x]:
                        lab[y - 1, x] = cur
                        stack.append((y - 1, x))
                    if y + 1 < h and mask[y + 1, x] and not lab[y + 1, x]:
                        lab[y + 1, x] = cur
                        stack.append((y + 1, x))
                    if x > 0 and mask[y, x - 1] and not lab[y, x - 1]:
                        lab[y, x - 1] = cur
                        stack.append((y, x - 1))
                    if x + 1 < w and mask[y, x + 1] and not lab[y, x + 1]:
                        lab[y, x + 1] = cur
                        stack.append((y, x + 1))
                sizes.append(n)
    return cur, sizes


def sdf_metrics(hf=HF_M):
    """SDF-level metrics: CC at 50%/85% + measured width profile."""
    from pianoplayer import handspec as H
    from pianoplayer.handmesh import _build_sdf

    hf = float(hf)
    vol, (xs, ys, zs) = _build_sdf(hf)
    step = float(xs[1] - xs[0]) if xs.size > 1 else H.GRID_STEP

    from pianoplayer.handmesh import rest_chains as _rc
    chains, _ = _rc(hf)

    # x window of fingers 2-5 (±1.5 cm): the thumb reaches into the same
    # slices but is graded separately (thumb/index proportion).
    fx = np.concatenate([np.asarray(chains[f])[:, 0] for f in (2, 3, 4, 5)])
    xlo, xhi = float(fx.min()) - 1.5, float(fx.max()) + 1.5

    def mask_at(y, window=True):
        j = int(np.clip(round((y - ys[0]) / step), 0, ys.size - 1))
        m = vol[:, j, :].T <= 0.0  # (nz, nx) bool
        if window:
            keep = (xs >= xlo) & (xs <= xhi)
            m[:, ~keep] = False
        return m

    def count_cc(mask):
        # downsample like diag_hand_look, merge hairline bridges
        small = mask[::2, ::2]
        n, sizes = _label(small)
        big = [s for s in sizes if s >= 8]
        return len(big)

    # finger-2 proximal 50% and distal 85% y (right-hand local, x hf)
    q2 = np.asarray(chains[2], float)
    seg = q2[1] - q2[0]
    y50 = float(q2[0][1] + 0.5 * seg[1])
    y85 = float(q2[0][1] + 0.85 * (q2[3][1] - q2[0][1]))
    cc50 = count_cc(mask_at(y50))
    cc85 = count_cc(mask_at(y85))

    # measured +x (pinky-side) extent at the profile stations: the thumb
    # side legitimately carries the CMC/metacarpal outside the palm loft
    # and is graded separately (thumb/index proportion).  Stations past
    # y=5.8 carry the MCP mounds/fingers (graded by CC/gap instead).
    prof = []
    for (y, c, a1, _a2, _p) in H.ARM_PROFILE:
        if not (-16.0 <= y <= 5.8):
            continue
        m = mask_at(y * hf, window=False)
        cols = np.flatnonzero(m.any(axis=0))
        if len(cols) == 0:
            mx = 0.0
        else:
            mx = float(xs[0] + cols.max() * step)
        prof.append((y, float((c + a1) * hf), mx))
    prof = sorted(prof)
    err = max(abs(hw - sp) for _, sp, hw in prof) if prof else float("nan")
    # monotonic of the measured profile both ways from the wrist
    ys_p = [p[0] for p in prof]
    wi = min(range(len(ys_p)), key=lambda i: abs(ys_p[i] - H.WRIST_Y))
    mono = True
    for i in range(wi, 0, -1):
        if not prof[i - 1][2] > prof[i][2] - 1e-9:
            mono = False
    for i in range(wi, len(prof) - 1):
        if not prof[i + 1][2] > prof[i][2] - 1e-9:
            mono = False
    return {
        "cc50": int(cc50), "cc85": int(cc85), "y50": y50, "y85": y85,
        "profile": [(y, sp, hw) for (y, sp, hw) in prof],
        "profile_max_err": float(err),
        "profile_mono_ok": bool(mono),
    }


def measure_all(hf=HF_M, with_sdf=True):
    """Full table; returns metrics dict."""
    m = spec_metrics(hf)
    if with_sdf:
        m.update(sdf_metrics(hf))
    return m


def table_text(m):
    L = ["hand-anatomy metrics (size M, hf=0.82):",
         "  wrist/palm half-width = %.3f  (target [%.2f, %.2f])"
         % (m["wrist_palm"], TARGETS["wrist_palm_min"],
            TARGETS["wrist_palm_max"]),
         "  profile monotonic     = %s (%s)"
         % (m["mono_ok"], m["mono_msg"]),
         "  thumb/index length    = %.3f  (target [%.3f, %.2f])"
         % (m["thumb_index"], TARGETS["thumb_index_min"],
            TARGETS["thumb_index_max"]),
         "  PIP diam/spacing      = %s  (target [%.2f, %.2f])"
         % (" ".join("%.3f" % v for v in m["pip_spacing"]),
            TARGETS["pip_spacing_min"], TARGETS["pip_spacing_max"]),
         "  gap@50%% proximal      = %s cm  (target [%.2f, %.2f])"
         % (" ".join("%+.3f" % v for v in m["gap50"]),
            TARGETS["gap50_min"], TARGETS["gap50_max"])]
    if "cc50" in m:
        L += ["  CC@50%% proximal      = %d  (target == %d)"
              % (m["cc50"], TARGETS["cc50"]),
              "  CC@85%% length        = %d  (target >= %d)"
              % (m["cc85"], TARGETS["cc85_min"]),
              "  SDF half-width maxerr = %.3f cm (tol %.2f), mono=%s"
              % (m["profile_max_err"], TARGETS["meas_profile_tol"],
                 m["profile_mono_ok"])]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=float, default=HF_M)
    ap.add_argument("--no-sdf", action="store_true")
    args = ap.parse_args()
    m = measure_all(hf=args.hf, with_sdf=not args.no_sdf)
    print(table_text(m))
    print(json.dumps({k: v for k, v in m.items() if k != "profile"},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
