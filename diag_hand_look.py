"""
H0 tool: fixed-pose offscreen shots + mask metrics + baseline table.

    python diag_hand_look.py

Poses (size M, white keys only):
  mid  : RH C4 D4 E4 F4 G4 (fingers 1..5, thumb pressing),
         LH G2 A2 B2 C3 D3 (fingers 5/4/3/2/1, thumb pressing)
  ends : RH C7 D7 E7 F7 G7 (fingers 1..5, thumb pressing),
         LH A0 B0 C1 D1 E1 (fingers 5/4/3/2/1, thumb pressing)

Outputs in ./hand_look_out/:
  actual.png   full-window beauty shot of the mid pose (default camera/bg)
  mid_crop.png tight crop around both hands (mid pose)
  ends.png / ends_crop.png same for the ends pose
  compare.png  mid_crop + ends_crop side by side (or expected1 + mid when
               docs/expected1.png exists), with a "no expected image" note
  metrics.txt  the numbers printed below

Metrics (mask = pixels brighter than threshold with keys/background hidden
and a black background, screenshot asarray):
  tilt_deg      principal-axis (PCA) angle of each hand's mask vs vertical
                (image rows). Hands are split as the two largest connected
                components (numpy BFS, no scipy); the hand with the larger
                centroid x is RH.
  gap_components  on the horizontal slice row located half a mean finger
                bone behind the fingertip edge of each hand's bbox, count
                foreground runs left after masking out the thumb side:
                RH drops x < 30th percentile of its mask cols, LH mirrored
                (drops x > 70th percentile). A splayed thumb would otherwise
                count as a 2nd component even when the four fingers touch,
                so the target 1 applies to the 4-finger region only. Gaps
                narrower than 3 px are merged (anti-aliasing noise).
  finger_proj_ratio / max_bend read off the rigs (not the image): for each
                idle finger 2..5, ratio = |tip_y - mean_MCP_y| / bone_len
                (tip/base from the finger point buffers, bone = sum of lens
                x hf), bend = rig.max_bend[f]. Reported as min ratio / max
                bend over the 8 idle fingers of both hands.

Pose convention: rig.settle(fan, press) first (as the plan says), then
60 x update(1/60) with the same targets so the arm yaw converges to its
steady state. settle() alone zeroes the yaw afterwards, which would hide
exactly the tilt signal H1.1 is supposed to fix.

Thresholds (H0 table, also imported by test_hand_look.py):
  tilt_mid_max <= 8 deg, tilt_ends_max <= 14 deg, gap == 1,
  ratio_min >= 0.80, bend_max <= 0.9
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "hand_look_out")

TARGETS = {
    "tilt_mid_max": 8.0,
    "tilt_ends_max": 14.0,
    "gap_components": 1,
    "finger_proj_ratio_min": 0.80,
    "idle_max_bend_max": 0.9,
}

MASK_THRESH = 30       # uint8 level: skin ~200+, black bg ~0
GAP_MERGE_PX = 3       # gaps narrower than this merge into one component


def _fans():
    from music21 import note

    from pianoplayer.utils import keypos

    def xs(names):
        return [float(keypos(note.Note(n))) for n in names]

    rh_mid = dict(zip((1, 2, 3, 4, 5), xs(["C4", "D4", "E4", "F4", "G4"])))
    lh_mid = dict(zip((5, 4, 3, 2, 1), xs(["G2", "A2", "B2", "C3", "D3"])))
    rh_end = dict(zip((1, 2, 3, 4, 5), xs(["C7", "D7", "E7", "F7", "G7"])))
    lh_end = dict(zip((5, 4, 3, 2, 1), xs(["A0", "B0", "C1", "D1", "E1"])))
    return (rh_mid, lh_mid), (rh_end, lh_end)


def _build():
    from pianoplayer.hand3d import HandRig
    from pianoplayer.vkeyboard import VirtualKeyboard

    vk = VirtualKeyboard(songname="hand-look", offscreen=True)
    vk.verbose = False
    rh = HandRig(side="right", size="M")
    rh.add_to(vk.vp)
    lh = HandRig(side="left", size="M")
    lh.add_to(vk.vp)
    vk.vp.render()
    return vk, rh, lh


def _pose(rig, fan, press):
    rig.settle(dict(fan), dict(press))
    for _ in range(60):
        rig.update(1.0 / 60.0, dict(fan), dict(press))


def _hand_actor_ids(rigs):
    ids = set()
    for rig in rigs:
        for m in rig.actors():
            try:
                ids.add(int(m.actor.GetAddressAsString("")))
            except Exception:
                ids.add(id(m.actor))
    return ids


def _set_scene(vk, rigs, hands_only):
    """Show only the hands (mask shot) or everything (beauty shot)."""
    want = _hand_actor_ids(rigs)
    for a in vk.vp.get_actors():
        try:
            key = int(a.GetAddressAsString(""))
        except Exception:
            key = id(a)
        try:
            a.SetVisibility(1 if (not hands_only or key in want) else 0)
        except Exception:
            pass
    try:
        if hands_only:
            vk.vp.renderer.SetBackground(0.0, 0.0, 0.0)
        else:
            from pianoplayer.vkeyboard import BG_COLOR

            vk.vp.renderer.SetBackground(*BG_COLOR)
    except Exception:
        pass
    vk.vp.render()


def _shot(vk):
    img = vk.vp.screenshot(asarray=True)
    return np.asarray(img)


def _label(mask):
    """4-connected components via BFS on the (downsampled) mask.

    Returns (label_image, sizes). Assumes no scipy (may be missing)."""
    small = mask[::2, ::2]
    h, w = small.shape
    lab = np.zeros((h, w), np.int32)
    sizes = []
    cur = 0
    for r in range(h):
        row_m = small[r]
        row_l = lab[r]
        for c in range(w):
            if row_m[c] and not row_l[c]:
                cur += 1
                n = 0
                stack = [(r, c)]
                row_l[c] = cur
                while stack:
                    y, x = stack.pop()
                    n += 1
                    if y > 0 and small[y - 1, x] and not lab[y - 1, x]:
                        lab[y - 1, x] = cur
                        stack.append((y - 1, x))
                    if y + 1 < h and small[y + 1, x] and not lab[y + 1, x]:
                        lab[y + 1, x] = cur
                        stack.append((y + 1, x))
                    if x > 0 and small[y, x - 1] and not lab[y, x - 1]:
                        lab[y, x - 1] = cur
                        stack.append((y, x - 1))
                    if x + 1 < w and small[y, x + 1] and not lab[y, x + 1]:
                        lab[y, x + 1] = cur
                        stack.append((y, x + 1))
                sizes.append(n)
    return lab, sizes


def _pca_tilt(ys, xs):
    """Angle [deg] of the principal axis vs vertical (image rows)."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    x -= x.mean()
    y -= y.mean()
    cxx = float((x * x).mean())
    cyy = float((y * y).mean())
    cxy = float((x * y).mean())
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    return float(np.degrees(np.arctan2(abs(vx), abs(vy))))


def _runs(row):
    """Number of foreground runs in a 1D bool row, merging tiny gaps."""
    idx = np.flatnonzero(row)
    if len(idx) == 0:
        return 0
    gaps = np.diff(idx)
    return int(1 + np.count_nonzero(gaps > GAP_MERGE_PX))


def _hand_metrics(mask, label_id, lab_full_res_coords, px_per_cm, bone_cm,
                  thumb_side):
    """tilt + gap for one hand given its component id.

    lab_full_res_coords: (ys, xs) pixel coords of this hand (full res).
    thumb_side: 'left' (RH: drop x < 30th pct) or 'right' (LH: mirrored)."""
    ys, xs = lab_full_res_coords
    tilt = _pca_tilt(ys, xs)
    rmin, rmax = int(ys.min()), int(ys.max())
    off = int(round(0.5 * bone_cm * px_per_cm))
    rslice = min(max(int(rmin + off), rmin), rmax)
    row = mask[rslice]
    cols = np.flatnonzero(np.any(mask[rmin:rmax + 1], axis=0))
    if len(cols) == 0:
        return tilt, 0, rslice
    if thumb_side == "left":
        cut = np.percentile(xs, 30.0)
        keep = np.flatnonzero(row & (np.arange(mask.shape[1]) >= cut))
    else:
        cut = np.percentile(xs, 70.0)
        keep = np.flatnonzero(row & (np.arange(mask.shape[1]) <= cut))
    sel = np.zeros_like(row, bool)
    sel[keep] = True
    # restrict to this hand's column span so the other hand never leaks in
    span = np.zeros_like(row, bool)
    span[cols.min():cols.max() + 1] = True
    gap = _runs(sel & span)
    return tilt, gap, rslice


def _rig_metrics(rigs):
    """min proj ratio / max bend over idle fingers 2..5 of both hands."""
    from pianoplayer.hand3d import _SPECS

    ratios = []
    bends = []
    for rig in rigs:
        try:
            ky = float(np.mean([rig._geom[f]["pts"][-2][1]
                                for f in (2, 3, 4, 5)]))
        except Exception:
            continue
        for f in (2, 3, 4, 5):
            try:
                tip = rig._geom[f]["pts"][-1]
                bone = float(sum(_SPECS[f]["lens"]) * rig.hf)
                ratios.append(abs(float(tip[1]) - ky) / bone)
                bends.append(float(rig.max_bend[f]))
            except Exception:
                pass
    return min(ratios) if ratios else float("nan"), \
        max(bends) if bends else float("nan"), \
        float(np.mean(ratios)) if ratios else float("nan")


def _measure_pose(vk, rh, lh):
    from pianoplayer.hand3d import _SPECS

    _set_scene(vk, (rh, lh), hands_only=True)
    img = _shot(vk)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    mask = img.max(axis=2) > MASK_THRESH
    lab, sizes = _label(mask)
    if len(sizes) < 2:
        raise RuntimeError("expected 2 hand components, got %d" % len(sizes))
    order = np.argsort(sizes)[::-1][:2]
    # back to full resolution: label id -> pixel coords
    hands = []
    for lid in order:
        rr, cc = np.where(lab == int(lid) + 1)
        hands.append((np.asarray(rr * 2, int), np.asarray(cc * 2, int)))
    # larger centroid x = RH (keyboard +x = screen right)
    cx = [float(h[1].mean()) for h in hands]
    rh_h, lh_h = (hands[0], hands[1]) if cx[0] > cx[1] else (hands[1], hands[0])
    try:
        pscale = float(vk.vp.camera.GetParallelScale())
        px_per_cm = img.shape[0] / (2.0 * pscale)
    except Exception:
        px_per_cm = 10.0
    bone_cm = float(np.mean([sum(_SPECS[f]["lens"]) * rh.hf
                             for f in (2, 3, 4, 5)]))
    tilt_rh, gap_rh, _ = _hand_metrics(mask, 0, rh_h, px_per_cm, bone_cm,
                                       "left")
    tilt_lh, gap_lh, _ = _hand_metrics(mask, 1, lh_h, px_per_cm, bone_cm,
                                       "right")
    ratio_min, bend_max, ratio_mean = _rig_metrics((rh, lh))
    return {
        "tilt_rh": tilt_rh, "tilt_lh": tilt_lh,
        "tilt_max": max(tilt_rh, tilt_lh),
        "gap_rh": gap_rh, "gap_lh": gap_lh, "gap_max": max(gap_rh, gap_lh),
        "ratio_min": ratio_min, "ratio_mean": ratio_mean, "bend_max": bend_max,
        "mask": mask,
    }


def measure_all(write_files=True, expected_path=None):
    """Pose both hands (mid + ends), screenshot, measure. Returns metrics."""
    os.makedirs(OUT, exist_ok=True)
    (rh_mid, lh_mid), (rh_end, lh_end) = _fans()
    vk, rh, lh = _build()

    # ---- mid pose ----
    _pose(rh, rh_mid, {1: 1.0})
    _pose(lh, lh_mid, {1: 1.0})
    _set_scene(vk, (rh, lh), hands_only=False)
    beauty_mid = _shot(vk)
    m_mid = _measure_pose(vk, rh, lh)

    # ---- ends pose ----
    _pose(rh, rh_end, {1: 1.0})
    _pose(lh, lh_end, {1: 1.0})
    _set_scene(vk, (rh, lh), hands_only=False)
    beauty_ends = _shot(vk)
    m_ends = _measure_pose(vk, rh, lh)
    _set_scene(vk, (rh, lh), hands_only=False)

    metrics = {
        "tilt_rh_mid": m_mid["tilt_rh"], "tilt_lh_mid": m_mid["tilt_lh"],
        "tilt_mid_max": m_mid["tilt_max"],
        "tilt_rh_ends": m_ends["tilt_rh"], "tilt_lh_ends": m_ends["tilt_lh"],
        "tilt_ends_max": m_ends["tilt_max"],
        "gap_rh": m_mid["gap_rh"], "gap_lh": m_mid["gap_lh"],
        "gap_max": m_mid["gap_max"],
        "gap_rh_ends": m_ends["gap_rh"], "gap_lh_ends": m_ends["gap_lh"],
        "ratio_min": m_mid["ratio_min"], "ratio_mean": m_mid["ratio_mean"],
        "bend_max": m_mid["bend_max"],
        "ratio_min_ends": m_ends["ratio_min"], "bend_max_ends": m_ends["bend_max"],
    }

    if write_files:
        from PIL import Image

        def crop_of(beauty, mask):
            ys, xs = np.where(mask)
            r0, r1 = max(int(ys.min()) - 20, 0), min(int(ys.max()) + 20,
                                                    beauty.shape[0])
            c0, c1 = max(int(xs.min()) - 20, 0), min(int(xs.max()) + 20,
                                                    beauty.shape[1])
            return beauty[r0:r1, c0:c1], (r0, r1, c0, c1)

        Image.fromarray(beauty_mid).save(os.path.join(OUT, "actual.png"))
        mid_crop, _ = crop_of(beauty_mid, m_mid["mask"])
        Image.fromarray(mid_crop).save(os.path.join(OUT, "mid_crop.png"))
        Image.fromarray(beauty_ends).save(os.path.join(OUT, "ends.png"))
        ends_crop, _ = crop_of(beauty_ends, m_ends["mask"])
        Image.fromarray(ends_crop).save(os.path.join(OUT, "ends_crop.png"))

        # Default search paths for expected image
        search_paths = []
        if expected_path:
            search_paths.append(expected_path)
        search_paths.extend([
            os.path.join(HERE, "docs", "expected1.png"),
            r"C:\Users\Admin\Pictures\Screenshots\expected1.png",
        ])
        exp = None
        for p in search_paths:
            if os.path.isfile(p):
                exp = p
                break
        if exp:
            left = np.asarray(Image.open(exp).convert("RGB"))
            note = f"expected: {exp}"
            right = mid_crop
        else:
            left = mid_crop
            right = ends_crop
            note = "no expected image: mid (left) vs ends (right)"
        h = max(left.shape[0], right.shape[0])

        def pad(a):
            w = a.shape[1] * h // max(a.shape[0], 1)
            return np.asarray(Image.fromarray(a).resize((w, h)))

        compare = np.concatenate([pad(left), pad(right)], axis=1)
        Image.fromarray(compare).save(os.path.join(OUT, "compare.png"))
        with open(os.path.join(OUT, "metrics.txt"), "w") as fh:
            fh.write(table_text(metrics) + "\n# %s\n" % note)
        print("# %s" % note)

    try:
        vk.vp.close()
    except Exception:
        pass
    return metrics


def table_text(m):
    lines = [
        "hand-look metrics (rest = idle fingers 2..5, both hands):",
        "  tilt_RH_mid    = %6.2f deg   (target <= %.1f)"
        % (m["tilt_rh_mid"], TARGETS["tilt_mid_max"]),
        "  tilt_LH_mid    = %6.2f deg   (target <= %.1f)"
        % (m["tilt_lh_mid"], TARGETS["tilt_mid_max"]),
        "  tilt_mid_max   = %6.2f deg" % m["tilt_mid_max"],
        "  tilt_RH_ends   = %6.2f deg   (target <= %.1f)"
        % (m["tilt_rh_ends"], TARGETS["tilt_ends_max"]),
        "  tilt_LH_ends   = %6.2f deg   (target <= %.1f)"
        % (m["tilt_lh_ends"], TARGETS["tilt_ends_max"]),
        "  tilt_ends_max  = %6.2f deg" % m["tilt_ends_max"],
        "  gap_RH / gap_LH = %d / %d   (target == %d)"
        % (m["gap_rh"], m["gap_lh"], TARGETS["gap_components"]),
        "  gap_max        = %d" % m["gap_max"],
        "  proj_ratio_min = %.3f (mean %.3f)  (target >= %.2f)"
        % (m["ratio_min"], m["ratio_mean"],
           TARGETS["finger_proj_ratio_min"]),
        "  idle_bend_max  = %.3f rad    (target <= %.2f)"
        % (m["bend_max"], TARGETS["idle_max_bend_max"]),
    ]
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", default=None,
                        help="Path to expected image for side-by-side comparison")
    args = parser.parse_args()
    m = measure_all(write_files=True, expected_path=args.expected)
    print(table_text(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
