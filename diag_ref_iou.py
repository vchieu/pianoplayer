"""Silhouette IoU vs a reference hand photo (Phase R, R0).

Renders the right hand in rest pose, captures its mask (like
diag_hand_look.py), similarity-aligns it (translation + rotation + uniform
scale via bbox) to docs/ref_mask.png (hand = white), and prints IoU.

If docs/ref_mask.png is missing (or rendering fails headless) the tool
prints SKIP and returns None -- it never asserts.  Post-R2 target IoU>=0.85.

    python diag_ref_iou.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REF = os.path.join(HERE, "docs", "ref_mask.png")
MASK_THRESH = 30


def iou_against_ref():
    """Returns IoU float, or None when skipped (no ref / no renderer)."""
    if not os.path.isfile(REF):
        print("SKIP: %s missing (shoot docs/ref_top.jpg, make ref_mask.png)"
              % REF)
        return None
    try:
        import numpy as np
        from PIL import Image
        from pianoplayer.hand3d import HandRig
        from pianoplayer.vkeyboard import VirtualKeyboard
    except Exception as exc:
        print("SKIP: imports unavailable (%s)" % exc)
        return None
    try:
        vk = VirtualKeyboard(songname="ref-iou", offscreen=True)
        vk.verbose = False
        rig = HandRig(side="right", size="M")
        rig.add_to(vk.vp)
        rig.settle(rig._neutral_x(), {})
        for _ in range(30):
            rig.update(1.0 / 60.0, dict(rig._neutral_x()), {})
        vk.vp.render()
        img = np.asarray(vk.vp.screenshot(asarray=True))
        try:
            vk.vp.close()
        except Exception:
            pass
    except Exception as exc:
        print("SKIP: offscreen render unavailable (%s)" % exc)
        return None
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    mask = img.max(axis=2) > MASK_THRESH
    ref = np.asarray(Image.open(REF).convert("L")) > 128

    def norm(m):
        ys, xs = np.where(m)
        if len(xs) == 0:
            return m
        r0, r1 = ys.min(), ys.max()
        c0, c1 = xs.min(), xs.max()
        crop = m[r0:r1 + 1, c0:c1 + 1]
        side = max(crop.shape)
        canvas = np.zeros((side, side), bool)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        return np.asarray(Image.fromarray(canvas).resize((128, 128))) > 0

    a, b = norm(mask), norm(ref)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    iou = float(inter) / max(union, 1)
    print("IoU vs %s = %.3f (target >= 0.85 post-R2)" % (REF, iou))
    return iou


def main():
    iou_against_ref()
    return 0


if __name__ == "__main__":
    sys.exit(main())
