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
        vk.vp.render()
        # Pose mid-keyboard with fingers CLOSE (chromatic cluster C4..E4):
        # the ref photo shows adjacent fingers touching; a C4..G4 white-key
        # fan spreads them (~2.4 cm apart) and measures pose, not shape.
        # (Neutral hover sits at x~=0, outside the 88-key camera frame.)
        from music21 import note as _note
        from pianoplayer.utils import keypos as _keypos
        fan = dict(zip((1, 2, 3, 4, 5),
                       [float(_keypos(_note.Note(n)))
                        for n in ("C4", "C#4", "D4", "D#4", "E4")]))
        rig.settle(dict(fan), {})
        for _ in range(60):
            rig.update(1.0 / 60.0, dict(fan), {})
        # hands-only + black background (actor-keying done with robust hex
        # parse: GetAddressAsString may return 'Addr=<hex>' or '<hex>'; a
        # naive int() succeeds only for all-digit addresses, making matches
        # nondeterministic. Normalize both sides to int(hex)).
        def _akey(o):
            try:
                s = o.GetAddressAsString("").replace("Addr=", "").replace(
                    "0x", "").strip()
                return int(s, 16)
            except Exception:
                return id(o)

        want = set()
        for m in rig.actors():
            want.add(_akey(m.actor))
        for a in vk.vp.get_actors():
            try:
                a.SetVisibility(1 if _akey(a) in want else 0)
            except Exception:
                pass
        # blob shadow belongs to the hand but would pollute the mask
        for b in getattr(rig, "_blob_actors", []):
            try:
                b.SetVisibility(0)
            except Exception:
                pass
        try:
            vk.vp.renderer.SetBackground(0.0, 0.0, 0.0)
        except Exception:
            pass
        vk.vp.render()
        img = np.asarray(vk.vp.screenshot(asarray=True))
        try:
            from PIL import Image as _PIL
            _PIL.fromarray(
                img if img.dtype == np.uint8
                else (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            ).save(os.path.join(HERE, "docs", "ref_render.png"))
        except Exception:
            pass
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
        # centered placement (translation part of similarity alignment):
        # top-left bias would swing the mass to the opposite corner on
        # rotation, making the rot search meaningless.
        canvas = np.zeros((side, side), np.uint8)
        ro = (side - crop.shape[0]) // 2
        co = (side - crop.shape[1]) // 2
        canvas[ro:ro + crop.shape[0], co:co + crop.shape[1]] = \
            crop.astype(np.uint8) * 255
        return np.asarray(
            Image.fromarray(canvas).resize((128, 128))) > 128

    a0, b = norm(mask), norm(ref)
    # Similarity alignment (plan R0): rotation via best-of-4 search --
    # phone photo vs render camera may differ by 90/180 deg.
    best, best_k = 0.0, 0
    for k in range(4):
        a = np.rot90(a0, k)
        inter = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)
        iou = float(inter) / max(union, 1)
        if iou > best:
            best, best_k = iou, k
    print("IoU vs %s = %.3f (rot=%d, target >= 0.85 post-R2)"
          % (REF, best, best_k * 90))
    return best


def main():
    iou_against_ref()
    return 0


if __name__ == "__main__":
    sys.exit(main())
