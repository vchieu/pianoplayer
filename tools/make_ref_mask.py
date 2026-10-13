#!/usr/bin/env python3
"""Make docs/ref_mask.png from docs/ref_top.jpg (R0 IoU gate).

Usage (run from repo root):
    python tools/make_ref_mask.py docs/ref_top.jpg [docs/ref_mask.png]

How it works: YCbCr skin-color threshold (Cr 133-173, Cb 77-127) +
cleanup (Median/Min/MaxFilter) + keep largest component + fill holes.
Hand = white (255), background = black (0).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter


def skin_mask(img):
    small = img.copy()
    small.thumbnail((640, 640))
    a = np.asarray(small.convert("YCbCr")).astype(np.int32)
    Y, Cb, Cr = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    m = (Y > 40) & (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
    pim = Image.fromarray((m * 255).astype(np.uint8))
    pim = pim.filter(ImageFilter.MedianFilter(5))
    pim = pim.filter(ImageFilter.MinFilter(5))
    pim = pim.filter(ImageFilter.MaxFilter(7))
    return pim.resize(img.size, Image.NEAREST)


def largest_component(mask):
    a = np.asarray(mask) > 128
    H, W = a.shape
    seen = np.zeros_like(a, dtype=bool)
    best = []
    for r in range(H):
        for c in range(W):
            if not a[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < H and 0 <= nx < W and a[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) > len(best):
                best = cells
    out = np.zeros_like(a, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def fill_holes(a):
    H, W = a.shape
    bg = np.zeros_like(a, dtype=bool)
    stack = [(y, x) for x in range(W) for y in (0, H - 1) if not a[y, x]]
    stack += [(y, x) for y in range(H) for x in (0, W - 1) if not a[y, x]]
    while stack:
        y, x = stack.pop()
        if not (0 <= y < H and 0 <= x < W) or a[y, x] or bg[y, x]:
            continue
        bg[y, x] = True
        stack += [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
    return a | ~bg


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("docs", "ref_top.jpg")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("docs", "ref_mask.png")
    img = Image.open(src).convert("RGB")
    comp = largest_component(skin_mask(img))
    if comp.sum() < 100:
        print("WARN: skin region tiny -- paint mask by hand instead")
        return 1
    Image.fromarray(fill_holes(comp).astype(np.uint8) * 255).save(out)
    print("wrote %s (hand=white). verify visually, then: python diag_ref_iou.py" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
