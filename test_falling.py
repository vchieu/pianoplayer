#!/usr/bin/env python3
"""Geometry of the falling-note layer (numpy only, no window)."""
import sys

import numpy as np

from pianoplayer import falling as F

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    top = 54.4
    sp = F.fall_speed(top)
    check("speed: note enters at the top edge FALL_SECONDS early",
          abs(F.HIT_Y + F.FALL_SECONDS * sp - top) < 1e-9)

    # bar lower edge touches the line exactly at t_on
    e = F.bar_extent(2.0, 3.0, 2.0, sp, top_y=top)
    check("bar bottom on the hit line at t_on", e is not None and abs(e[0] - F.HIT_Y) < 1e-9)
    e = F.bar_extent(2.0, 3.0, 1.0, sp, top_y=top)
    check("bar 1 s early sits sp cm above the line", abs(e[0] - (F.HIT_Y + sp)) < 1e-9)

    # clipped once it is being held; gone once released
    e = F.bar_extent(2.0, 3.0, 2.5, sp, top_y=top)
    check("held bar is clipped at the line", e is not None and e[0] == F.HIT_Y)
    check("released bar disappears", F.bar_extent(2.0, 3.0, 3.01, sp, top_y=top) is None)
    check("bar past the top edge is not drawn",
          F.bar_extent(2.3, 2.4, 0.0, sp, top_y=top) is None)

    # abutting notes of one key leave a visible gap
    a = F.bar_extent(1.0, 1.5, 0.0, sp, top_y=1e9)
    b = F.bar_extent(1.5, 2.0, 0.0, sp, top_y=1e9)
    check("gap between abutting bars >= BAR_GAP", b[0] - a[1] >= F.BAR_GAP - 1e-9,
          "gap=%.2f" % (b[0] - a[1]))

    # very short note still gets a minimum length
    e = F.bar_extent(1.0, 1.01, 0.0, sp, top_y=1e9)
    check("MIN_BAR", abs((e[1] - e[0]) - F.MIN_BAR) < 1e-9)

    # label
    check("label hidden on a tiny bar", F.label_extent(10.0, 10.0 + F.MIN_BAR) is None)
    lab = F.label_extent(10.0, 14.0)
    check("label inside a long bar", lab is not None and lab[1] == F.LABEL_H
          and lab[0] + lab[1] <= 14.0)
    lab = F.label_extent(F.HIT_Y, F.HIT_Y + 4.0)
    check("held label rests on the line", abs(lab[0] - (F.HIT_Y + F.LABEL_PAD)) < 1e-9)

    # digits
    for d in range(1, 6):
        t = F.digit_template(d)
        check("digit %d template shape" % d, t.shape == (F.MAX_QUADS * 4, 2))
        check("digit %d inside the unit box" % d,
              t.min() > -0.2 and t.max() < 1.2, "min=%.2f max=%.2f" % (t.min(), t.max()))
        # at least some area: bounding box of the real quads is large
        check("digit %d is not empty" % d, np.ptp(t[:, 1]) > 0.7 and np.ptp(t[:, 0]) > 0.3)
    v = F.place_digit(F.digit_template(3), 10.0, 6.0, 1.7, 0.7)
    check("digit centred on the bar", abs((v[:, 0].min() + v[:, 0].max()) / 2 - 10.0) < 0.3)
    check("digit z", np.all(v[:, 2] == 0.7))
    check("digit height", abs((v[:, 1].max() - v[:, 1].min()) - 1.7) < 0.5)

    # F1: stroke evenness -- a horizontal bar must be as thick as a vertical
    # stem (digit 4's bar vs digit 1's stem), within 5%
    def quad_spans(v):
        hs, vs = [], []
        for k in range(F.MAX_QUADS):
            q = v[4 * k:4 * k + 4]
            w = q[:, 0].max() - q[:, 0].min()
            h = q[:, 1].max() - q[:, 1].min()
            if w < 1e-9 and h < 1e-9:
                continue                    # unused quad, collapsed to a point
            if h > 2 * w:
                vs.append(w)
            elif w > 2 * h:
                hs.append(h)
        return hs, vs
    v1 = F.place_digit(F.digit_template(1), 10.0, 6.0, 1.7, 0.7)
    v4 = F.place_digit(F.digit_template(4), 10.0, 6.0, 1.7, 0.7)
    _, stem_w = quad_spans(v1)
    bar_h, _ = quad_spans(v4)
    ok = bool(stem_w) and bool(bar_h)
    rel = (abs(sum(stem_w) / len(stem_w) - sum(bar_h) / len(bar_h))
           / ((sum(stem_w) / len(stem_w) + sum(bar_h) / len(bar_h)) / 2)) if ok else 1.0
    check("stroke evenness H vs V <5%",
          ok and rel < 0.05, "stem=%.3f bar=%.3f rel=%.3f"
          % (sum(stem_w) / len(stem_w) if stem_w else -1,
             sum(bar_h) / len(bar_h) if bar_h else -1, rel))

    print("\n%d failure(s)" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
