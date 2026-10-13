"""
Offscreen preview of the procedural hand.

    python test_hand_render.py

Writes PNGs in ./test_hand_render/ : for every pose three views
(perspective, profile, top-down).
"""
import os
import sys

from vedo import Box, Plotter

from pianoplayer.hand3d import HandRig
from pianoplayer.utils import KEYBSIZE, KEY_WB, handSizeFactor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_hand_render")
FREST = [None, -7.0, -2.8, 0.0, 2.8, 5.6]

VIEWS = {
    # name : (camera pos, focal point, view up)
    "persp": ((74, -44, 30), (52, -6, 3), (0, 0, 1)),
    "side":  ((80, -4, 13), (52, -5, 3), (0, 0, 1)),
    "top":   ((52, -6, 58), (52, -6, 0), (0, 1, 0)),
    "wide":  ((110, -66, 78), (75, 0, 2), (-0.16, 0.8, 0.55)),
}


def fan(side, size, f, x_note):
    """x of every fingertip when finger `f` plays the key at `x_note`."""
    hf = handSizeFactor(size)
    frest = [v * hf if v is not None else None for v in FREST]
    out = {}
    for j in (1, 2, 3, 4, 5):
        d = frest[j] - frest[f]
        out[j] = x_note + (-d if side == "left" else d)
    return out


def build_keys(vp, nr_octaves=7):
    nts = ("C", "D", "E", "F", "G", "A", "B")
    tol = 0.12
    span = nr_octaves * KEY_WB * 7
    vp += Box(pos=(span / 2 + KEYBSIZE, 6, 1),
              length=span + 1, height=3, width=5, c=(0.45, 0.28, 0.16))
    vp += Box(pos=(span / 2 + KEYBSIZE, 0, -1),
              length=span + 1, height=1, width=17, c=(0.35, 0.22, 0.13))
    for ioct in range(nr_octaves):
        for ik in range(7):
            x = ik * KEY_WB + (ioct + 1) * KEYBSIZE + KEY_WB / 2
            vp += Box(pos=(x, -2, 0), length=KEY_WB - tol, height=1,
                      width=12, c="white")
            if nts[ik] not in ("E", "B"):
                vp += Box(pos=(x + KEY_WB / 2, 0, 1), length=KEY_WB * 0.6,
                          height=1, width=8, c=(0.05, 0.05, 0.055))


def pose(side, size, f, x_note, press):
    """Settle a rig in the pose where finger `f` plays `x_note`."""
    rig = HandRig(side=side, size=size)
    rig.settle(fan(side, size, f, x_note), press)
    return rig


def shot(name, views, rigs, focus=(52, -6, 3)):
    for view in views:
        pos, focal, up = VIEWS[view]
        if view not in ("wide", "top"):
            focal = (focus[0], focal[1], focal[2])
            pos = (focus[0] + (pos[0] - 52), pos[1], pos[2])
        plt = Plotter(axes=0, size=(1400, 760), bg="cornsilk", bg2="lb",
                      offscreen=True)
        build_keys(plt)
        for r in rigs:
            r.add_to(plt)
            # Blob shadows are a top-down projection trick: from the side
            # they float as ghosts/veils (one actor even lands in the sky).
            # Mirror vkeyboard.toggle_view: show only for top-down.
            if view in ("side", "persp"):
                try:
                    r.set_shadows_visible(False)
                except Exception:
                    pass
        plt.camera = dict(pos=pos, focalPoint=focal, viewup=up)
        plt.render()
        path = os.path.join(OUT, "%s_%s.png" % (name, view))
        plt.screenshot(path)
        plt.close()
        print("wrote", path)


def main():
    os.makedirs(OUT, exist_ok=True)

    # middle finger pressing C3 (white key)
    rig = pose("right", "M", 3, 50.68, {3: 1.0})
    shot("rh_press", ("persp", "side", "top"), [rig])

    # thumb on a black key C#3
    rig = pose("right", "M", 1, KEYBSIZE * 3 + KEY_WB, {1: 1.0})
    shot("rh_black", ("persp", "side", "top"), [rig])

    # spread chord: thumb C3, middle E3, little G3
    hf = handSizeFactor("M")
    t = {1: 50.68, 2: 50.68 + 2.8 * hf, 3: 50.68 + 5.6 * hf,
         4: 50.68 + 8.4 * hf, 5: 50.68 + 11.2 * hf}
    rig = HandRig(side="right", size="M")
    rig.settle(t, {1: 1.0, 3: 1.0, 5: 1.0})
    shot("rh_chord", ("persp", "side", "top"), [rig], focus=(56, -6, 3))

    # left hand on a white key
    rig = pose("left", "M", 3, 50.68, {3: 1.0})
    shot("lh_press", ("persp", "side", "top"), [rig])

    # both hands, wide view
    r1 = pose("right", "M", 3, 66.0, {3: 1.0})
    r2 = pose("left", "M", 2, 40.0, {3: 1.0})
    shot("wide", ("wide",), [r1, r2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
