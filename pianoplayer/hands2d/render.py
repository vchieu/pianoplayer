#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Name:         Hands2D
# Purpose:      The 2D hand model (pygame-piano-animator) drawn offscreen and
#               composited over the 3D keys in the same vedo renderer
# -----------------------------------------------------------------------------
"""
`Hands2D` is the 2D hand render backend.

The hands are the *verbatim* model of
`D:\\Project\\pygame-piano-animator` (vendored next to this file: `hands.py`
poses and solves the hands, `skins.py` draws them): every frame it asks the
two `HandAnimator`s for a pose at score time `now`, draws them with
`hands.draw_hands` on an offscreen pygame surface whose pixels are laid out
on the 3D keyboard's own pixel grid, and pushes the RGBA result into a
`vtkImageData` shown by a `vtkImageActor` in the same renderer, above the 3D
keys.

The pixel mapping is the only thing pianoplayer adds:

    ppc  = window_width_px / (kb_w + 2 * KB_MARGIN_X)      (pixels per cm)
    x0   = cx - (kb_w/2 + KB_MARGIN_X)      left world x on screen
    x1   = cx + (kb_w/2 + KB_MARGIN_X)      right
    y0   = KB_BOTTOM_Y                       bottom world y
    y1   = cy + pscale                       top    (cy + pscale is the window
                                                 centre: the keys sit at the
                                                 bottom edge)
    px(x) = (x - x0) * ppc                   pygame column grows *right*,
    py(y) = (band_top - y) * ppc              pygame row grows *down*, world y
                                                 grows *up*

The 2D model draws in that surface pixel space and only ever reaches from
`front_y` (the white keys' front edge) downwards, so the surface is
**cropped to the hand band** `[y0, band_top]` with `band_top = BAND_TOP_Y`:
the hands' own geometry is in inches, so their reach in world centimetres is
the same at every window size and everything above the band is provably
transparent. The crop costs nothing visually and is what makes the backend
playable (draw + upload + the VTK texture are ~2.3x cheaper).
`py_of_world_y` / `world_y_of_row` map world y to/from the band's rows.

No scaling happens in VTK (`img.SetSpacing(1/ppc, 1/ppc, 1)`), and every
fingertip the 2D solver places on `key_rects[pitch].centerx` is the pixel of
the 3D key's own x - `key_rects` is rebuilt from `_key_x` after layout,
because the shim keyboard's black-key offsets are the 2D app's, not
pianoplayer's.

pygame is imported lazily and only ever as a drawing surface: no display, no
`pygame.init()`, no mixer, no files. If it (or the VTK image classes) is
missing, `VirtualKeyboard` falls back to the 3D hands with a warning.

Typical use::

    h2 = Hands2D(vk.vp, geom)      # geom from VirtualKeyboard._ensure_hands2d
    h2.build({"R": rh_notes, "L": lh_notes})
    h2.update(now, dt)              # every frame
    h2.close()
"""
from __future__ import annotations

__all__ = ["Hands2D", "Z_OVERLAY", "BAND_TOP_Y"]

Z_OVERLAY = 2.2           # world z of the overlay plane: above the keys (z<=1.5)
                          # and the falling bars (z<=0.70), below the camera
BLACK_FRONT_CM = 4.0      # world y of the black keys' front edge (-4.0): the
                          # black-key front in the surface sits 4 cm above the
                          # white keys' front edge (-8.0), like in 3D
WHITE_FRONT_Y = -8.0      # world y of the white keys' front edge
BLACK_FRONT_SHARE = 0.37  # the shim keyboard leaves 0.37 of its rect below the
                          # black keys (black_h = 0.63 * rect.h): the depth of
                          # the visible playing area
PREPARE_BUDGET = 0.005    # s of a frame's spare time given to hands.prepare
# Top of the overlay band, in world y.  The 2D hands are drawn from the white
# keys' front edge *downwards*; measured over nine scores (12 s of playback
# each, both hands, all four skins) the topmost painted pixel never rose
# above +9.8 cm -- the drawn outline/shadow reaches ~8 cm above the topmost
# joint, so it is not the fingers alone.  14.0 keeps ~4 cm (60 px at
# 1800x1000) of provably transparent margin on top while covering only 44% of
# the window height.  Raise it if a future skin paints further up;
# test_hands2d.py asserts the top rows of the band stay empty.
BAND_TOP_Y = 14.0
WARM_BUDGET = 0.25        # s for the one-off cache warm-up before the clock
FALLBACK_FINGER = 3       # a note with no finger still gets a hand on it


class Hands2D:
    """Offscreen 2D hands composited into a vedo/VTK scene.

    `geom` is the pixel mapping described in the module docstring; it carries
    the window size, the camera (pscale, cy) and the keyboard (cx, kb_w), and
    `key_x(midi)` for the 3D keyboard's own key positions.
    """

    def __init__(self, vp, geom, style=None):
        self.geom = dict(geom or {})
        self.style = style or "cartoon"
        self.vp = None
        self.kb = None                # the shim keyboard the 2D hands measure with
        self.band_top_y = 0.0         # world y of the band's top row
        self.anims = {}               # "R" / "L" -> HandAnimator
        self.song = None
        self.pianist = None
        self.actor = None             # vtkImageActor
        self.img = None               # vtkImageData
        self.surf = None              # pygame.Surface (SRCALPHA)
        self.size = (0, 0)
        self._rgba = None             # (H, W, 4) uint8, reused every frame
        self._np = None               # numpy, once the surface exists
        self._vtk_to_numpy = None
        self._failed = None           # first failure message (one warning only)
        self._add_to(vp)

    # ---------------------------------------------------------------- mapping
    def px_of_world_x(self, x):
        """Pygame/VTK column of world x [cm]."""
        return (float(x) - self.geom["x0"]) * self.geom["ppc"]

    def py_of_world_y(self, y):
        """Pygame row of world y [cm] inside the band (row 0 = band top)."""
        return (self.band_top_y - float(y)) * self.geom["ppc"]

    def world_y_of_row(self, row):
        """World y [cm] of a band row (inverse of `py_of_world_y`)."""
        return self.band_top_y - float(row) / self.geom["ppc"]

    def world_x_of_col(self, col):
        """World x [cm] of a surface column (inverse of `px_of_world_x`)."""
        return self.geom["x0"] + float(col) / self.geom["ppc"]

    # ------------------------------------------------------------------ setup
    def _add_to(self, vp):
        """Create the surface, the VTK image and (once) add the actor to `vp`."""
        if self.actor is not None:
            if vp is not None and self.vp is not vp:
                self.add_to(vp)
            return True
        try:
            import numpy as np
            import pygame
            from vtkmodules.util.numpy_support import vtk_to_numpy
            from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
            from vtkmodules.vtkCommonDataModel import vtkImageData
            from vtkmodules.vtkRenderingCore import vtkImageActor

            w = int(self.geom["w"])
            ppc = float(self.geom["ppc"])
            # the band: the window's lower part, from the bottom edge up to
            # BAND_TOP_Y.  The 2D hands are drawn from the white keys' front
            # edge downwards, so nothing above the band is ever painted.
            self.band_top_y = min(BAND_TOP_Y, float(self.geom["y1"]))
            h = int(round((self.band_top_y - self.geom["y0"]) * ppc))
            if w < 2 or h < 2:
                raise ValueError("bad overlay size %rx%r" % (w, h))

            # The shim keyboard: the realistic key layout, then every key's
            # centre forced onto pianoplayer's own key x (no _BLACK_OFFSETS).
            # Its rect top may sit above the band (the band starts mid-keyboard
            # area): the model only uses rect.h, rect.bottom and rect.centerx.
            rect_h = (BLACK_FRONT_CM * ppc) / BLACK_FRONT_SHARE
            kb = self._make_keyboard(pygame)
            kb.rect = pygame.Rect(0, int(round(self.py_of_world_y(WHITE_FRONT_Y))) - rect_h,
                                  w, rect_h)
            key_x = self.geom["key_x"]
            for midi in range(kb.low, kb.high + 1):
                kb.key_rects[midi].centerx = int(round(self.px_of_world_x(key_x(midi))))
            kb.white_w = float(self.geom["key_w"]) * ppc       # one white key, in px
            kb.black_w = kb.white_w * 0.6
            kb.black_h = int(rect_h * 0.63)
            kb.tails = {}
            kb.style = "realistic"
            self.kb = kb

            self.surf = pygame.Surface((w, h), pygame.SRCALPHA, 32)
            self.size = (w, h)
            self._rgba = np.zeros((h, w, 4), dtype=np.uint8)

            img = vtkImageData()
            img.SetDimensions(w, h, 1)
            img.SetSpacing(1.0 / ppc, 1.0 / ppc, 1.0)
            img.SetOrigin(float(self.geom["x0"]), float(self.geom["y0"]), Z_OVERLAY)
            img.AllocateScalars(VTK_UNSIGNED_CHAR, 4)
            img.Modified()
            actor = vtkImageActor()
            actor.SetInputData(img)
            self.img = img
            self.actor = actor
            self._np = np
            self._vtk_to_numpy = vtk_to_numpy
            self.vp = vp
            if vp is not None:
                self.add_to(vp)
            return True
        except Exception as exc:              # no pygame / no vtk image classes
            self._failed = str(exc)
            return False

    def _make_keyboard(self, pygame):
        """The shim keyboard; every position that matters is overridden after
        `layout` (the key centres, the key widths, the rect)."""
        from .common import Keyboard
        return Keyboard(pygame.Rect(0, 0, 52, 4))

    def add_to(self, vp):
        """Add the overlay actor to `vp` (idempotent)."""
        if self.actor is None or vp is None:
            return False
        self.vp = vp
        try:
            vp.renderer.AddActor(self.actor)
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------ build
    def build(self, notes_by_hand, style=None):
        """Make the 2D animators for `notes_by_hand` = {"R": [_Ev...], "L": [...]}.

        `_Ev` is pianoplayer's note event (t_on, t_off, midi, vel, finger);
        the finger it carries is the one the 2D hands animate (a note with no
        finger still gets one, so every pressed key has a hand over it).
        """
        from . import hands, midi_loader, pianist as pianists, skins

        style = style or self.style or "cartoon"
        self.style = style
        notes = []
        for side in (midi_loader.RIGHT, midi_loader.LEFT):
            for e in notes_by_hand.get(side, None) or ():
                if getattr(e, "midi", None) is None:
                    continue
                try:
                    fing = int(e.finger)
                except Exception:
                    fing = 0
                notes.append(midi_loader.Note(
                    pitch=int(e.midi), start=float(e.t_on), end=float(e.t_off),
                    velocity=max(1, min(127, int(round(float(e.vel) * 127)))),
                    track=0, hand=side,
                    finger=fing if 1 <= fing <= 5 else FALLBACK_FINGER,
                    gliss=False))
        self.song = midi_loader.MidiSong(notes)
        # an unknown --hand-style name falls back to the 2D default skin
        # instead of breaking the drawing (skins.normalize keeps its palette)
        self.pianist = pianists.Pianist("Default", pianists.DEFAULT_ID,
                                        skin=skins.normalize({"style": style, "outline": 0.0, "shadow": False}))
        self.anims = {}
        for side in (midi_loader.RIGHT, midi_loader.LEFT):
            mine = [n for n in self.song.notes if n.hand == side]
            if not mine:
                continue
            self.anims[side] = hands.HandAnimator(
                self.song, side,
                # a finger for every note skips the 2D fingering planner
                fingering={id(n): n.finger for n in mine},
                pianist=self.pianist)
        hands.pair_hands(self.anims.values())
        self._warm()
        return self

    def _warm(self):
        """Pay the cold-start cost (caches, first pose, first draw) up front."""
        if not self.anims or self.kb is None or self.surf is None:
            return
        from . import hands
        t0 = self.song.notes[0].start
        try:
            hands.prepare_hands(self.anims.values(), t0, self.kb, WARM_BUDGET)
            self._draw(t0)
        except Exception as exc:
            self._warn(exc)
            self.set_visible(False)

    def settle(self, *args, **kwargs):
        """No-op: the 2D model has no rest-pose call to prime (3D parity)."""

    # ----------------------------------------------------------------- update
    def update(self, now, dt=None, budget=PREPARE_BUDGET):
        """Pose + draw + upload at score time `now` (never raises)."""
        if not self.anims or self.kb is None or self.surf is None:
            return
        if self.actor is not None and not self.actor.GetVisibility():
            return                       # hidden (perspective view): skip the work
        try:
            from . import hands
            hands.prepare_hands(self.anims.values(), now, self.kb, budget)
            self._draw(now)
        except Exception as exc:
            self._warn(exc)
            self.set_visible(False)

    def _draw(self, now):
        """Draw every pose at `now` onto the surface and push it to VTK."""
        from . import hands
        poses = [a.pose(now, self.kb) for a in self.anims.values()]
        self.surf.fill((0, 0, 0, 0))
        hands.draw_hands(self.surf, poses)
        self._upload()
        return poses

    def _upload(self):
        """Copy the surface into the vtkImageData (buffers are reused).

        The pygame surface is top-down, a VTK image is bottom-up (row 0 = the
        lowest y), so the rows are reversed: `rgba[y]` is the pygame row
        H-1-y.

        The surface keeps pygame's default 32-bit masks: drawing into it is
        ~4x faster than into an RGBA-masked one (the skins blit a lot), and
        the BGRA->RGBA swap the upload then needs costs less than that saves.
        """
        import pygame
        h, w = self.size[1], self.size[0]
        buf = pygame.image.tobytes(self.surf, "RGBA")
        self._rgba[:, :, :] = self._np.frombuffer(buf, dtype=self._np.uint8
                                                  ).reshape(h, w, 4)[::-1]
        self._vtk_to_numpy(self.img.GetPointData().GetScalars())[:] = \
            self._rgba.reshape(-1, 4)
        self.img.Modified()

    def alpha_pixels(self):
        """The overlay's alpha channel in pygame orientation (row 0 = the top
        of the window), so `a[py_of_world_y(y), px_of_world_x(x)]` is the
        pixel drawn at world (x, y)."""
        return self._rgba[::-1, :, 3]

    # ----------------------------------------------------------------- misc
    def set_visible(self, visible):
        """Show/hide the overlay (the perspective view has no flat plane)."""
        if self.actor is None:
            return
        try:
            self.actor.SetVisibility(1 if visible else 0)
        except Exception:
            pass

    def in_renderer(self):
        """True when the actor is really in the scene's renderer (tests)."""
        if self.actor is None or self.vp is None:
            return False
        try:
            return bool(self.vp.renderer.HasViewProp(self.actor))
        except Exception:
            return False

    def close(self):
        """Drop the VTK actor and the surface."""
        try:
            if self.actor is not None and self.vp is not None:
                self.vp.renderer.RemoveActor(self.actor)
        except Exception:
            pass
        try:
            if self.surf is not None:
                self.surf.fill((0, 0, 0, 0))
        except Exception:
            pass
        self.actor = None
        self.img = None
        self.surf = None
        self._rgba = None
        self.anims = {}
        self.vp = None

    def _warn(self, exc):
        """Report the first failure only: the layer is then switched off."""
        if self._failed is None:
            self._failed = "%s: %s" % (type(exc).__name__, exc)
            print("2D hands disabled:", self._failed)