"""
common.py (shim) - just the key geometry the vendored hands.py measures with.

`KEY_LEN_WW` (the white key's length in white-key widths - the keys' true
proportions, used by hands._chord_fits) and a `Keyboard` that lays out the
realistic key shapes. The rest of the source module (window layout, key
drawing, falling notes, colours) is 2D-app UI pianoplayer does not use:
the real keyboard is drawn in 3D by vkeyboard.build_keyboard.

The `equal` key style is not ported - nothing in the vendored hands.py
selects it (kb.style is only read, never compared) and hands._chord_fits
relies on the realistic default.
"""
from __future__ import annotations

from .midi_loader import (HIGHEST_PIANO_KEY, LOWEST_PIANO_KEY, is_black_key)

KEY_LEN_WW = 5.6            # white key length, in white-key widths (the keys' true proportions)

# Small sideways nudges that make black keys sit like they do on a real piano.
_BLACK_OFFSETS = {1: -0.12, 3: 0.12, 6: -0.14, 8: 0.0, 10: 0.14}

# Key styles: "realistic" (a real piano's proportions) or "equal" (PASHKULI's
# equal lanes). Only "realistic" is implemented here.
KEY_STYLES = ("realistic",)
_key_style = "realistic"


def key_style():
    return _key_style


def set_key_style(style):
    global _key_style
    _key_style = style if style in KEY_STYLES else "realistic"


class Keyboard:
    """The realistic key layout: rect -> key_rects / white_w / black_w / black_h."""

    def __init__(self, rect, low=LOWEST_PIANO_KEY, high=HIGHEST_PIANO_KEY, style=None):
        self.low, self.high = low, high
        self.style = style or key_style()
        self.layout(rect)

    def layout(self, rect):
        import pygame

        self.rect = pygame.Rect(rect)
        self._base = None
        self.tails = {}              # equal keys only: pitch -> Rect of a white key's back
        self.lane_shade = []         # equal keys only: (x, width) of the lanes above white keys
        self.style = "realistic"
        whites = [p for p in range(self.low, self.high + 1) if not is_black_key(p)]
        self.white_w = rect.w / len(whites)
        self.black_w = self.white_w * 0.6
        self.black_h = int(rect.h * 0.63)

        self.key_rects = {}          # pitch -> pygame.Rect of the key itself
        self.lanes = {}              # pitch -> (x, width) for falling notes
        self.lane_lines = []         # x positions of faint guides at each C and F
        white_index = 0
        for p in range(self.low, self.high + 1):
            if is_black_key(p):
                boundary = rect.x + white_index * self.white_w
                x = boundary - self.black_w / 2 + _BLACK_OFFSETS[p % 12] * self.black_w
                r = pygame.Rect(round(x), rect.y, round(self.black_w), self.black_h)
                self.key_rects[p] = r
                self.lanes[p] = (r.x, r.w)
            else:
                x0 = round(rect.x + white_index * self.white_w)
                x1 = round(rect.x + (white_index + 1) * self.white_w)
                self.key_rects[p] = pygame.Rect(x0, rect.y, x1 - x0, rect.h)
                pad = max(1, round(self.white_w * 0.08))
                self.lanes[p] = (x0 + pad, x1 - x0 - 2 * pad)
                if p % 12 in (0, 5):
                    self.lane_lines.append(x0)
                white_index += 1