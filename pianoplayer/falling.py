"""Synthesia-style falling notes: pure numpy geometry (no VTK import here)."""
from __future__ import annotations

import numpy as np

HIT_Y = 4.85            # world y of the landing line (top edge of the fall-line box)
FALL_SECONDS = 2.2      # a note is visible this long before it must be struck
BAR_GAP = 0.18          # cm trimmed off the top of a bar so abutting notes stay apart
MIN_BAR = 0.45          # cm, shortest bar drawn
LABEL_H = 1.7           # cm, finger digit height
MIN_LABEL_H = 0.7
LABEL_PAD = 0.15        # cm between bar edge and digit
STROKE = 0.075          # digit stroke half-width as a fraction of digit height
DIGIT_W = 0.62          # digit width / height

# polylines in a unit box, x right, y DOWN (as a font designer would draw)
_STROKES = {
    1: [[(0.30, 0.25), (0.55, 0.08), (0.55, 0.92)]],
    2: [[(0.2, 0.28), (0.3, 0.12), (0.5, 0.06), (0.7, 0.12), (0.78, 0.28),
         (0.7, 0.45), (0.2, 0.92), (0.82, 0.92)]],
    3: [[(0.2, 0.15), (0.5, 0.06), (0.75, 0.18), (0.75, 0.38), (0.5, 0.50),
         (0.78, 0.62), (0.78, 0.82), (0.5, 0.94), (0.2, 0.85)]],
    4: [[(0.68, 0.92), (0.68, 0.06), (0.15, 0.65), (0.85, 0.65)]],
    5: [[(0.75, 0.08), (0.28, 0.08), (0.24, 0.46), (0.5, 0.40), (0.72, 0.48),
         (0.78, 0.68), (0.68, 0.88), (0.46, 0.94), (0.22, 0.86)]],
}
MAX_QUADS = 16          # segments + joint squares of the busiest digit (5 -> 15)


def fall_speed(top_y, hit_y=HIT_Y, seconds=FALL_SECONDS):
    """cm per second so a note enters at the top edge exactly `seconds` early."""
    return (top_y - hit_y) / float(seconds)


def bar_extent(t_on, t_off, now, speed, hit_y=HIT_Y, top_y=1e9):
    """(y0, y1) of the visible part of a bar, or None if nothing is visible.

    The bar's lower edge touches `hit_y` exactly at `t_on`; its upper edge
    follows `t_off`.  Below the line it is clipped away (it "enters" the key).
    """
    y0 = hit_y + (t_on - now) * speed
    y1 = hit_y + (t_off - now) * speed
    if y1 - y0 > 2.0 * BAR_GAP:
        y1 -= BAR_GAP
    y1 = max(y1, y0 + MIN_BAR)
    y0 = max(y0, hit_y)
    y1 = min(y1, top_y)
    if y1 - y0 < 1e-6 or y0 >= top_y:
        return None
    return y0, y1


def label_extent(y0, y1):
    """(ly0, height) of the digit inside the visible bar, or None to hide it."""
    h = min(LABEL_H, 0.8 * (y1 - y0))
    if h < MIN_LABEL_H:
        return None
    ly0 = y0 + LABEL_PAD
    if ly0 + h > y1 - 0.05:
        return None
    return ly0, h


def digit_template(d):
    """(MAX_QUADS*4, 2) float array of quad corners in the unit box, y UP.

    One thin rectangle per polyline segment plus one small square on every
    interior vertex (hides the notches at the bends).  Unused quads are
    collapsed onto one point so they have zero area.
    """
    out = np.zeros((MAX_QUADS * 4, 2))
    k = 0
    t = STROKE
    for line in _STROKES.get(int(d), []):
        pts = np.array([(x, 1.0 - y) for x, y in line])
        for a, b in zip(pts[:-1], pts[1:]):
            v = b - a
            n = float(np.hypot(*v))
            nx, ny = -v[1] / n * t, v[0] / n * t
            out[4 * k:4 * k + 4] = [a + (nx, ny), b + (nx, ny),
                                    b - (nx, ny), a - (nx, ny)]
            k += 1
        for c in pts[1:-1]:
            out[4 * k:4 * k + 4] = [c + (-t, -t), c + (t, -t),
                                    c + (t, t), c + (-t, t)]
            k += 1
    assert k <= MAX_QUADS, k
    if k < MAX_QUADS:
        out[4 * k:] = out[0]
    return out


def place_digit(template, cx, ly0, h, z):
    """(MAX_QUADS*4, 3) world vertices for a digit centred on cx."""
    p = np.empty((template.shape[0], 3))
    p[:, 0] = cx + (template[:, 0] - 0.5) * DIGIT_W * h
    p[:, 1] = ly0 + template[:, 1] * h
    p[:, 2] = z
    return p


# --------------------------------------------------------------------------- #
# VTK layer (imports VTK lazily so everything above works without a window)
# --------------------------------------------------------------------------- #
PARK = (0.0, 0.0, -50.0)    # where unused quads are collapsed (zero area)
BAR_Z = 0.60                # above the keys' slab, below the digits
DIGIT_Z = 0.70
DARK_DIGIT = (0.05, 0.05, 0.07)


def _quad_mesh(n_quads, with_rgb):
    """PolyData of `n_quads` independent quads + numpy views of its buffers."""
    from vedo import Mesh
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    pts = np.tile(np.array(PARK, np.float64), (n_quads * 4, 1))
    pd = vtkPolyData()
    vpts = vtkPoints()
    vpts.SetData(numpy_to_vtk(pts, deep=1))
    pd.SetPoints(vpts)
    cells = vtkCellArray()
    for i in range(n_quads):
        cells.InsertNextCell(4)
        for j in range(4):
            cells.InsertCellPoint(4 * i + j)
    pd.SetPolys(cells)
    mesh = Mesh(pd)
    pd = mesh.dataset                      # same object, or vedo's copy: use this one
    vpts = pd.GetPoints()
    pbuf = vtk_to_numpy(vpts.GetData())    # numpy VIEW of the VTK buffer
    rgb = rgb_arr = None
    if with_rgb:
        rgb = np.zeros((n_quads * 4, 3), np.uint8)
        rgb_arr = numpy_to_vtk(rgb, deep=0)   # shares memory with `rgb`
        rgb_arr.SetName("RGB")
        pd.GetPointData().SetScalars(rgb_arr)
        mesh.mapper.SetColorModeToDirectScalars()
        mesh.color(None)
    try:
        mesh.actor.GetProperty().LightingOff()
    except Exception:
        pass
    return mesh, pd, vpts, pbuf, rgb, rgb_arr


class FallingNotes:
    """Bars + finger digits above the keys.  Two meshes, zero per-frame allocation."""

    def __init__(self, plotter, top_y, x_of, accent_r, accent_l, key_w,
                 white_w, black_w, capacity=384, seconds=FALL_SECONDS):
        self.vp = plotter
        self.top_y = float(top_y)
        self.speed = fall_speed(self.top_y, HIT_Y, seconds)
        self.x_of = x_of                  # callable(ev) -> world x of its key
        self.col = {"R": np.array(accent_r), "L": np.array(accent_l)}
        self.white_w, self.black_w = white_w, black_w
        self.cap = int(capacity)
        self.tpl = {d: digit_template(d) for d in range(1, 6)}
        (self.bars, self.bars_pd, self.bars_vpts, self.bars_p,
         self.bars_rgb, self.bars_rgb_arr) = _quad_mesh(self.cap, True)
        (self.digs, self.digs_pd, self.digs_vpts, self.digs_p,
         _, _) = _quad_mesh(self.cap * MAX_QUADS, False)
        self.digs.color(DARK_DIGIT)
        self.vp += self.bars
        self.vp += self.digs
        self._n_bars = 0                  # quads written last frame
        self._n_digs = 0
        self._lo = {}                     # per hand: first note not yet finished

    def reset(self):
        self._lo = {}

    def update(self, now, players):
        nb = nd = 0
        for p in players:
            notes = p.notes
            lo = self._lo.get(p.side, 0)
            while lo < len(notes) and notes[lo].v_off <= now:
                lo += 1
            self._lo[p.side] = lo
            base = self.col[p.side]
            for i in range(lo, len(notes)):
                e = notes[i]
                if e.t_on - now > FALL_SECONDS:
                    break                              # sorted by t_on
                if e.v_off <= now or nb >= self.cap:
                    continue
                ext = bar_extent(e.t_on, e.v_off, now, self.speed, HIT_Y, self.top_y)
                if ext is None:
                    continue
                y0, y1 = ext
                x = self.x_of(e)
                hw = 0.5 * (self.black_w if e.black else self.white_w)
                self.bars_p[4 * nb:4 * nb + 4] = (
                    (x - hw, y0, BAR_Z), (x + hw, y0, BAR_Z),
                    (x + hw, y1, BAR_Z), (x - hw, y1, BAR_Z))
                c = base * (0.78 if e.black else 1.0)
                self.bars_rgb[4 * nb:4 * nb + 4] = np.clip(c * 255.0 + 0.5, 0, 255).astype(np.uint8)
                nb += 1
                if e.finger in self.tpl:
                    lab = label_extent(y0, y1)
                    if lab is not None and nd < self.cap:
                        s = nd * MAX_QUADS * 4
                        self.digs_p[s:s + MAX_QUADS * 4] = place_digit(
                            self.tpl[e.finger], x, lab[0], lab[1], DIGIT_Z)
                        nd += 1
        # park whatever was drawn last frame but not now
        if nb < self._n_bars:
            self.bars_p[4 * nb:4 * self._n_bars] = PARK
        if nd < self._n_digs:
            self.digs_p[nd * MAX_QUADS * 4:self._n_digs * MAX_QUADS * 4] = PARK
        self._n_bars, self._n_digs = nb, nd
        for vpts, pd, extra, mesh in (
                (self.bars_vpts, self.bars_pd, self.bars_rgb_arr, self.bars),
                (self.digs_vpts, self.digs_pd, None, self.digs)):
            vpts.Modified()
            if extra is not None:
                extra.Modified()
            pd.Modified()
            mesh.actor.Modified()
