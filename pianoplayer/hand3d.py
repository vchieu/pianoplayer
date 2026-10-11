"""
Procedural anatomical hand for the 3D piano player.

The hand is only three draw calls:

* **arm + palm** -- a single lofted surface.  A superelliptical cross-section
  is swept from the far end of the forearm, through the wrist, across the palm
  and up to the knuckle line; the thenar (thumb) and hypothenar (pinky)
  eminences are angular bulges of that same section, so there is no seam
  where the thumb muscle used to be a separate ellipsoid.  It is rigid: the
  whole surface is moved by one matrix.
* **five fingers** -- one tapered tube each.  The tube is a surface of
  revolution swept along the *solved* bone chain, so it bends continuously
  instead of being three cylinders with spheres in between.  Its points and
  normals are rewritten in place every frame from a pair of numpy buffers
  (no allocation, no filter).
* **five nails** -- small ellipsoids riding on the back of each fingertip.

Every frame the rig is re-posed with a small analytic IK solver:

* the chain (MCP -> PIP -> DIP -> tip) bends in the plane spanned by the
  chord (target - base) and a "pole" vector pointing upwards, so the PIP
  joint arches above the straight line exactly like a real finger does;
* the PIP bend is the dominant one and the DIP follows it with
  ``b2 = COUPLING * b1`` (a real finger is not a 3-joint snake);
* the bend angle is found by bisection on the chord length, so the tip
  lands *exactly* on the key whenever the target is inside reach.

Local hand frame (before mirroring for the left hand):

    origin : wrist centre
    +y     : toward the fingers (into the keyboard)
    +x     : toward the pinky (right hand)
    +z     : up (dorsal side)
"""

from __future__ import annotations

import math

import numpy as np
from vedo import Ellipsoid, Mesh
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonMath import vtkMatrix4x4

from pianoplayer.utils import (
    BLACK_TOP,
    CONTACT_Y_BLACK,
    CONTACT_Y_WHITE,
    KEY_WB,
    KEYBSIZE,
    WHITE_TOP,
    handSizeFactor,
    key_drop,
    key_is_black,
    key_top,
)

# --------------------------------------------------------------------------- #
# looks
# --------------------------------------------------------------------------- #
SKIN = (0.93, 0.73, 0.60)
SKIN_PRESS = (0.98, 0.57, 0.48)
NAIL = (0.94, 0.82, 0.79)
NAIL_PRESS = (0.90, 0.58, 0.56)

FINGER_NAMES = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "little"}
FINGERS = (1, 2, 3, 4, 5)

# D1: wrist-root placement scales with hand size so every size XXS..XXL
# reaches the keys with a natural curl (tuned against test_hand_reach.py:
# tip_err < 0.05 cm, max_bend < 1.3 rad for all 70 size/finger/key cells).
# The plan's LEVEL (contact - 8.6*hf - 0.62*finger_len) proved infeasible as
# written: the thumb (MCP 6.2 cm behind the fingers, only 0.6 cm longer) runs
# out of reach on big hands while the middle finger over-folds on black keys,
# and no single root translation satisfies both (see Deviations).  Kept shape:
_ROOT_CONTACT_Y = -3.0    # mean CONTACT_Y_WHITE: where fingertips meet the keys
_ROOT_WRIST_GAP = 8.6     # wrist-to-MCP span carried by the palm (x hf)
_ROOT_REACH_FRAC = 0.69   # tuned 0.62 -> 0.69: MCP standoff as a fraction of
                          # the middle-finger length
_ROOT_Z_BASE = 2.0        # ROOT_Z = BASE * (C0 + K*hf); tuned from
_ROOT_Z_C0 = 0.50         #   2.3 * (0.6 + 0.4*hf): lower for small hands
_ROOT_Z_HF_K = 0.60       #   (dz dominates), higher for large ones (bend room)
# D1b: the thumb MCP sits progressively further forward as the hand grows;
# without this the thumb cannot reach either key colour at L/XL/XXL while the
# middle finger must not over-fold (same root serves both).  XXS keeps the
# as-shipped 2.4; the shift is in world cm so small hands are untouched.
_THUMB_BASE_Y0 = 2.4
_THUMB_FWD = 3.6           # thumb base y += FWD * (hf - HF_XXS)
_HF_XXS = 0.33
# D1c: the pinky is ~2*hf shorter than the middle finger but starts from
# almost the same knuckle, so it runs out of reach on white keys at every
# size.  Its contact point sits this much closer to the player (world cm,
# both colours; still on the key).
_PINKY_CONTACT_DY = 0.80
# D3: arm yaw around the wrist so the forearm angles outward toward the
# elbow.  shoulder_x = kb_center +/- 15 (right hand's shoulder sits right of
# centre, left hand's left of it); the 88-key span centre is fixed geometry.
_KB_CENTER = 4.0 * KEYBSIZE + 3.0 * KEY_WB   # (_key_x(21)+_key_x(108))/2
_SHOULDER_OFF = 15.0
_YAW_ARM_LEN = 45.0
_YAW_ARM_MAX = 0.45       # rad, clamp on the arm-yaw part (velocity yaw adds)

# --------------------------------------------------------------------------- #
# geometry of the hand, in cm, for a size-L (hf = 1.0) right hand
#   base : MCP joint position in the local hand frame
#   lens : MCP->PIP, PIP->DIP, DIP->tip lengths
#   radii: MCP, PIP, DIP, fingertip radii
#   pole : preferred side the finger arches toward (upwards)
# --------------------------------------------------------------------------- #
_SPECS = {
    1: dict(base=(-5.6, 2.4, 0.45), lens=(4.2, 3.0, 2.5),
            radii=(1.06, 0.96, 0.83, 0.70), pole=(0.15, 0.55, 1.00)),
    2: dict(base=(-2.1, 8.2, 1.85), lens=(3.9, 2.5, 2.0),
            radii=(0.96, 0.89, 0.77, 0.65), pole=(0.00, 0.35, 1.00)),
    3: dict(base=(0.15, 8.6, 1.95), lens=(4.3, 2.7, 2.1),
            radii=(0.98, 0.91, 0.79, 0.66), pole=(0.00, 0.35, 1.00)),
    4: dict(base=(2.4, 8.3, 1.80), lens=(3.9, 2.5, 2.0),
            radii=(0.93, 0.86, 0.74, 0.62), pole=(0.00, 0.35, 1.00)),
    5: dict(base=(4.5, 7.7, 1.60), lens=(3.2, 2.1, 1.8),
            radii=(0.86, 0.79, 0.68, 0.57), pole=(0.00, 0.30, 1.00)),
}

# arm + palm loft: (y, centre-x, half-width, half-thickness, superellipse p)
# p = 2 is a circle, larger is flatter-sided like the edge of a palm.
# The far end stays long whatever the hand size: on screen the arm has to
# leave the frame, otherwise it looks like a cut-off tube floating in air.
_ARM = (
    (-46.0, 0.00, 3.60, 2.70, 2.00),
    (-30.0, 0.00, 3.40, 2.60, 2.00),
    (-16.0, 0.00, 3.00, 2.30, 2.05),
    (-9.0, 0.00, 2.65, 1.90, 2.15),
    (-4.5, 0.00, 2.50, 1.55, 2.35),
    (-1.8, 0.00, 2.55, 1.40, 2.55),      # wrist
    (0.8, -0.35, 3.20, 1.36, 2.80),
    (3.2, -0.70, 4.10, 1.48, 3.00),
    (5.8, -0.10, 4.55, 1.46, 3.10),
    (8.2, 1.00, 4.30, 1.32, 3.05),      # knuckle line: span -3.3 .. 5.3
    (9.2, 1.00, 3.90, 0.95, 2.80),
    (10.0, 1.00, 0.60, 0.55, 2.20),     # closed cap, ~0.8 cm shorter
)
_ARM_PIN = 2         # first stations keep their y (they are the out-of-frame arm)
_THENAR = 0.50       # thumb-side bulge, over the palm only (D2: 0.34 -> 0.50,
                     # thenar sits lower at y~=2..5, clear of the knuckles)
_HYPO = 0.12         # pinky-side bulge
_ARM_M = 26          # points around an arm section

_FING_M = 14         # points around a finger section
# rounded fingertip: stations at these fractions of the cap
_TIP_U = (0.0, 0.30, 0.55, 0.75, 0.89, 0.965)
# D4: finger cross-section is an ellipse, flatter on the back, with a
# fuller belly (pad side) so light models a real fingertip, not a tube.
_FING_FLAT = 0.88    # z-radius scale of the section ellipse
_FING_BELLY = 0.85   # extra z scale on the pad side (sin < 0)
# D5: resting fingers curl instead of hovering flat (robot look).
_REST_LIFT = {1: 0.4, 2: 0.9, 3: 1.1, 4: 1.0, 5: 0.8}  # hover height per finger
# resting fingertip retreats toward the wrist (curl): the thumb needs the
# full pull (its MCP sits far behind the fingers) while the fingers only
# need a hint -- a uniform 0.8 hooks them in chord poses (measured 1.38 rad).
_REST_PULL_Y = {1: 1.0, 2: 0.35, 3: 0.35, 4: 0.35, 5: 0.5}
_REST_PULL_Z = 0.6    # ... and rises, so the finger arches instead of splaying
# rest offsets are world-cm tuned at M (hf=0.82): scale them with hand size,
# otherwise tiny hands hyper-extend (starfish) and huge hands over-curl.
_REST_REF_HF = 0.82
_BREATHE_AMP = 0.08   # "breathing" oscillation amplitude [cm]
_BREATHE_FREQ = 0.35  # ... frequency [Hz]
_BREATHE_PHASE = {1: 0.0, 2: 1.3, 3: 2.1, 4: 3.4, 5: 4.2}  # per-finger phase
# C4: when a finger holds a key (press > 0.5) but its tip is off (> 0.1 cm),
# the root chases with this faster time constant for that frame only.
_CHASE_TAU = 0.03
_CHASE_ERR = 0.1
# D6: looks -- skin material, joint shading, fake fingertip shadows.
_SKIN_AMBIENT = 0.35
_SKIN_DIFFUSE = 0.75
_SKIN_SPECULAR = 0.08
_NAIL_SPECULAR = 0.4
_NAIL_SL = 0.45          # nail length as a fraction of the distal phalanx
# joint rings shade this much darker/pinker than mid-bone (vertex colours).
_JOINT_SHADE = (0.93, 0.895, 0.905)
_JOINT_WIDTH = 0.18       # gaussian width (local t) of the joint shading
# fake contact shadows: flat dark discs under the fingertips (the real
# shadow-map is off).  Opacity fades as the fingertip rises.  Tuned small
# and faint: oversized discs read as dirt, not shadow.
_SHADOW_OPACITY = 0.22
_SHADOW_SX = 1.2          # x half-width as a multiple of fingertip radius
_SHADOW_SY = 1.6          # y half-width ditto
_SHADOW_OFF = (0.4, -0.4)  # offset from the fingertip, toward the light
_SHADOW_FADE_H = 2.5       # fingertip height [cm] at which the shadow is gone
_SHADOW_FLOOR = 0.05
_SHADOW_COL = (0.04, 0.025, 0.025)
# D4: static thenar-web ellipsoid filling the thumb/index wedge, in palm
# frame (radii x hf, set in _build; centre y follows the thumb MCP in
# __init__ because D1b moves it forward with hand size).  Tuned so it just
# bridges palm edge -> thumb tube without doming out of the silhouette.
_WEB_RADII = (1.20, 2.40, 1.10)
_WEB_POS_X = -3.90    # midpoint of thumb (-5.6) and index (-2.1) MCP x
_WEB_POS_Z = 0.70

_UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# small maths helpers
# --------------------------------------------------------------------------- #
def _frame_from_dir(d, up=(0.0, 0.0, 1.0)):
    """Orthonormal frame whose +y axis is `d` and whose +z axis is as close
    as possible to `up` (used to orient the nail on a fingertip).

    Returns a 3x3 nested tuple, laid out like ``np.column_stack((x, y, z))``
    i.e. ``out[i][j]`` = i-th component of the j-th axis.
    """
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        dx, dy, dz = 0.0, 1.0, 0.0
    else:
        dx, dy, dz = dx / n, dy / n, dz / n

    ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
    xx = dy * uz - dz * uy                 # x = y x up
    xy = dz * ux - dx * uz
    xz = dx * uy - dy * ux
    nx = math.sqrt(xx * xx + xy * xy + xz * xz)
    if nx < 1e-6:
        for ax, ay, az in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            xx = dy * az - dz * ay
            xy = dz * ax - dx * az
            xz = dx * ay - dy * ax
            nx = math.sqrt(xx * xx + xy * xy + xz * xz)
            if nx > 1e-6:
                break
    xx, xy, xz = xx / nx, xy / nx, xz / nx

    zx = xy * dz - xz * dy                 # z = x x y
    zy = xz * dx - xx * dz
    zz = xx * dy - xy * dx
    return ((xx, dx, zx), (xy, dy, zy), (xz, dz, zz))


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def solve_chain(base, target, lens, pole, bend_max=1.75, coupling=0.4):
    """
    Analytic 3-segment IK.

    Returns ``(pts, b1)``: ``pts`` is ``[p0, p1, p2, p3]`` (p0 is `base`,
    p3 is `target` when the target is inside reach, otherwise the closest
    reachable point) and ``b1`` is the PIP bend angle [rad] (DIP bends by
    ``coupling * b1``).  The chain bends toward `pole`, PIP bending by
    ``b1`` and DIP by ``coupling*b1``.
    """
    bx, by, bz = float(base[0]), float(base[1]), float(base[2])
    vx = float(target[0]) - bx
    vy = float(target[1]) - by
    vz = float(target[2]) - bz
    d = math.sqrt(vx * vx + vy * vy + vz * vz)
    if d < 1e-9:
        vx, vy, vz = float(pole[0]), float(pole[1]), float(pole[2])
        d = math.sqrt(vx * vx + vy * vy + vz * vz)
        if d < 1e-9:
            vx, vy, vz, d = 0.0, 1.0, 0.0, 1.0
    ux, uy, uz = vx / d, vy / d, vz / d

    # pole, orthogonalised against the chord -> in-plane "up" direction
    px, py, pz = float(pole[0]), float(pole[1]), float(pole[2])
    dp = px * ux + py * uy + pz * uz
    px -= dp * ux
    py -= dp * uy
    pz -= dp * uz
    n = math.sqrt(px * px + py * py + pz * pz)
    if n < 1e-6:
        if abs(ux) < 0.9:
            ax, ay, az = 1.0, 0.0, 0.0
        else:
            ax, ay, az = 0.0, 1.0, 0.0
        px = uy * az - uz * ay
        py = uz * ax - ux * az
        pz = ux * ay - uy * ax
        n = math.sqrt(px * px + py * py + pz * pz)
        if n < 1e-6:
            px, py, pz, n = 0.0, 0.0, 1.0, 1.0
    wx, wy, wz = px / n, py / n, pz / n

    l1, l2, l3 = lens

    def shape(b1):
        # the chain flexes clockwise (finger flexion, palm side down):
        # the joints then arch *above* the chord, toward the pole vector
        b2 = coupling * b1
        a = b1 + b2
        p1 = (l1, 0.0)
        p2 = (p1[0] + l2 * math.cos(b1), p1[1] - l2 * math.sin(b1))
        p3 = (p2[0] + l3 * math.cos(a), p2[1] - l3 * math.sin(a))
        return p1, p2, p3

    def reach(b1):
        _, _, q = shape(b1)
        return math.hypot(q[0], q[1])

    # |T(b1)| decreases monotonically on [0, bend_max] -> plain bisection
    dc = min(max(d, reach(bend_max)), l1 + l2 + l3)
    lo, hi = 0.0, bend_max
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if reach(mid) > dc:
            lo = mid
        else:
            hi = mid
    b1 = 0.5 * (lo + hi)
    p1, p2, p3 = shape(b1)

    phi = -math.atan2(p3[1], p3[0])
    c, s = math.cos(phi), math.sin(phi)
    pts = []
    for q in ((0.0, 0.0), p1, p2, p3):
        x = q[0] * c - q[1] * s
        y = q[0] * s + q[1] * c
        pts.append((bx + x * ux + y * wx,
                    by + x * uy + y * wy,
                    bz + x * uz + y * wz))
    return pts, b1


# --------------------------------------------------------------------------- #
# surface builders
# --------------------------------------------------------------------------- #
def _hump(y, a, b):
    """Smooth 0..1..0 bump over the y interval [a, b]."""
    u = (y - a) / (b - a)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return math.sin(math.pi * u) ** 1.4


def _adist(th, a):
    """Minimal angular distance between `th` and `a`, elementwise."""
    return np.abs((th - a + math.pi) % (2.0 * math.pi) - math.pi)


def _frames(T):
    """Orthonormal side axes for a stack of unit tangents `T` (n,3).

    x is kept as close as possible to the world +x axis so the dorsal side of
    every finger keeps facing up; z = x x T then points out of the back of the
    hand.  Returns (X, Z), each (n,3).
    """
    X = np.cross(T, _UP)
    n = np.linalg.norm(X, axis=1)
    bad = n < 1e-6
    if bad.any():
        X[bad] = np.cross(T[bad], np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(X, axis=1)
        n[n < 1e-9] = 1.0
    X /= n[:, None]
    return X, np.cross(X, T)


def _tube_faces(K, M):
    """Ring quads plus a triangle fan closing both ends.

    The winding runs with increasing theta, which -- given that x is built
    first and z second -- makes the geometric normal point *into* the tube.
    VTK then sees every outer face as back-facing and inverts its normal in
    the fragment shader, so a correctly outward normal would shade black.
    Reversing each cell keeps the faces front-facing instead.
    """
    faces = []
    for k in range(K - 1):
        for m in range(M):
            a = k * M + m
            faces.append(((k + 1) * M + m, (k + 1) * M + (m + 1) % M,
                          k * M + (m + 1) % M, a))
    base, tip = K * M, K * M + 1
    last = (K - 1) * M
    for m in range(M):
        faces.append((base, (m + 1) % M, m))
        faces.append((tip, last + (m + 1) % M, last + m))
    return faces


def _material(mesh, ambient, diffuse, specular):
    """D6: set the VTK lighting coefficients (kept separate from colours)."""
    try:
        p = mesh.actor.GetProperty()
        p.SetAmbient(ambient)
        p.SetDiffuse(diffuse)
        p.SetSpecular(specular)
    except Exception:
        pass


def _skin_mesh(pts, nrm, faces, color):
    """Wrap a surface so that its point and normal buffers are ours to fill.

    Both arrays end up VTK-owned (the OpenGL mapper insists on that) and we
    keep a numpy view of each: re-posing a finger is then nothing but writing
    into two arrays and calling Modified() - no allocation, no filter.
    """
    pd = vtkPolyData()
    vpts = vtkPoints()
    vpts.SetData(numpy_to_vtk(np.array(pts, np.float64), deep=1))
    pd.SetPoints(vpts)

    cells = vtkCellArray()
    for fc in faces:
        cells.InsertNextCell(len(fc))
        for i in fc:
            cells.InsertCellPoint(int(i))
    pd.SetPolys(cells)

    mesh = Mesh(pd)
    mesh.color(color)
    mesh.compute_normals(cells=False)   # hands VTK an array we can write into
    pd = mesh.dataset
    vpts = pd.GetPoints()
    ndata = pd.GetPointData().GetNormals()
    pbuf = vtk_to_numpy(vpts.GetData())
    nbuf = vtk_to_numpy(ndata)
    nbuf[:] = nrm
    # writing through a numpy view does not bump the VTK timestamp: if we
    # skip this the mapper keeps serving the normals it first saw (zeros for
    # a finger) and the surface shades pure black.
    ndata.Modified()
    vpts.Modified()
    return mesh, pbuf, nbuf, (vpts, ndata, pd)


def _arm_surface(mirror, hf, M):
    """Points and normals of the lofted arm+palm, in world-local cm."""
    th = 2.0 * math.pi * np.arange(M) / M
    ct, st = np.cos(th), np.sin(th)

    n = len(_ARM)
    off_x = np.zeros((n, M))
    off_z = np.zeros((n, M))
    dir_u = np.zeros((n, M))     # section normal, x part (unit section space)
    dir_w = np.zeros((n, M))     # section normal, z part
    cx = np.zeros(n)
    ys = np.zeros(n)

    for k, (y, c, a1, a2, p) in enumerate(_ARM):
        e = 2.0 / p
        x = a1 * (np.sign(ct) * np.abs(ct) ** e)
        z = a2 * (np.sign(st) * np.abs(st) ** e)
        # thenar / hypothenar eminences: local bulges of the same section
        ta = _THENAR * _hump(y, -0.5, 5.5)
        hy = _HYPO * _hump(y, 2.5, 9.5)
        if ta:
            x -= a1 * ta * np.exp(-(_adist(th, math.pi) / 0.62) ** 2)
        if hy:
            x += a1 * hy * np.exp(-(_adist(th, 0.0) / 0.62) ** 2)
        off_x[k], off_z[k] = x, z
        # in *unscaled* section coordinates, so that scaling the surface by
        # hf cannot skew the normal as well as the shape
        dir_u[k] = x / max(a1, 1e-6)
        dir_w[k] = z / max(a2, 1e-6)
        cx[k], ys[k] = c * hf, (y if k < _ARM_PIN else y * hf)

    C = np.stack([cx, ys, np.zeros(n)], axis=1)
    T = np.empty_like(C)
    T[1:-1] = C[2:] - C[:-2]
    T[0] = C[1] - C[0]
    T[-1] = C[-1] - C[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    X, Z = _frames(T)

    V = (C[:, None, :] + X[:, None, :] * (off_x * hf)[:, :, None]
         + Z[:, None, :] * (off_z * hf)[:, :, None])
    N = X[:, None, :] * dir_u[:, :, None] + Z[:, None, :] * dir_w[:, :, None]
    N /= np.linalg.norm(N, axis=2, keepdims=True)

    pts = V.reshape(-1, 3)
    nrm = N.reshape(-1, 3)
    pts = np.concatenate([pts, [C[0], C[-1]]], axis=0)
    nrm = np.concatenate([nrm, [-T[0], T[-1]]], axis=0)
    pts = np.ascontiguousarray(pts)
    nrm = np.ascontiguousarray(nrm)
    faces = _tube_faces(n, M)
    if mirror < 0:
        pts[:, 0] *= -1.0
        nrm[:, 0] *= -1.0
        # a reflection has a negative determinant, so it reverses the face
        # orientation: without this every outer face of the left hand would be
        # back-facing and VTK would invert its normal (shading it black).
        faces = [tuple(reversed(f)) for f in faces]
    return pts, nrm, faces


def _finger_radius(seg, t, radii, t0):
    """Tube radius at a station (segment index, local fraction)."""
    r0, r1, r2, r3 = radii
    if seg == 0:
        r = r0 + (r1 - r0) * t
        r *= 1.0 + 0.13 * math.exp(-((t - 0.13) / 0.17) ** 2)   # MCP knuckle
        r *= 1.0 + 0.25 * math.exp(-(t / 0.22) ** 2)            # web thickening
        return r
    if seg == 1:
        r = r1 + (r2 - r1) * t
        # the joint creases pinch very slightly, the way a knuckle does
        r *= 1.0 - 0.05 * math.exp(-(t / 0.22) ** 2) \
                 - 0.05 * math.exp(-((t - 1.0) / 0.22) ** 2)
        return r
    if t <= t0:
        return r2 + (r3 - r2) * (t / t0)
    u = min(1.0, (t - t0) / (1.0 - t0))
    return r3 * math.sqrt(max(0.0, 1.0 - u * u))     # hemispherical pulp


def _finger_stations(lens, radii):
    """Stations along the bone chain: segment, local fraction, arc length,
    radius.  Denser at the joints and at the fingertip, where it curves."""
    l1, l2, l3 = lens
    t0 = max(0.35, 1.0 - radii[3] / l3)      # where the rounded tip starts
    ts = ([(0, t) for t in (0.00, 0.16, 0.34, 0.54, 0.76, 1.00)]
          + [(1, t) for t in (0.30, 0.62, 1.00)]
          + [(2, t0 + (1.0 - t0) * u) for u in _TIP_U])

    seg = np.array([s for s, _ in ts], np.int64)
    t = np.array([v for _, v in ts], np.float64)
    lens_arr = (l1, l2, l3)
    s = np.array([sum(lens_arr[:sg]) + v * lens_arr[sg] for sg, v in ts])
    r = np.array([_finger_radius(sg, v, radii, t0) for sg, v in ts])
    return seg, t, s, r


# --------------------------------------------------------------------------- #
# the rig
# --------------------------------------------------------------------------- #
class HandRig:
    """Builds a procedural hand and re-poses it every frame."""

    ROOT_Y = -13.5            # wrist sits behind the front edge of the keys
    ROOT_Z = 2.3
    BLACK_SHIFT = 1.15         # how far the hand moves into the keyboard
                               # (D1: tuned 1.5 -> 1.15; less over-fold on black)
    HOVER = 0.55              # fingertip/key gap while a finger waits
    KEY_SINK = 0.10           # fingertip slightly above the pressed key
    TAU = 0.065               # root spring time constant [s]
    YAW_GAIN = 0.006          # rad of yaw per cm/s of hand speed
    YAW_MAX = 0.16
    PITCH = math.radians(-5.0)
    ROLL = math.radians(5.0)  # sign follows the hand side
    BEND_MAX = 1.75
    COUPLING = 0.4

    def __init__(self, side="right", size="M", skin=SKIN, kb_center=None):
        self.side = "left" if str(side).lower().startswith("l") else "right"
        self.s = 1.0 if self.side == "right" else -1.0
        self.hf = float(handSizeFactor(size))
        self.skin = tuple(skin)
        # D3: keyboard centre for the arm-yaw shoulder reference (the 88-key
        # span is fixed geometry, so callers need not pass it).
        self.kb_center = _KB_CENTER if kb_center is None else float(kb_center)

        # D1: scale the wrist root with hand size (class attrs stay as the
        # as-shipped hf=1.0 reference).
        finger_len = sum(_SPECS[3]["lens"]) * self.hf
        self.ROOT_Y = (_ROOT_CONTACT_Y - _ROOT_WRIST_GAP * self.hf
                       - _ROOT_REACH_FRAC * finger_len)
        self.ROOT_Z = _ROOT_Z_BASE * (_ROOT_Z_C0
                                      + _ROOT_Z_HF_K * self.hf)
        # D1b: thumb MCP forward shift (world cm, progressive with size).
        self._thumb_by = max(_THUMB_BASE_Y0,
                             _THUMB_BASE_Y0 + _THUMB_FWD * (self.hf - _HF_XXS))
        # D4: thenar-web centre in palm frame (follows the thumb MCP).
        self._web_local = self._L(
            (_WEB_POS_X, 0.5 * (self._thumb_by + 8.2), _WEB_POS_Z))

        self.root = np.array([0.0, self.ROOT_Y, self.ROOT_Z])
        self.yaw = 0.0
        self._vx = 0.0
        self._t = 0.0               # D5: animation clock for the breathing
        self._targets_x = {}
        self.tip_err = {}     # finger -> cm between true q[3] and its target
        self.max_bend = {}    # finger -> PIP bend angle b1 [rad] of last solve
        self._col = {}
        self._geom = {}
        self.nail = {}
        self.shadow = {}
        self._actors = []

        self._build()

    # ------------------------------------------------------------------ build
    def _L(self, p):
        """Local point: mirror x for the left hand, scale by hand size."""
        return np.array([p[0] * self.s * self.hf, p[1] * self.hf, p[2] * self.hf])

    def _place(self, mesh):
        m = vtkMatrix4x4()
        m.Identity()
        mesh.actor.SetUserMatrix(m)
        mesh._rig_matrix = m
        self._actors.append(mesh)
        return mesh

    def _build(self):
        # arm + palm: one lofted surface, moved as a whole --------------
        pts, nrm, faces = _arm_surface(self.s, self.hf, _ARM_M)
        self.arm, _, _, _ = _skin_mesh(pts, nrm, faces, self.skin)
        self._place(self.arm)
        _material(self.arm, _SKIN_AMBIENT, _SKIN_DIFFUSE, _SKIN_SPECULAR)

        # fingers: one skinned tube each --------------------------------
        th = 2.0 * math.pi * np.arange(_FING_M) / _FING_M
        for f in FINGERS:
            spec = _SPECS[f]
            base = spec["base"]
            if f == 1:
                # D1b: thumb MCP forward (per-instance, progressive with size)
                base = (base[0], self._thumb_by, base[2])
            lens = [l * self.hf for l in spec["lens"]]
            radii = [r * self.hf for r in spec["radii"]]
            seg, t, s, r = _finger_stations(lens, radii)
            K = len(seg)
            fpts = np.zeros((K * _FING_M + 2, 3))
            fnrm = np.zeros((K * _FING_M + 2, 3))
            mesh, fpts, fnrm, mods = _skin_mesh(
                fpts, fnrm, _tube_faces(K, _FING_M), self.skin)
            self._place(mesh)
            self._geom[f] = dict(
                mesh=mesh, mods=mods, pts=fpts, nrm=fnrm, K=K,
                seg=seg, t=t, r=r, drds=np.gradient(r, s),
                cos=np.cos(th), sin=np.sin(th),
                lens=lens, radii=radii,
                base_local=self._L(base),
                pole_local=np.array([spec["pole"][0] * self.s,
                                     spec["pole"][1], spec["pole"][2]]))
            _material(mesh, _SKIN_AMBIENT, _SKIN_DIFFUSE, _SKIN_SPECULAR)
            # D6: per-point joint shading (PIP/DIP rings darker and pinker).
            n_pts = K * _FING_M + 2
            t0 = max(0.35, 1.0 - radii[3] / lens[2])
            w = _JOINT_WIDTH
            joint = np.zeros(K)
            joint += np.exp(-((t - 1.0) / w) ** 2) * (seg == 0)
            joint += np.exp(-(t / w) ** 2) * (seg == 1)
            joint += np.exp(-((t - 1.0) / w) ** 2) * (seg == 1)
            joint += (np.exp(-((t - t0) / 0.15) ** 2) * (seg == 2)
                      * (t < t0 + 0.3))
            joint = np.clip(joint, 0.0, 1.0)
            js = np.array(_JOINT_SHADE)
            ring_shade = 1.0 - joint[:, None] * (1.0 - js[None, :])
            shade = np.ones((n_pts, 3))
            shade[:K * _FING_M] = np.repeat(ring_shade, _FING_M, axis=0)
            g = self._geom[f]
            g["shade"] = shade
            g["rgb"] = np.zeros((n_pts, 3), np.uint8)
            arr = numpy_to_vtk(g["rgb"], deep=0)
            arr.SetName("RGB")
            mesh.dataset.GetPointData().SetScalars(arr)
            mesh.mapper.SetColorModeToDirectScalars()
            mesh.color(None)          # use the point colours, not the uniform
            g["rgb_arr"] = arr
            self._paint(f, self.skin)
            r3 = radii[3]
            nail = self._place(
                Ellipsoid(pos=(0, 0, 0), axis1=(r3 * 0.74, 0, 0),
                          axis2=(0, r3 * 1.00, 0), axis3=(0, 0, r3 * 0.22),
                          res=14, c=NAIL))
            self.nail[f] = nail
            _material(nail, _SKIN_AMBIENT, _SKIN_DIFFUSE, _NAIL_SPECULAR)
            # D6: flat fake contact shadow riding under the fingertip.
            sh = self._place(
                Ellipsoid(pos=(0, 0, 0),
                          axis1=(r3 * _SHADOW_SX, 0, 0),
                          axis2=(0, r3 * _SHADOW_SY, 0),
                          axis3=(0, 0, 0.06), res=12, c=_SHADOW_COL))
            sh.alpha(_SHADOW_OPACITY)
            try:
                sh.actor.GetProperty().LightingOff()
            except Exception:
                pass
            self.shadow[f] = sh

        # D4: static thenar-web ellipsoid filling the thumb/index wedge.
        # It rides the palm frame (placed in update() with the arm's matrix).
        wr = [_WEB_RADII[i] * self.hf for i in range(3)]
        self.web = self._place(
            Ellipsoid(pos=(0, 0, 0), axis1=(wr[0], 0, 0),
                      axis2=(0, wr[1], 0), axis3=(0, 0, wr[2]),
                      res=14, c=self.skin))
        _material(self.web, _SKIN_AMBIENT, _SKIN_DIFFUSE, _SKIN_SPECULAR)

        # start with a neutral, slightly curled pose ----------------------
        self.reset()

    def _neutral_x(self):
        """Keyboard x of every finger when the hand just hovers at rest."""
        return {f: float(self.root[0] + self._L(_SPECS[f]["base"])[0])
                for f in FINGERS}

    # ---------------------------------------------------------------- update
    @staticmethod
    def _set(mesh, R, t):
        """Copy a 3x3 rotation + translation into the mesh's rig matrix.

        `R` is either a numpy 3x3 or a nested tuple (see `_frame_from_dir`);
        both index the same way."""
        m = mesh._rig_matrix
        r0, r1, r2 = R[0], R[1], R[2]
        m.SetElement(0, 0, r0[0])
        m.SetElement(0, 1, r0[1])
        m.SetElement(0, 2, r0[2])
        m.SetElement(0, 3, t[0])
        m.SetElement(1, 0, r1[0])
        m.SetElement(1, 1, r1[1])
        m.SetElement(1, 2, r1[2])
        m.SetElement(1, 3, t[1])
        m.SetElement(2, 0, r2[0])
        m.SetElement(2, 1, r2[1])
        m.SetElement(2, 2, r2[2])
        m.SetElement(2, 3, t[2])
        m.SetElement(3, 3, 1.0)
        m.Modified()
        mesh.actor.Modified()

    @staticmethod
    def _skin_finger(g, q):
        """Sweep the finger tube along the solved bone chain, in place."""
        seg, t = g["seg"], g["t"]
        C = q[seg] + t[:, None] * (q[seg + 1] - q[seg])
        T = np.empty_like(C)
        T[1:-1] = C[2:] - C[:-2]
        T[0] = C[1] - C[0]
        T[-1] = C[-1] - C[-2]
        n = np.linalg.norm(T, axis=1, keepdims=True)
        T /= np.where(n < 1e-9, 1.0, n)

        X, Z = _frames(T)
        # D4: elliptical section (flatter back, fuller pad) instead of round
        belly = np.where(g["sin"] < 0.0, _FING_BELLY, 1.0)
        zy = g["sin"] * _FING_FLAT * belly
        radial = (g["cos"][None, :, None] * X[:, None, :]
                  + zy[None, :, None] * Z[:, None, :])
        V = C[:, None, :] + g["r"][:, None, None] * radial
        # the surface normal of a tapering tube leans away from the tip;
        # the ellipse contributes its (cos/a, sin/b) normal (plan: a=1, b=FLAT)
        N = (g["cos"][None, :, None] * X[:, None, :]
             + (g["sin"] / _FING_FLAT)[None, :, None] * Z[:, None, :]
             - g["drds"][:, None, None] * T[:, None, :])
        N /= np.linalg.norm(N, axis=2, keepdims=True)

        K, M = g["K"], _FING_M
        pts, nrm = g["pts"], g["nrm"]
        pts[:K * M] = V.reshape(-1, 3)
        nrm[:K * M] = N.reshape(-1, 3)
        pts[K * M], nrm[K * M] = q[0], -T[0]           # MCP end cap
        pts[K * M + 1], nrm[K * M + 1] = q[3], T[-1]   # fingertip pole
        for m in g["mods"]:
            m.Modified()
        g["mesh"].actor.Modified()

    def _paint(self, f, col):
        """D6: repaint a finger's vertex colours (uniform `col` x joint shade)."""
        g = self._geom[f]
        g["rgb"][:] = (np.clip(np.asarray(col, float)[None, :]
                               * g["shade"], 0.0, 1.0) * 255.0 + 0.5
                       ).astype(np.uint8)
        g["rgb_arr"].Modified()
        g["mesh"].actor.Modified()

    def _tint(self, f, a):
        prev = self._col.get(f)
        if prev is not None and abs(prev - a) < 0.02:
            return
        self._col[f] = a
        c = tuple(SKIN[i] + (SKIN_PRESS[i] - SKIN[i]) * a for i in range(3))
        self._paint(f, c)
        cn = tuple(NAIL[i] + (NAIL_PRESS[i] - NAIL[i]) * a for i in range(3))
        self.nail[f].color(cn)

    def update(self, dt, finger_x, press=None, advance_black=None,
                 aim_finger=None):
        """
        Re-pose the whole hand.

        Parameters
        ----------
        dt : float
            seconds since the previous frame
        finger_x : dict
            keyboard x position [cm] each fingertip should be above, keys
            1..5 (missing fingers keep their previous position)
        press : dict
            0..1 press amount per finger (1 = key fully depressed)
        advance_black : None | bool
            True when a note falling on a black key is about to be played:
            the hand then slides into the keyboard early enough to reach it.
            None means "decide from the pressed fingers only".
        aim_finger : None | int
            finger about to strike (its target note is within AIM_LEAD but
            not yet pressed): it lowers early to `top + HOVER` while the
            other idle fingers keep their per-finger REST_LIFT curl (D5).
            None (e.g. in tests) means no finger is aiming.
        """
        dt = float(min(max(dt, 1e-4), 0.1))
        press = press or {}
        self._t += dt

        # ---- fingertip targets ----------------------------------------
        targets = {}
        black_pressed = False
        for f in FINGERS:
            x = finger_x.get(f) if finger_x else None
            if x is None:
                x = self._targets_x.get(f)
            if x is None:
                continue
            black = key_is_black(x)
            name = FINGER_NAMES[f]
            y = CONTACT_Y_BLACK[name] if black else CONTACT_Y_WHITE[name]
            if f == 5:
                # D1c: pinky contact sits closer to the player (short finger)
                y -= _PINKY_CONTACT_DY
            top = BLACK_TOP if black else WHITE_TOP
            a = float(min(max(press.get(f, 0.0), 0.0), 1.0))
            pressed_z = top - key_drop(x, y, a) + self.KEY_SINK
            rest = 1.0 - a
            rs = self.hf / _REST_REF_HF
            if f == aim_finger and rest > 0.0:
                # D5: the finger about to strike lowers early.
                z = (top + self.HOVER) * rest + pressed_z * a
                yy = y
            elif rest > 0.0:
                # D5: idle fingers curl at their per-finger rest height and
                # breathe; everything scales with `rest` so a fully pressed
                # finger (a == 1) still lands exactly on its key (A2).
                breathe = (_BREATHE_AMP * rs
                           * math.sin(2.0 * math.pi * _BREATHE_FREQ * self._t
                                      + _BREATHE_PHASE[f]) * rest)
                z = ((top + (_REST_LIFT[f] + _REST_PULL_Z) * rs) * rest
                     + pressed_z * a + breathe)
                yy = y - _REST_PULL_Y[f] * rs * rest
            else:
                z = pressed_z
                yy = y
            targets[f] = (float(x), yy, z, a)
            black_pressed = black_pressed or (black and a > 0.004)
        self._targets_x = {f: t[0] for f, t in targets.items()}

        # ---- root: spring toward the mean of the targets ---------------
        xs = [t[0] for t in targets.values()]
        rx_t = float(np.mean(xs)) if xs else float(self.root[0])
        any_black = black_pressed if advance_black is None else (
            black_pressed or bool(advance_black))
        ry_t = self.ROOT_Y + (self.BLACK_SHIFT if any_black else 0.0)

        # C4: a finger holding a key (press > 0.5) whose tip is still off
        # (tip_err > 0.1 cm from the previous frame) makes the root chase
        # with a faster constant for this frame; normal TAU otherwise.
        chase = any(float(press.get(f, 0.0)) > 0.5
                    and float(self.tip_err.get(f, 0.0)) > _CHASE_ERR
                    for f in FINGERS)
        alpha = 1.0 - math.exp(-dt / (_CHASE_TAU if chase else self.TAU))
        prev_x = float(self.root[0])
        self.root[0] += (rx_t - self.root[0]) * alpha
        self.root[1] += (ry_t - self.root[1]) * alpha
        vx = (float(self.root[0]) - prev_x) / dt
        self._vx = 0.7 * self._vx + 0.3 * vx
        self.root[2] = self.ROOT_Z + min(0.35, 0.010 * abs(self._vx))

        yaw_vel = max(-self.YAW_MAX,
                        min(self.YAW_MAX, -self.YAW_GAIN * self._vx))
        # D3: the forearm yaws around the wrist toward the elbow side, more
        # so near the ends of the keyboard; the velocity yaw adds on top.
        shoulder_x = self.kb_center + self.s * _SHOULDER_OFF
        yaw_arm = math.atan2(float(self.root[0]) - shoulder_x, _YAW_ARM_LEN)
        yaw_arm = max(-_YAW_ARM_MAX, min(_YAW_ARM_MAX, yaw_arm))
        yaw_t = yaw_arm + yaw_vel
        self.yaw += (yaw_t - self.yaw) * (1.0 - math.exp(-dt / 0.12))

        # ---- hand orientation -----------------------------------------
        R = _rot_z(self.yaw) @ _rot_y(-self.s * self.ROLL) @ _rot_x(self.PITCH)
        self._set(self.arm, R, self.root)
        self._set(self.web, R, self.root + R @ self._web_local)

        # ---- fingers ---------------------------------------------------
        q = np.empty((4, 3))
        for f in FINGERS:
            t = targets.get(f)
            if t is None:
                continue
            x, y, z, a = t
            g = self._geom[f]
            base = self.root + R @ g["base_local"]
            pole = R @ g["pole_local"]
            pts, b1 = solve_chain(base, (x, y, z), g["lens"],
                                  pole, self.BEND_MAX,
                                  self.COUPLING)
            for i, p in enumerate(pts):
                q[i] = p
            self.tip_err[f] = float(np.linalg.norm(
                q[3] - np.array([x, y, z])))
            self.max_bend[f] = float(b1)
            self._skin_finger(g, q)

            d23 = q[3] - q[2]
            n3 = float(np.linalg.norm(d23))
            dir3 = d23 / n3 if n3 > 1e-9 else np.array([0.0, 1.0, 0.0])
            f2 = _frame_from_dir(dir3)
            nz = 0.80 * g["radii"][3]
            sl = _NAIL_SL * g["lens"][2]
            self._set(self.nail[f], f2,
                      (q[2][0] + dir3[0] * sl + f2[0][2] * nz,
                       q[2][1] + dir3[1] * sl + f2[1][2] * nz,
                       q[2][2] + dir3[2] * sl + f2[2][2] * nz))
            self._tint(f, a)
            # D6: fake contact shadow under the fingertip, fading with height.
            tip = q[3]
            ktop = key_top(x)
            h = max(0.0, float(tip[2]) - ktop)
            fade = max(_SHADOW_FLOOR, 1.0 - h / _SHADOW_FADE_H)
            sc = 1.0 + h / 6.0
            self._set(self.shadow[f], np.eye(3) * sc,
                      (float(tip[0]) + _SHADOW_OFF[0],
                       float(tip[1]) + _SHADOW_OFF[1], ktop + 0.03))
            self.shadow[f].alpha(_SHADOW_OPACITY * fade)

    # --------------------------------------------------------------- helpers
    def settle(self, finger_x, press=None, steps=8, advance_black=None):
        """Snap the hand onto its targets (used at startup/after a jump)."""
        for _ in range(max(1, int(steps))):
            self.update(0.1, finger_x, press, advance_black)
        self._vx = 0.0
        self.yaw = 0.0
        self.update(1e-4, finger_x, press, advance_black)
        return self

    def reset(self):
        self.root = np.array([0.0, self.ROOT_Y, self.ROOT_Z])
        self.yaw = 0.0
        self._vx = 0.0
        self._targets_x = {}
        for f in FINGERS:
            self._col.pop(f, None)
        self.update(1.0, self._neutral_x(), {})
        return self

    def actors(self):
        """All the meshes composing the hand (add them to a Plotter)."""
        return list(self._actors)

    def add_to(self, plotter):
        for m in self._actors:
            plotter.add(m)
        return plotter
