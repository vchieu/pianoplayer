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
  (no allocation, no filter).  Nails are a vertex-colour patch on the back
  of each fingertip, not a mesh.
* **one blob shadow per hand** -- a black projector-decal actor sharing the
  arm/palm dataset, flattening it onto the key plane (opaque, so the depth
  buffer keeps it behind the hand).

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

try:
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
except Exception:  # pragma: no cover - vedo pulls in vtk anyway
    vtkActor = None
    vtkPolyDataMapper = None

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
)
from pianoplayer import handspec as _HS

# --------------------------------------------------------------------------- #
# looks
# --------------------------------------------------------------------------- #
SKIN = (0.95, 0.80, 0.70)
SKIN_PRESS = (0.96, 0.72, 0.64)
# H3: the nail is a vertex-colour patch, not a mesh: pinkish white close to
# the skin (within ~6%), mixed 0.6 over the base colour.
_NAIL_PATCH = (0.97, 0.82, 0.74)
_NAIL_MIX = 0.6

FINGER_NAMES = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "little"}
FINGERS = (1, 2, 3, 4, 5)

# D1: wrist-root placement scales with hand size so every size XXS..XXL
# reaches the keys with a natural curl (tuned against test_hand_reach.py:
# tip_err < 0.05 cm, max_bend < 1.3 rad for all 70 size/finger/key cells).
# The plan's LEVEL (contact - 8.6*hf - 0.62*finger_len) proved infeasible as
# written: the thumb (MCP 6.2 cm behind the fingers, only 0.6 cm longer) runs
# out of reach on big hands while the middle finger over-folds on black keys,
# and no single root translation satisfies both (see Deviations).  Kept shape:
_ROOT_CONTACT_Y = _HS.ROOT_CONTACT_Y    # mean CONTACT_Y_WHITE
_ROOT_WRIST_GAP = _HS.ROOT_WRIST_GAP    # wrist-to-MCP span carried by palm
_ROOT_REACH_FRAC = _HS.ROOT_REACH_FRAC  # MCP standoff as fraction of
                                        # the middle-finger length
_ROOT_Z_BASE = _HS.ROOT_Z_BASE  # ROOT_Z = BASE * (C0 + K*hf); tuned from
_ROOT_Z_C0 = _HS.ROOT_Z_C0      #   2.3 * (0.6 + 0.4*hf): lower for small hands
_ROOT_Z_HF_K = _HS.ROOT_Z_HF_K  #   (dz dominates), higher for large (bend room)
# D1b: the thumb MCP sits progressively further forward as the hand grows;
# without this the thumb cannot reach either key colour at L/XL/XXL while the
# middle finger must not over-fold (same root serves both).
_THUMB_BASE_Y0 = _HS.THUMB_BASE_Y0
_THUMB_FWD = _HS.THUMB_FWD       # thumb base y += FWD * (hf - HF_XXS)
_HF_XXS = _HS.HF_XXS
# D1c: the pinky is ~2*hf shorter than the middle finger but starts from
# almost the same knuckle, so it runs out of reach on white keys at every
# size.  Its contact point sits this much closer to the player (world cm,
# both colours; still on the key).
_PINKY_CONTACT_DY = _HS.PINKY_CONTACT_DY
# D1c (thumb twin): the anatomical thumb contacts its key nearer the player.
_THUMB_CONTACT_DY = _HS.THUMB_CONTACT_DY
# D3: arm yaw around the wrist so the forearm angles outward toward the
# elbow.  shoulder_x = kb_center +/- 15 (right hand's shoulder sits right of
# centre, left hand's left of it); the 88-key span centre is fixed geometry.
_KB_CENTER = 4.0 * KEYBSIZE + 3.0 * KEY_WB   # (_key_x(21)+_key_x(108))/2
_SHOULDER_OFF = 15.0
_YAW_ARM_LEN = 45.0
_YAW_ARM_MAX = 0.10       # rad, clamp on the arm-yaw part (H1: 0.45 -> 0.10;
                          # velocity yaw adds)
_YAW_ARM_GAIN = 0.4       # H1: the forearm only follows a fraction of the
                          # shoulder angle, so the hand stays near vertical
# H1.2: idle fingers relax toward their natural (parallel) slots instead of
# following the target fan; 0.75 keeps all 70 reach cells green (larger was
# not probed safe). Guarded by held/aim/rest in update().
_IDLE_NATURAL_LERP = 0.75
# H1.3: idle thumb splays outward by this much (x hf); 1.0 breaks two reach
# cells (XXS-f4, XXL-f5), 0.5 keeps 70/70.
_THUMB_SPLAY = 0.5
# H1.4 (D1): idle fingers extend forward by this much (cm x rs, added to the
# idle yy term) instead of lowering _REST_LIFT/_REST_PULL_Y/_ROOT_Z_BASE
# (every literal variant fails 8-24 reach cells). The middle finger is the
# longest, so it needs the largest push. Idle ratio 0.84-0.90, bend
# 0.52-0.88, 70/70 green.
_IDLE_FWD = {1: 1.2, 2: 1.2, 3: 1.7, 4: 1.2, 5: 1.2}

# --------------------------------------------------------------------------- #
# geometry of the hand -- single source of truth is pianoplayer/handspec.py
# (R0.2, imported at top as _HS).  The _NAMES below stay as import aliases
# so diagnostics and the mesh module keep working unchanged.
# --------------------------------------------------------------------------- #
_SPECS = {
    f: dict(base=_HS.mcp_base(f), lens=tuple(_HS.FINGER_SPECS[f]["lens"]),
            radii=tuple(_HS.FINGER_SPECS[f]["radii"]),
            pole=tuple(_HS.FINGER_SPECS[f]["pole"]))
    for f in _HS.FINGERS
}

# arm + palm loft: (y, centre-x, half-width, half-thickness, superellipse p)
# p = 2 is a circle, larger is flatter-sided like the edge of a palm.
# The far end stays long whatever the hand size: on screen the arm has to
# leave the frame, otherwise it looks like a cut-off tube floating in air.
_ARM = tuple(_HS.ARM_PROFILE)
_ARM_PIN = _HS.ARM_PIN     # first stations keep their y (out-of-frame arm)
_THENAR = _HS.THENAR       # thumb-side bulge, over the palm only
_HYPO = _HS.HYPO           # pinky-side bulge
_ARM_M = 26          # points around an arm section

_FING_M = 14         # points around a finger section
# rounded fingertip: stations at these fractions of the cap
_TIP_U = (0.0, 0.30, 0.55, 0.75, 0.89, 0.965)
# D4: finger cross-section is an ellipse, flatter on the back, with a
# fuller belly (pad side) so light models a real fingertip, not a tube.
_FING_FLAT = 0.88    # z-radius scale of the section ellipse
_FING_TIP_FLAT = 0.80  # H2: flatter still over the distal segment (per-station
                       # flat array in _geom; global _FING_FLAT stays 0.88)
_FING_BELLY = 0.85   # extra z scale on the pad side (sin < 0)
# H2: visual finger widths (IK uses lens, so reach is untouched): fingers
# 2..4 x1.10 and the pinky x1.08 so neighbours touch (P5 guardrail: <= x1.15).
_FING_R_SCALE = dict(_HS.FING_R_SCALE)
_FING_TIP_TAPER = 0.92  # H2: distal segment 8% slimmer than before
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
# D6: looks -- skin material, joint shading, blob shadows.
_SKIN_AMBIENT = 0.40
_SKIN_DIFFUSE = 0.70
_SKIN_SPECULAR = 0.08
# H3: whole-hand blob shadows: one black projector-decal actor per arm mesh,
# sharing its dataset (no geometry copy).  The user matrix flattens the arm
# onto the plane z' = _SHADOW_Z with a slight light offset (PROJ @ arm
# matrix, refreshed each frame).  Deliberately OPAQUE: the default alpha
# pass composites translucent-behind-opaque over the hands here (plain-Boxes
# probe: 143 instead of 230), veiling the whole hand grey, and depth peeling
# (which composites correctly) costs +190 ms/frame in software rendering --
# ten times the H4 budget.  Opaque is depth-correct everywhere and cheap;
# black stays invisible on black keys.  Hidden in perspective view.
_SHADOW_OFF = (0.35, -0.35)  # offset from the true position, toward the light
_SHADOW_COL = (0.04, 0.025, 0.025)
_SHADOW_Z = 1.6              # fixed projection plane, just above black keys
# H3: fingertip contact AO baked into the finger vertex colours (grounds the
# fingertip where it meets the key; no actor, no depth involved).
_PRESS_AO = 0.20             # max darkening of the pad-side tip at full press
# R1.6: multiply-only press tint for the skinned mesh (scalars multiply
# the texture, so the base is white and blush only darkens/warms the tip).
_SKIN_PRESS_MUL = (1.00, 0.88, 0.90)
# joint rings shade this much darker/pinker than mid-bone (vertex colours).
_JOINT_SHADE = (0.93, 0.895, 0.905)
_JOINT_WIDTH = 0.18       # gaussian width (local t) of the joint shading
# D4: static thenar-web ellipsoid filling the thumb/index wedge, in palm
# frame (radii x hf, set in _build; centre y follows the thumb MCP in
# __init__ because D1b moves it forward with hand size).  Tuned so it just
# bridges palm edge -> thumb tube without doming out of the silhouette.
_WEB_RADII = tuple(_HS.WEB_RADII)
_WEB_POS_X = _HS.WEB_POS_X    # midpoint of thumb and index MCP x
_WEB_POS_Z = _HS.WEB_POS_Z

_UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# small maths helpers
# --------------------------------------------------------------------------- #
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


def _smoothstep(e0, e1, x):
    """Hermite 0..1 ramp between the edges `e0` and `e1` (H3 press blush)."""
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
        # H2: slight bumps just inside both joints of the middle phalanx
        r *= 1.0 + 0.04 * math.exp(-((t - 0.20) / 0.12) ** 2) \
                 + 0.04 * math.exp(-((t - 0.80) / 0.12) ** 2)
        return r
    if t <= t0:
        return _FING_TIP_TAPER * (r2 + (r3 - r2) * (t / t0))
    u = min(1.0, (t - t0) / (1.0 - t0))
    return _FING_TIP_TAPER * r3 * math.sqrt(max(0.0, 1.0 - u * u))


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
    """Builds a procedural hand and re-poses it every frame.

    `mesh="tubes"` (default until Tier-B acceptance) poses five tapered
    tubes + the lofted arm/palm; `mesh="skinned"` replaces them with one
    smooth-unioned mesh deformed by linear-blend skinning from the same IK
    chains (tubes stay as the automatic fallback). `texture_path` overrides
    the procedural skin PNG with a user-supplied photo (same planar UV).
    """

    ROOT_Y = -13.5            # wrist sits behind the front edge of the keys
    ROOT_Z = 2.3
    BLACK_SHIFT = 1.15         # how far the hand moves into the keyboard
                               # (D1: tuned 1.5 -> 1.15; less over-fold on black)
    HOVER = 0.55              # fingertip/key gap while a finger waits
    KEY_SINK = 0.10           # fingertip slightly above the pressed key
    TAU = 0.065               # root spring time constant [s]
    YAW_GAIN = 0.006          # rad of yaw per cm/s of hand speed
    YAW_MAX = 0.05
    PITCH = math.radians(-5.0)
    ROLL = math.radians(5.0)  # sign follows the hand side
    BEND_MAX = 1.75
    COUPLING = 0.65   # R5.4: DIP = 0.65 x PIP like a real finger (0.4 -> 0.65;
                      # reach remeasured: 70/70 green, worst bend 1.20 -> 1.06)
    SMOOTH_TAU = 0.05    # R1.8: idle-finger x glide time constant [s]

    def __init__(self, side="right", size="M", skin=SKIN, kb_center=None,
                 mesh="skinned", texture_path=None):
        self.side = "left" if str(side).lower().startswith("l") else "right"
        self.s = 1.0 if self.side == "right" else -1.0
        self.hf = float(handSizeFactor(size))
        self.skin = tuple(skin)
        self.mesh_mode = (str(mesh).lower()
                          if str(mesh).lower() in ("tubes", "skinned")
                          else "tubes")
        self.texture_path = texture_path
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
        self._sx = {}           # R1.8: smoothed fan x per finger (idle glide)
        self._blob_world = False  # R1.7: blob projects skinned (world) verts
        self.tip_err = {}     # finger -> cm between true q[3] and its target
        self.max_bend = {}    # finger -> PIP bend angle b1 [rad] of last solve
        self._col = {}
        self._geom = {}
        self._actors = []
        self._blob_actors = []    # H3: raw vtkActors sharing arm/finger data
        self._arm_blob = None     # ... (arm blob actor, matrix) or None
        self._skin = None         # Tier-B skinned mesh state (or None)
        self._skin_q = {}         # finger -> current q (4,3) world (skinned)
        self._skin_ms = 0.0       # last _update_skin cost [ms] (profiling)

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
        # H3: whole-hand blob shadow sharing the arm dataset.
        self._add_blob(self.arm.dataset)

        # fingers: one skinned tube each --------------------------------
        th = 2.0 * math.pi * np.arange(_FING_M) / _FING_M
        for f in FINGERS:
            spec = _SPECS[f]
            base = spec["base"]
            if f == 1:
                # D1b: thumb MCP forward (per-instance, progressive with size)
                base = (base[0], self._thumb_by, base[2])
            lens = [l * self.hf for l in spec["lens"]]
            # H2: visual width (IK only uses lens); thumb keeps its size.
            radii = [r * self.hf * _FING_R_SCALE[f] for r in spec["radii"]]
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
                flat=np.where(seg == 2, _FING_TIP_FLAT, _FING_FLAT),
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
            # H4: side-edge darkening (finger volume), precomputed at build.
            edge = np.ones(n_pts)
            edge[:K * _FING_M] = np.tile(1.0 - 0.10 * np.abs(np.cos(th)), K)
            g["edge"] = edge
            g["rgb"] = np.zeros((n_pts, 3), np.uint8)
            arr = numpy_to_vtk(g["rgb"], deep=0)
            arr.SetName("RGB")
            mesh.dataset.GetPointData().SetScalars(arr)
            mesh.mapper.SetColorModeToDirectScalars()
            mesh.color(None)          # use the point colours, not the uniform
            g["rgb_arr"] = arr
            # H3: press-blend coordinate (MCP = 0 .. tip = 1) and the
            # vertex-colour nail patch (back of the distal segment).
            smax = float(s.max()) if float(s.max()) > 0.0 else 1.0
            snorm = np.ones(n_pts)
            snorm[:K * _FING_M] = np.repeat(s / smax, _FING_M)
            snorm[K * _FING_M] = 0.0       # MCP end cap
            snorm[K * _FING_M + 1] = 1.0   # fingertip pole
            g["snorm"] = snorm
            g["t0"] = t0
            sin_rep = np.tile(np.sin(th), K)
            seg_rep = np.repeat(seg, _FING_M)
            t_rep = np.repeat(t, _FING_M)
            nailm = np.zeros(n_pts, bool)
            nailm[:K * _FING_M] = ((sin_rep > 0.0) & (seg_rep == 2)
                                   & (t_rep > t0 - 0.1))
            g["nailm"] = nailm
            padm = np.zeros(n_pts, bool)
            padm[:K * _FING_M] = np.tile(np.sin(th) < 0.0, K)
            g["padm"] = padm
            self._paint(f, 0.0)

        # D4: static thenar-web ellipsoid filling the thumb/index wedge.
        # It rides the palm frame (placed in update() with the arm's matrix).
        wr = [_WEB_RADII[i] * self.hf for i in range(3)]
        self.web = self._place(
            Ellipsoid(pos=(0, 0, 0), axis1=(wr[0], 0, 0),
                      axis2=(0, wr[1], 0), axis3=(0, 0, wr[2]),
                      res=14, c=self.skin))
        _material(self.web, _SKIN_AMBIENT, _SKIN_DIFFUSE, _SKIN_SPECULAR)

        # Tier B: one smooth-unioned mesh, skinned from the same IK chains.
        # Tubes stay built (fallback + compat buffers); skinned mode hides
        # their actors instead of deleting them.
        if self.mesh_mode == "skinned":
            try:
                self._build_skin()
            except Exception as exc:
                print("HandRig: skinned mesh unavailable (%s); "
                      "falling back to tubes" % exc)
                self.mesh_mode = "tubes"
                self._skin = None

        # start with a neutral, slightly curled pose ----------------------
        self.reset()

    def _build_skin(self):
        """Load the cached rest mesh + weights, make the one skin mesh."""
        from pianoplayer.handmesh import (
            build_skin_texture,
            get_rest_data,
            rest_uv,
        )
        from pianoplayer.handmesh import texture_path as _default_tex
        data = get_rest_data(self.hf, self.side)
        rest = np.array(data["pts"], float)
        nrm0 = np.array(data["nrm"], float)
        names, bones = data["names"], data["bones"]
        idx = np.array(data["idx"], np.int64)
        w = np.array(data["w"], float)
        P0 = np.array(data["P0"], float)
        R0 = np.array(data["R0"], float)
        mesh, pbuf, nbuf, mods = _skin_mesh(rest, nrm0, data["faces"],
                                            self.skin)
        mesh._rig_matrix = None        # static planar UVs (rest pose; kept through skinning) + texture
        uv, _ = rest_uv(rest)
        try:
            import vtkmodules.util.numpy_support as _ns
            tc = _ns.numpy_to_vtk(np.ascontiguousarray(uv, np.float32),
                                  deep=1)
            tc.SetName("TCoords")
            mesh.dataset.GetPointData().SetTCoords(tc)
        except Exception:
            pass
        tpath = self.texture_path or _default_tex(self.hf, self.side)
        _tex_ok = False
        try:
            import os as _os
            if not _os.path.isfile(tpath):
                build_skin_texture(rest, bones, tpath, names=names,
                                   hf=self.hf, side=self.side)
            from PIL import Image as _Image
            with _Image.open(tpath) as _im:
                _im.verify()  # garbage file -> raise -> keep SKIN below
            mesh.texture(tpath)
        except Exception:
            pass  # no texture: keep the SKIN actor colour from _skin_mesh
        else:
            # R1.5: the texture background already carries the skin tone, so
            # the actor colour must be white -- SKIN x SKIN would square it.
            # White is set ONLY on success (except-branch keeps SKIN).
            _tex_ok = True
            mesh.color((1.0, 1.0, 1.0))
        _material(mesh, _SKIN_AMBIENT, _SKIN_DIFFUSE, _SKIN_SPECULAR)
        B = len(names)
        rigid = np.array([nm in ("forearm", "palm", "f1_cmc") for nm in
                          names])
        fb = {}
        for f in FINGERS:
            fb[f] = [names.index("f%d_%d" % (f, i)) for i in range(3)]
        # Precompute flat list of finger bone indices (15 bones) for vectorized setup
        fb_flat = np.array([fb[f][i] for f in FINGERS for i in range(3)], dtype=np.int64)
        # float32 working copies (N=3002: halves the per-frame LBS flops;
        # pbuf/nbuf casts on assignment are cheap).
        rest32 = np.ascontiguousarray(rest, np.float32)
        nrm32 = np.ascontiguousarray(nrm0, np.float32)
        P032 = np.ascontiguousarray(P0, np.float32)
        R032 = np.ascontiguousarray(R0, np.float32)
        w32 = np.ascontiguousarray(w, np.float32)
        # Precompute rigid vertex mask (weight 1.0 on rigid bone)
        rigid_mask = (w32[:, 0] >= 0.99) & np.isin(idx[:, 0], np.flatnonzero(rigid))
        self._skin = dict(mesh=mesh, mods=mods, pbuf=pbuf, nbuf=nbuf,
                          rest=rest32, nrm0=nrm32, idx=idx, w=w32,
                          P0=P032, R0=R032, P0g=P032[idx],
                          names=names, bones=bones, B=B, rigid=rigid, fb=fb,
                          fb_flat=fb_flat,
                          Rb=np.empty((B, 3, 3), np.float32),
                          Pb=np.empty((B, 3), np.float32),
                          rigid_mask=rigid_mask)
        # R1.6 vertex-blush prep, from the bone weights: dominant finger +
        # distalness (weight on that finger's f?_2 bone).  The RGB scalars
        # start at the texture base (white: scalars multiply the texture)
        # or, with no texture, at SKIN -- so a texture failure still shows
        # a skin-coloured hand, not a white one. A pressed finger blushes
        # only its tip via _tint_skin().
        try:
            N = rest32.shape[0]
            aff = np.zeros((N, 5), np.float32)
            dst = np.zeros((N, 5), np.float32)
            for k, f in enumerate(FINGERS):
                b0, b1, b2 = [names.index("f%d_%d" % (f, i))
                              for i in range(3)]
                for b in (b0, b1, b2):
                    m = idx == b
                    aff[:, k] += (w * m).sum(axis=1)
                m2 = idx == b2
                dst[:, k] = (w * m2).sum(axis=1)
            dom = np.argmax(aff, axis=1).astype(np.int8)
            dist = dst[np.arange(N), dom].astype(np.float32)
            self._skin["dom"] = dom
            self._skin["dist"] = np.clip(dist, 0.0, 1.0)
            if _tex_ok:
                base255 = np.array([255, 255, 255], np.uint8)
            else:
                base255 = (np.clip(np.asarray(self.skin, float), 0.0, 1.0)
                            * 255.0 + 0.5).astype(np.uint8)
            rgb = np.tile(base255[None, :], (N, 1))
            arr = numpy_to_vtk(rgb, deep=0)
            arr.SetName("RGB")
            mesh.dataset.GetPointData().SetScalars(arr)
            mesh.mapper.SetColorModeToDirectScalars()
            mesh.color((1.0, 1.0, 1.0) if _tex_ok else self.skin)
            self._skin["blush"] = rgb
            self._skin["blush_arr"] = arr
            self._skin["blush_col"] = {}
            self._skin["blush_base"] = base255
        except Exception:
            pass
        # seed the skin chains (world) from the rest pose so a partial fan
        # update can never KeyError before reset() solves every finger.
        from pianoplayer.handmesh import rest_bones as _rb
        _bb, _nn = _rb(self.hf, self.side)
        _by = {nm: (np.asarray(a, float), np.asarray(b, float))
               for nm, (a, b) in zip(_nn, _bb)}
        _R0 = (_rot_z(0.0) @ _rot_y(-self.s * self.ROLL)
               @ _rot_x(self.PITCH))
        _r0 = np.array([0.0, self.ROOT_Y, self.ROOT_Z])
        for f in FINGERS:
            try:
                a0, _b0 = _by["f%d_0" % f]
                _a1, b1 = _by["f%d_1" % f]
                _a2, b2 = _by["f%d_2" % f]
                ql = np.stack([a0, _b0, b1, b2])
            except KeyError:
                ql = np.zeros((4, 3))
            self._skin_q[f] = _r0 + ql @ _R0.T
        # hide the tube actors (NOT deleted: fallback + compat buffers stay)
        try:
            self.arm.actor.SetVisibility(0)
            self.web.actor.SetVisibility(0)
            for f in FINGERS:
                self._geom[f]["mesh"].actor.SetVisibility(0)
        except Exception:
            pass
        # R1.7: the blob projects the skinned (world-coordinate) mesh, with
        # finger shapes.  The tube-arm blob is retired (its dataset is hidden).
        try:
            if self._arm_blob is not None:
                _old, _mat = self._arm_blob
                try:
                    _old.SetVisibility(0)
                except Exception:
                    pass
                try:
                    self._blob_actors.remove(_old)
                except ValueError:
                    pass
                self._arm_blob = None
            self._add_blob(mesh.dataset)
            self._blob_world = True
        except Exception:
            pass

    def _update_skin(self, R):
        """Vectorised LBS of the skin mesh from the solved chains (F3)."""
        import time as _tm
        t0 = _tm.perf_counter()
        S = self._skin
        Rb, Pb = S["Rb"], S["Pb"]
        Rb[S["rigid"]] = np.asarray(R, np.float32)
        Pb[S["rigid"]] = np.asarray(self.root, np.float32)
        # Vectorized finger bone setup: stack all _skin_q -> (5,4,3)
        q_all = np.stack([self._skin_q[f] for f in FINGERS])  # (5,4,3)
        d = q_all[:, 1:] - q_all[:, :-1]  # (5,3,3) per-finger bone dirs
        n = np.linalg.norm(d, axis=2, keepdims=True)
        dirs = d / np.maximum(n, 1e-9)    # (5,3,3)
        dirs_flat = dirs.reshape(-1, 3)   # (15,3)
        # Precompute flat list of finger bone indices
        fb_flat = S["fb_flat"]
        R0_fingers = S["R0"][fb_flat]     # (15,3,3)
        X, Z = _frames(dirs_flat)
        Rc = np.stack([X, dirs_flat, Z], axis=2).astype(np.float32)  # (15,3,3)
        # Batched: Rb[b] = Rc @ R0.T, Pb[b] = q[i]
        Rb[fb_flat] = Rc @ np.transpose(R0_fingers, (0, 2, 1))
        Pb[fb_flat] = q_all[:, :-1].reshape(-1, 3)
        idx, w = S["idx"], S["w"]
        rest = S["rest"]
        nrm0 = S["nrm0"]
        P0g = S["P0g"]
        rigid_mask = S["rigid_mask"]
        non_rigid = ~rigid_mask
        # Rigid vertices: single transform (root + R @ v)
        if rigid_mask.any():
            S["pbuf"][rigid_mask] = self.root + (R @ rest[rigid_mask].T).T
            S["nbuf"][rigid_mask] = (R @ nrm0[rigid_mask].T).T
        # Non-rigid vertices: full 4-bone blend using einsum
        if non_rigid.any():
            nr = non_rigid
            idx_nr = idx[nr]           # (M,4)
            w_nr = w[nr]               # (M,4)
            rest_nr = rest[nr]         # (M,3)
            nrm0_nr = nrm0[nr]         # (M,3)
            P0g_nr = P0g[nr]           # (M,4,3)
            Rb_idx = Rb[idx_nr]        # (M,4,3,3)
            Pb_idx = Pb[idx_nr]        # (M,4,3)
            offset = rest_nr[:, None, :] - P0g_nr  # (M,4,3)
            # Batched matrix-vector multiply via einsum
            v = Pb_idx + np.einsum('mikj,mij->mik', Rb_idx, offset)
            S["pbuf"][nr] = (v * w_nr[..., None]).sum(axis=1)
            # Normals: rotate by bone rotations, blend via einsum, final normalize
            n_rot = np.einsum('mikj,mj->mik', Rb_idx, nrm0_nr)  # (M,4,3)
            nbuf = np.einsum('mi,mij->mj', w_nr, n_rot)
            nbuf /= np.maximum(np.linalg.norm(nbuf, axis=1, keepdims=True), 1e-12)
            S["nbuf"][nr] = nbuf
        for m in S["mods"]:
            m.Modified()
        S["mesh"].actor.Modified()
        self._skin_ms = 1000.0 * (_tm.perf_counter() - t0)

    def _neutral_x(self):
        """Keyboard x of every finger when the hand just hovers at rest."""
        return {f: float(self.root[0] + self._L(_SPECS[f]["base"])[0])
                for f in FINGERS}

    # ---------------------------------------------------------------- update
    @staticmethod
    def _set(mesh, R, t):
        """Copy a 3x3 rotation + translation into the mesh's rig matrix.

        `R` is either a numpy 3x3 or a nested tuple; both index the same way.
        """
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
        # D4: elliptical section (flatter back, fuller pad) instead of round;
        # H2: the flat is per-station (flatter over the distal segment).
        belly = np.where(g["sin"] < 0.0, _FING_BELLY, 1.0)
        zy = g["sin"][None, :] * g["flat"][:, None] * belly[None, :]
        radial = (g["cos"][None, :, None] * X[:, None, :]
                  + zy[:, :, None] * Z[:, None, :])
        V = C[:, None, :] + g["r"][:, None, None] * radial
        # the surface normal of a tapering tube leans away from the tip;
        # the ellipse contributes its (cos/a, sin/b) normal (plan: a=1, b=FLAT)
        N = (g["cos"][None, :, None] * X[:, None, :]
             + (g["sin"][None, :] / g["flat"][:, None])[:, :, None]
             * Z[:, None, :]
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

    def _paint(self, f, amount):
        """H3: repaint a finger's vertex colours.

        The press colour only blushes the fingertip
        (`smoothstep(0.6, 1.0, snorm) * amount`); the nail region keeps a
        fixed pinkish-white patch (nails do not blush with the fingertip).
        """
        g = self._geom[f]
        a = float(min(max(amount, 0.0), 1.0))
        w = _smoothstep(0.6, 1.0, g["snorm"]) * a
        skin = np.asarray(SKIN, float)[None, :]
        pr = np.asarray(SKIN_PRESS, float)[None, :]
        base = skin + (pr - skin) * w[:, None]
        nm = g["nailm"]
        if bool(nm.any()):
            sk = np.asarray(SKIN, float)[None, :]
            patch = np.asarray(_NAIL_PATCH, float)[None, :]
            base[nm] = sk * (1.0 - _NAIL_MIX) + patch * _NAIL_MIX
        # H3: contact AO: the pad side near the tip darkens with the press,
        # grounding the fingertip on its key (disjoint from the nail patch).
        ao = (_smoothstep(0.55, 1.0, g["snorm"]) * a
              * g["padm"].astype(float))
        base *= (1.0 - _PRESS_AO * ao)[:, None]
        g["rgb"][:] = (np.clip(base * g["shade"] * g["edge"][:, None],
                                0.0, 1.0) * 255.0 + 0.5
                       ).astype(np.uint8)
        g["rgb_arr"].Modified()
        g["mesh"].actor.Modified()

    def _tint(self, f, a):
        prev = self._col.get(f)
        if prev is not None and abs(prev - a) < 0.02:
            return
        self._col[f] = a
        self._paint(f, a)

    def _tint_skin(self, f, a):
        """R1.6: blush only the pressed fingertip of the skinned mesh."""
        S = self._skin
        if S is None or "blush" not in S:
            return
        prev = S["blush_col"].get(f)
        if prev is not None and abs(prev - a) < 0.02:
            return
        S["blush_col"][f] = a
        try:
            w = _smoothstep(0.55, 1.0, S["dist"]) * float(min(max(a, 0.0),
                                                              1.0))
            m = (S["dom"] == (f - 1)) & (w > 0.0)
            rgb = S["blush"]
            base = np.asarray(S.get("blush_base", (255, 255, 255)),
                              float)[None, :]
            rgb[m] = (np.clip(
                base * (1.0 - w[m, None] * (1.0 - np.asarray(_SKIN_PRESS_MUL,
                                                            float)[None, :])),
                0.0, 255.0) + 0.5).astype(np.uint8)
            # released verts back to the base (white textured, SKIN plain)
            m0 = (S["dom"] == (f - 1)) & (w == 0.0)
            rgb[m0] = np.asarray(S.get("blush_base", (255, 255, 255)),
                                 np.uint8)
            S["blush_arr"].Modified()
            S["mesh"].actor.Modified()
        except Exception:
            pass

    # ------------------------------------------------------------- shadows
    def _add_blob(self, pd):
        """H3: black projector-decal actor sharing the arm dataset `pd`.

        No geometry is copied; the actor flattens the arm/palm onto the key
        plane (PROJ @ arm matrix, refreshed in `_set_arm_shadow`).  Only the
        arm uses it: finger tubes dip below the shadow plane when pressed,
        so their own opaque shadow would cap the fingertip black, and their
        transparent one veils the hand (see _SHADOW_* notes).  Silently
        skipped when raw vtk actors are unavailable.
        """
        if vtkActor is None or vtkPolyDataMapper is None:
            return
        try:
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.SetScalarVisibility(False)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*_SHADOW_COL)
            try:
                actor.GetProperty().LightingOff()
            except Exception:
                pass
            mat = vtkMatrix4x4()
            mat.Identity()
            self._arm_blob = (actor, mat)
            actor.SetUserMatrix(mat)
            self._blob_actors.append(actor)
        except Exception:
            pass

    def _set_arm_shadow(self, R, t):
        """Refresh the arm blob matrix: the PROJ flattening after the arm's
        own rigid motion.  Called with the same R, t as the arm itself."""
        if not self._arm_blob:
            return
        actor, mat = self._arm_blob
        if self._blob_world:
            # R1.7: skinned verts are already in world coordinates, so the
            # projection stands alone (no R premultiply).
            mat.SetElement(0, 0, 1.0)
            mat.SetElement(0, 1, 0.0)
            mat.SetElement(0, 2, 0.0)
            mat.SetElement(0, 3, _SHADOW_OFF[0])
            mat.SetElement(1, 0, 0.0)
            mat.SetElement(1, 1, 1.0)
            mat.SetElement(1, 2, 0.0)
            mat.SetElement(1, 3, _SHADOW_OFF[1])
            mat.SetElement(2, 0, 0.0)
            mat.SetElement(2, 1, 0.0)
            mat.SetElement(2, 2, 0.0)
            mat.SetElement(2, 3, _SHADOW_Z)
            mat.Modified()
            actor.Modified()
            return
        r0, r1 = R[0], R[1]
        mat.SetElement(0, 0, r0[0])
        mat.SetElement(0, 1, r0[1])
        mat.SetElement(0, 2, r0[2])
        mat.SetElement(0, 3, t[0] + _SHADOW_OFF[0])
        mat.SetElement(1, 0, r1[0])
        mat.SetElement(1, 1, r1[1])
        mat.SetElement(1, 2, r1[2])
        mat.SetElement(1, 3, t[1] + _SHADOW_OFF[1])
        mat.SetElement(2, 0, 0.0)
        mat.SetElement(2, 1, 0.0)
        mat.SetElement(2, 2, 0.0)
        mat.SetElement(2, 3, _SHADOW_Z)
        mat.Modified()
        actor.Modified()

    def set_shadows_visible(self, v):
        """Show (orthographic top view) or hide (perspective) the blob
        shadows; called from `vkeyboard.toggle_view`."""
        for a in self._blob_actors:
            try:
                a.SetVisibility(1 if v else 0)
            except Exception:
                pass

    def update(self, dt, finger_x, press=None, advance_black=None,
                 aim_finger=None, held=None, aim_fingers=None):
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
        aim_finger : None | int | set
            finger(s) about to strike (their target notes are within
            AIM_LEAD but not yet pressed): they lower early to `top +
            HOVER` while the other idle fingers keep their per-finger
            REST_LIFT curl (D5).  A set is accepted (see vkeyboard._pump);
            a lone int keeps working.  None (e.g. in tests) means no
            finger is aiming.  Held/aiming fingers track their key x
            exactly (no idle lerp, no smoothing glide); idle fingers
            glide toward the fan with SMOOTH_TAU (R1.8).
        aim_fingers : None | int | set
            R1.8 set form of `aim_finger`: fingers of all notes with
            `t_on - now < AIM_LEAD` that are not yet fired. Unioned with
            `aim_finger`, so old single-finger callers behave exactly as
            before; a set of several aiming fingers keeps each of them
            exactly on its key x (no lerp toward natural).
        held : None | set
            fingers currently holding a key (passed from `_frame` as
            `{f for f in FINGERS if p.active.get(f)}`): they keep exactly
            their key x, never lerped.  None (settle, diag, tests) means no
            finger is known-held, so every idle finger relaxes toward its
            natural slot.  The guard keys off `held`, never off the press
            amount, so a held finger targets its key exactly even in early
            frames when press is still ~ 0.
        """
        dt = float(min(max(dt, 1e-4), 0.1))
        press = press or {}
        self._t += dt
        held = set() if held is None else set(held)
        # R1.8: accept the aim set (union with the legacy single finger).
        aim_set = set()
        for _aim in (aim_finger, aim_fingers):
            if _aim is None:
                continue
            try:
                aim_set |= set(_aim)
            except TypeError:
                aim_set.add(_aim)

        # ---- fingertip targets ----------------------------------------
        targets = {}
        fan_xs = {}       # H1.2: pre-lerp x per finger; the root spring below
                          # chases their mean, so the idle lerp cannot drag
                          # the root (and the pressed fingers) away with it.
        black_pressed = False
        # H1.2: natural (parallel) slots from the root at frame start --
        # the root spring below moves afterwards, so this cannot feed back.
        natural = self._neutral_x()
        for f in FINGERS:
            x = finger_x.get(f) if finger_x else None
            if x is None:
                x = self._targets_x.get(f)
            if x is None:
                continue
            fan_xs[f] = float(x)
            # R1.8: held/aiming fingers track the key x exactly (no glide,
            # no idle lerp), so a press never teleports at the transition;
            # idle fingers glide toward the fan (tau = SMOOTH_TAU).  The
            # tracked x seeds the glide so a release resumes from the key.
            if f in held or f in aim_set:
                x = float(x)
                self._sx[f] = x
            else:
                prev = self._sx.get(f, float(x))
                k = 1.0 - math.exp(-dt / self.SMOOTH_TAU)
                x = prev + (float(x) - prev) * k
                self._sx[f] = x
            a = float(min(max(press.get(f, 0.0), 0.0), 1.0))
            rest = 1.0 - a
            if f not in held and f not in aim_set and rest > 0.5:
                # H1.2/1.3: idle fingers head for their natural slots and
                # the idle thumb splays outward like a real hand.
                x = ((1.0 - _IDLE_NATURAL_LERP) * float(x)
                     + _IDLE_NATURAL_LERP * natural[f])
                if f == 1:
                    x -= self.s * _THUMB_SPLAY * self.hf
            black = key_is_black(x)
            name = FINGER_NAMES[f]
            y = CONTACT_Y_BLACK[name] if black else CONTACT_Y_WHITE[name]
            if f == 1:
                # D1c (thumb twin): short anatomical thumb, nearer contact
                y -= _THUMB_CONTACT_DY
            if f == 5:
                # D1c: pinky contact sits closer to the player (short finger)
                y -= _PINKY_CONTACT_DY
            top = BLACK_TOP if black else WHITE_TOP
            pressed_z = top - key_drop(x, y, a) + self.KEY_SINK
            rs = self.hf / _REST_REF_HF
            if f in aim_set and rest > 0.0:
                # D5: the finger about to strike lowers early.
                z = (top + self.HOVER) * rest + pressed_z * a
                yy = y
            elif rest > 0.0:
                # D5: idle fingers curl at their per-finger rest height and
                # breathe; everything scales with `rest` so a fully pressed
                # finger (a == 1) still lands exactly on its key (A2).
                # H1.4 (D1): idle fingers also extend forward by _IDLE_FWD.
                breathe = (_BREATHE_AMP * rs
                           * math.sin(2.0 * math.pi * _BREATHE_FREQ * self._t
                                      + _BREATHE_PHASE[f]) * rest)
                z = ((top + (_REST_LIFT[f] + _REST_PULL_Z) * rs) * rest
                     + pressed_z * a + breathe)
                yy = (y - _REST_PULL_Y[f] * rs * rest
                      + _IDLE_FWD[f] * rs * rest)
            else:
                z = pressed_z
                yy = y
            targets[f] = (float(x), yy, z, a)
            black_pressed = black_pressed or (black and a > 0.004)
        self._targets_x = {f: t[0] for f, t in targets.items()}

        # ---- root: spring toward the mean of the (pre-lerp) targets --
        xs = [fan_xs[f] for f in targets] if fan_xs else []
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
        # R5.1b: a stretched hold drags the root target toward its own key:
        # the chord shortens instead of the fingertip detaching while the
        # hand serves new notes.  The pull fades in linearly above a 0.3 cm
        # deadband (an on/off switch limit-cycles against settle()'s 8
        # steps); the 0.4 ceiling keeps the fan mean dominant.
        pull = 0.0
        sxx = syy = 0.0
        for f in FINGERS:
            if f not in targets:
                continue
            if float(press.get(f, 0.0)) > 0.5:
                err = float(self.tip_err.get(f, 0.0))
                if err > 0.3:
                    w = min((err - 0.3) / 1.0, 1.0)
                    pull = max(pull, 0.4 * w)
                    sxx += targets[f][0]
                    syy += targets[f][1]
        if pull > 0.0:
            n = sum(1 for f in FINGERS
                    if f in targets and float(press.get(f, 0.0)) > 0.5
                    and float(self.tip_err.get(f, 0.0)) > 0.3)
            rx_t = (1.0 - pull) * rx_t + pull * sxx / max(n, 1)
            ry_t = (1.0 - pull) * ry_t + pull * syy / max(n, 1)
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
        yaw_arm = (math.atan2(float(self.root[0]) - shoulder_x,
                              _YAW_ARM_LEN) * _YAW_ARM_GAIN)
        yaw_arm = max(-_YAW_ARM_MAX, min(_YAW_ARM_MAX, yaw_arm))
        yaw_t = yaw_arm + yaw_vel
        self.yaw += (yaw_t - self.yaw) * (1.0 - math.exp(-dt / 0.12))

        # ---- hand orientation -----------------------------------------
        R = _rot_z(self.yaw) @ _rot_y(-self.s * self.ROLL) @ _rot_x(self.PITCH)
        self._set(self.arm, R, self.root)
        self._set(self.web, R, self.root + R @ self._web_local)
        self._set_arm_shadow(R, self.root)

        # ---- fingers ---------------------------------------------------
        q = np.empty((4, 3))
        skinned = self._skin is not None
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
            if skinned:
                # IK hooks stay exact (reach/regression/diag read these);
                # the hidden tubes keep only compat tip/cap points.
                self._skin_q[f] = q.copy()
                g["pts"][-1] = q[3]
                g["pts"][-2] = q[0]
                # R1.6: fingertip-only press blush on the skinned mesh.
                self._tint_skin(f, a)
                continue
            self._skin_finger(g, q)
            # H3: fingertip-only press blush (no nail mesh, no shadow disc).
            self._tint(f, a)
        if skinned:
            self._update_skin(R)

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
        self._sx = {}
        for f in FINGERS:
            self._col.pop(f, None)
        self.update(1.0, self._neutral_x(), {})
        return self

    def actors(self):
        """All the meshes composing the hand (add them to a Plotter).

        Skinned mode exposes only the one skin mesh: the hidden tube actors
        stay out so mask/diag logic cannot force them visible again.
        """
        if self._skin is not None:
            return [self._skin["mesh"]]
        return list(self._actors)

    def add_to(self, plotter):
        for m in self.actors():
            plotter.add(m)
        for a in self._blob_actors:
            try:
                plotter.add(a)
            except Exception:
                pass
        return plotter
