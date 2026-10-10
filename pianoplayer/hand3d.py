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
    WHITE_TOP,
    handSizeFactor,
    key_drop,
    key_is_black,
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
    (-46.0, 0.00, 1.92, 1.92, 2.00),
    (-26.0, 0.00, 1.96, 1.88, 2.00),
    (-14.0, 0.00, 2.02, 1.78, 2.05),
    (-8.0, 0.00, 2.12, 1.58, 2.15),
    (-4.5, 0.00, 2.26, 1.38, 2.35),
    (-1.8, 0.00, 2.48, 1.26, 2.55),      # wrist
    (0.8, -0.35, 3.20, 1.36, 2.80),
    (3.2, -0.70, 4.10, 1.48, 3.00),
    (5.8, -0.55, 4.80, 1.46, 3.10),
    (8.2, -0.15, 5.00, 1.32, 3.05),      # knuckle line
    (9.6, 0.00, 4.35, 0.90, 2.80),
    (10.8, 0.00, 0.60, 0.55, 2.20),      # closed cap
)
_ARM_PIN = 2         # first stations keep their y (they are the out-of-frame arm)
_THENAR = 0.34       # thumb-side bulge, over the palm only
_HYPO = 0.12         # pinky-side bulge
_ARM_M = 26          # points around an arm section

_FING_M = 14         # points around a finger section
# rounded fingertip: stations at these fractions of the cap
_TIP_U = (0.0, 0.30, 0.55, 0.75, 0.89, 0.965)

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

    Returns ``[p0, p1, p2, p3]``: p0 is `base`, p3 is `target` when the
    target is inside reach (otherwise the closest reachable point).  The
    chain bends toward `pole`, PIP bending by ``b1`` and DIP by ``coupling*b1``.
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
    return pts


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
        r *= 1.0 + 0.07 * math.exp(-(t / 0.11) ** 2)            # web thickening
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
    BLACK_SHIFT = 1.5         # how far the hand moves into the keyboard
    HOVER = 0.55              # fingertip/key gap while a finger waits
    KEY_SINK = 0.10           # fingertip slightly above the pressed key
    TAU = 0.065               # root spring time constant [s]
    YAW_GAIN = 0.006          # rad of yaw per cm/s of hand speed
    YAW_MAX = 0.16
    PITCH = math.radians(-5.0)
    ROLL = math.radians(5.0)  # sign follows the hand side
    BEND_MAX = 1.75
    COUPLING = 0.4

    def __init__(self, side="right", size="M", skin=SKIN):
        self.side = "left" if str(side).lower().startswith("l") else "right"
        self.s = 1.0 if self.side == "right" else -1.0
        self.hf = float(handSizeFactor(size))
        self.skin = tuple(skin)

        self.root = np.array([0.0, self.ROOT_Y, self.ROOT_Z])
        self.yaw = 0.0
        self._vx = 0.0
        self._targets_x = {}
        self._col = {}
        self._geom = {}
        self.nail = {}
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

        # fingers: one skinned tube each --------------------------------
        th = 2.0 * math.pi * np.arange(_FING_M) / _FING_M
        for f in FINGERS:
            spec = _SPECS[f]
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
                base_local=self._L(spec["base"]),
                pole_local=np.array([spec["pole"][0] * self.s,
                                     spec["pole"][1], spec["pole"][2]]))
            r3 = radii[3]
            self.nail[f] = self._place(
                Ellipsoid(pos=(0, 0, 0), axis1=(r3 * 0.74, 0, 0),
                          axis2=(0, r3 * 1.00, 0), axis3=(0, 0, r3 * 0.22),
                          res=14, c=NAIL))

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
        radial = (g["cos"][None, :, None] * X[:, None, :]
                  + g["sin"][None, :, None] * Z[:, None, :])
        V = C[:, None, :] + g["r"][:, None, None] * radial
        # the surface normal of a tapering tube leans away from the tip
        N = radial - g["drds"][:, None, None] * T[:, None, :]
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

    def _tint(self, f, a):
        prev = self._col.get(f)
        if prev is not None and abs(prev - a) < 0.02:
            return
        self._col[f] = a
        c = tuple(SKIN[i] + (SKIN_PRESS[i] - SKIN[i]) * a for i in range(3))
        cn = tuple(NAIL[i] + (NAIL_PRESS[i] - NAIL[i]) * a for i in range(3))
        self._geom[f]["mesh"].color(c)
        self.nail[f].color(cn)

    def update(self, dt, finger_x, press=None, advance_black=None):
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
        """
        dt = float(min(max(dt, 1e-4), 0.1))
        press = press or {}

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
            top = BLACK_TOP if black else WHITE_TOP
            a = float(min(max(press.get(f, 0.0), 0.0), 1.0))
            hover_z = top + self.HOVER
            pressed_z = top - key_drop(x, y, a) + self.KEY_SINK
            z = hover_z + a * (pressed_z - hover_z)
            targets[f] = (float(x), y, z, a)
            black_pressed = black_pressed or (black and a > 0.004)
        self._targets_x = {f: t[0] for f, t in targets.items()}

        # ---- root: spring toward the mean of the targets ---------------
        xs = [t[0] for t in targets.values()]
        rx_t = float(np.mean(xs)) if xs else float(self.root[0])
        any_black = black_pressed if advance_black is None else (
            black_pressed or bool(advance_black))
        ry_t = self.ROOT_Y + (self.BLACK_SHIFT if any_black else 0.0)

        alpha = 1.0 - math.exp(-dt / self.TAU)
        prev_x = float(self.root[0])
        self.root[0] += (rx_t - self.root[0]) * alpha
        self.root[1] += (ry_t - self.root[1]) * alpha
        vx = (float(self.root[0]) - prev_x) / dt
        self._vx = 0.7 * self._vx + 0.3 * vx
        self.root[2] = self.ROOT_Z + min(0.35, 0.010 * abs(self._vx))

        yaw_t = max(-self.YAW_MAX, min(self.YAW_MAX, -self.YAW_GAIN * self._vx))
        self.yaw += (yaw_t - self.yaw) * (1.0 - math.exp(-dt / 0.12))

        # ---- hand orientation -----------------------------------------
        R = _rot_z(self.yaw) @ _rot_y(-self.s * self.ROLL) @ _rot_x(self.PITCH)
        self._set(self.arm, R, self.root)

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
            for i, p in enumerate(solve_chain(base, (x, y, z), g["lens"],
                                              pole, self.BEND_MAX,
                                              self.COUPLING)):
                q[i] = p
            self._skin_finger(g, q)

            d23 = q[3] - q[2]
            n3 = float(np.linalg.norm(d23))
            dir3 = d23 / n3 if n3 > 1e-9 else np.array([0.0, 1.0, 0.0])
            f2 = _frame_from_dir(dir3)
            nz = 0.80 * g["radii"][3]
            sl = 0.55 * g["lens"][2]
            self._set(self.nail[f], f2,
                      (q[2][0] + dir3[0] * sl + f2[0][2] * nz,
                       q[2][1] + dir3[1] * sl + f2[1][2] * nz,
                       q[2][2] + dir3[2] * sl + f2[2][2] * nz))
            self._tint(f, a)

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
