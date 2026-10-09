"""
Procedural anatomical hand for the 3D piano player.

The hand is assembled from primitives: ellipsoids for the palm masses
(main palm, thenar/hypothenar eminences), cylinders for the forearm, the
wrist and the three phalanges of every finger, spheres for the joints and
an ellipsoid cap + nail for each fingertip.

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
from vedo import Cylinder, Ellipsoid, Sphere
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
NAIL = (0.96, 0.85, 0.83)
NAIL_PRESS = (0.93, 0.62, 0.60)

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

# palm masses: (centre, (semi-axis x, semi-axis y, semi-axis z))
_PALM = (
    ((0.00, 4.2, 0.75), ((4.30, 0, 0), (0, 4.60, 0), (0, 0, 1.20))),
    ((-3.3, 2.6, 0.55), ((2.20, 0, 0), (0, 3.40, 0), (0, 0, 1.10))),
    ((3.50, 3.0, 0.55), ((1.70, 0, 0), (0, 3.40, 0), (0, 0, 1.00))),
)
_WRIST = ((0.0, -2.2, 0.55), (0.0, 3.2, 0.75), 1.12)
# the far end is deliberately long: on screen the arm must leave the frame,
# otherwise it looks like a cut-off tube floating in mid air
_FOREARM = ((0.0, -46.0, 0.55), (0.0, -2.0, 0.55), 1.95)


# --------------------------------------------------------------------------- #
# small maths helpers
# --------------------------------------------------------------------------- #
def _frame_from_dir(d, up=(0.0, 0.0, 1.0)):
    """Orthonormal frame whose +y axis is `d` and whose +z axis is as close
    as possible to `up` (used to orient phalanges, pads and nails)."""
    y = np.asarray(d, dtype=float)
    n = float(np.linalg.norm(y))
    if n < 1e-9:
        y = np.array([0.0, 1.0, 0.0])
    else:
        y = y / n
    x = np.cross(y, np.asarray(up, dtype=float))
    nx = float(np.linalg.norm(x))
    if nx < 1e-6:
        for aux in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            x = np.cross(y, np.asarray(aux, dtype=float))
            nx = float(np.linalg.norm(x))
            if nx > 1e-6:
                break
    x = x / nx
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


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
    v = np.asarray(target, dtype=float) - np.asarray(base, dtype=float)
    d = float(np.linalg.norm(v))
    if d < 1e-9:
        v = np.asarray(pole, dtype=float)
        d = float(np.linalg.norm(v))
        if d < 1e-9:
            v = np.array([0.0, 1.0, 0.0])
            d = 1.0
    u = v / d

    # pole, orthogonalised against the chord -> in-plane "up" direction
    p = np.asarray(pole, dtype=float)
    p = p - float(np.dot(p, u)) * u
    n = float(np.linalg.norm(p))
    if n < 1e-6:
        aux = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
        p = np.cross(u, np.asarray(aux, dtype=float))
        n = float(np.linalg.norm(p))
        if n < 1e-6:
            p = np.array([0.0, 0.0, 1.0])
            n = 1.0
    w = p / n

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
        pts.append(x * u + y * w)
    b = np.asarray(base, dtype=float)
    return [b + q for q in pts]


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
        self._press = {}
        self._col = {}
        self._geom = {}
        self._root_parts = []      # (mesh, local centre, local frame)
        self.seg = {}
        self.jnt = {}
        self.pad = {}
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
        skin = self.skin

        # palm masses -----------------------------------------------------
        # NB: axes must stay positive even when mirroring the hand, otherwise
        # the ellipsoid comes out inside-out (winding reversed -> black).
        for centre, axes in _PALM:
            a1, a2, a3 = (np.abs(self._L(ax)) for ax in axes)
            mesh = Ellipsoid(pos=(0, 0, 0), axis1=a1, axis2=a2, axis3=a3,
                             res=18, c=skin)
            self._root_parts.append((self._place(mesh), self._L(centre), np.eye(3)))

        # wrist -----------------------------------------------------------
        p0, p1, r = _WRIST
        a, b = self._L(p0), self._L(p1)
        r_wrist = r * self.hf
        d = b - a
        mesh = Cylinder(pos=[(0, 0, 0), (0, float(np.linalg.norm(d)), 0)],
                        r=r_wrist, res=18, c=skin)
        self._root_parts.append((self._place(mesh), a, _frame_from_dir(d)))
        wrist_far = a

        # forearm (far end keeps its length for every hand size) ----------
        p0, p1, r = _FOREARM
        a = np.array([p0[0], p0[1], p0[2]])
        b = self._L(p1)
        r_fore = r * self.hf
        d = b - a
        mesh = Cylinder(pos=[(0, 0, 0), (0, float(np.linalg.norm(d)), 0)],
                        r=r_fore, res=18, c=skin)
        self._root_parts.append((self._place(mesh), a, _frame_from_dir(d)))

        # rounded joint where the two tubes meet: it hides the end caps,
        # which would otherwise show up as dark openings at the wrist
        mesh = Sphere(pos=(0, 0, 0), r=max(r_wrist, r_fore) * 1.04, res=16,
                      c=skin)
        self._root_parts.append((self._place(mesh), 0.5 * (wrist_far + b),
                                 np.eye(3)))

        # fingers ---------------------------------------------------------
        for f in FINGERS:
            spec = _SPECS[f]
            lens = [l * self.hf for l in spec["lens"]]
            radii = [r * self.hf for r in spec["radii"]]
            sr = [0.5 * (radii[0] + radii[1]),
                  0.5 * (radii[1] + radii[2]),
                  0.5 * (radii[2] + radii[3])]
            pad_long = radii[3] * 1.25
            distal_len = max(0.4, lens[2] - 0.35 * pad_long)
            self._geom[f] = dict(lens=lens, radii=radii, sr=sr, pad_long=pad_long)

            self.seg[f] = [
                self._place(Cylinder(pos=[(0, 0, 0), (0, lens[0], 0)], r=sr[0],
                                     res=14, c=skin)),
                self._place(Cylinder(pos=[(0, 0, 0), (0, lens[1], 0)], r=sr[1],
                                     res=14, c=skin)),
                self._place(Cylinder(pos=[(0, 0, 0), (0, distal_len, 0)], r=sr[2],
                                     res=14, c=skin)),
            ]
            self.jnt[f] = [
                self._place(Sphere(pos=(0, 0, 0), r=radii[0] * 0.96, res=14, c=skin)),
                self._place(Sphere(pos=(0, 0, 0), r=radii[1] * 1.02, res=14, c=skin)),
                self._place(Sphere(pos=(0, 0, 0), r=radii[2] * 1.02, res=14, c=skin)),
            ]
            self.pad[f] = self._place(
                Ellipsoid(pos=(0, 0, 0), axis1=(radii[3] * 1.02, 0, 0),
                          axis2=(0, pad_long, 0), axis3=(0, 0, radii[3] * 0.95),
                          res=16, c=skin))
            self.nail[f] = self._place(
                Ellipsoid(pos=(0, 0, 0), axis1=(radii[3] * 0.70, 0, 0),
                          axis2=(0, radii[3] * 0.85, 0), axis3=(0, 0, radii[3] * 0.24),
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
        m = mesh._rig_matrix
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float)
        for i in range(3):
            m.SetElement(i, 0, float(R[i, 0]))
            m.SetElement(i, 1, float(R[i, 1]))
            m.SetElement(i, 2, float(R[i, 2]))
            m.SetElement(i, 3, float(t[i]))
        m.SetElement(3, 0, 0.0)
        m.SetElement(3, 1, 0.0)
        m.SetElement(3, 2, 0.0)
        m.SetElement(3, 3, 1.0)
        m.Modified()
        mesh.actor.Modified()

    def _tint(self, f, a):
        prev = self._col.get(f)
        if prev is not None and abs(prev - a) < 0.02:
            return
        self._col[f] = a
        c = tuple(SKIN[i] + (SKIN_PRESS[i] - SKIN[i]) * a for i in range(3))
        cn = tuple(NAIL[i] + (NAIL_PRESS[i] - NAIL[i]) * a for i in range(3))
        for m in self.seg[f] + self.jnt[f] + [self.pad[f]]:
            m.color(c)
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

        for mesh, lp, lR in self._root_parts:
            self._set(mesh, R @ lR, self.root + R @ lp)

        # ---- fingers ---------------------------------------------------
        for f in FINGERS:
            t = targets.get(f)
            if t is None:
                continue
            x, y, z, a = t
            spec = _SPECS[f]
            g = self._geom[f]
            base = self.root + R @ self._L(spec["base"])
            pole = R @ np.array([spec["pole"][0] * self.s,
                                 spec["pole"][1], spec["pole"][2]])
            q0, q1, q2, q3 = solve_chain(base, (x, y, z), g["lens"], pole,
                                         self.BEND_MAX, self.COUPLING)

            f0 = _frame_from_dir(q1 - q0)
            f1 = _frame_from_dir(q2 - q1)
            f2 = _frame_from_dir(q3 - q2)
            eye = np.eye(3)

            self._set(self.seg[f][0], f0, q0)
            self._set(self.seg[f][1], f1, q1)
            self._set(self.seg[f][2], f2, q2)
            self._set(self.jnt[f][0], eye, q0)
            self._set(self.jnt[f][1], eye, q1)
            self._set(self.jnt[f][2], eye, q2)

            d3 = q3 - q2
            n3 = float(np.linalg.norm(d3))
            dir3 = d3 / n3 if n3 > 1e-9 else np.array([0.0, 1.0, 0.0])
            self._set(self.pad[f], f2, q3 - dir3 * (0.85 * g["pad_long"]))
            nail = q2 + dir3 * (0.5 * g["lens"][2]) + f2[:, 2] * (0.88 * g["sr"][2])
            self._set(self.nail[f], f2, nail)
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

    def actors(self):
        """All the meshes composing the hand (add them to a Plotter)."""
        return list(self._actors)

    def add_to(self, plotter):
        for m in self._actors:
            plotter.add(m)
        return plotter
