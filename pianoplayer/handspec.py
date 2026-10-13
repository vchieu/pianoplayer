"""Single source of truth for hand geometry (Phase R, R0.2).

All lengths are in cm for a size-L (hf = 1.0) right hand; callers scale by
``hf`` when used.  ``hand3d`` (IK + tubes) and ``handmesh`` (SDF skin) both
import from here so the two meshes can never drift apart again.

Local hand frame (before mirroring for the left hand)::

    origin : wrist centre
    +y     : toward the fingers (into the keyboard)
    +x     : toward the pinky (right hand)
    +z     : up (dorsal side)
"""

from __future__ import annotations

import hashlib

FINGERS = (1, 2, 3, 4, 5)
FINGER_NAMES = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "little"}

# --------------------------------------------------------------------------- #
# finger specs (hf = 1.0, right hand)
#   base : MCP joint position in the local hand frame
#   lens : MCP->PIP, PIP->DIP, DIP->tip lengths (IK chain)
#   radii: MCP, PIP, DIP, fingertip radii (tube/SDF section, pre-FING_R_SCALE)
#   pole : preferred side the finger arches toward (upwards)
# --------------------------------------------------------------------------- #
# NOTE (BUG-2 partial fix, R2.5/R5.3): thumb MCP->tip was 9.7 cm = 115% of
# the index finger (real hands ~65-80%).  Shortened to 7.3 cm (87% of index,
# inside the ±10% anatomy band) and the MCP moved forward to compensate reach
# (THUMB_BASE_Y0/FWD); the full 6.1 cm anatomical thumb needs the CMC
# articulation (R5.3, deferred -- see Deviations).
FINGER_SPECS = {
    1: dict(base=(-5.6, 3.0, 0.45), lens=(3.2, 2.2, 1.9),
            radii=(1.06, 0.96, 0.83, 0.70), pole=(0.15, 0.55, 1.00)),
    2: dict(base=(-2.1, 8.2, 1.85), lens=(3.9, 2.5, 2.0),
            radii=(0.96, 0.89, 0.77, 0.65), pole=(0.00, 0.35, 1.00)),
    3: dict(base=(0.15, 8.6, 1.95), lens=(4.3, 2.7, 2.1),
            radii=(0.98, 0.91, 0.79, 0.66), pole=(0.00, 0.35, 1.00)),
    4: dict(base=(2.4, 8.3, 1.80), lens=(3.9, 2.5, 2.0),
            radii=(0.93, 0.86, 0.74, 0.62), pole=(0.00, 0.35, 1.00)),
    5: dict(base=(4.5, 8.1, 1.60), lens=(3.2, 2.1, 1.8),
            radii=(0.86, 0.79, 0.68, 0.57), pole=(0.00, 0.30, 1.00)),
}

# visual finger-width multiplier (IK only uses lens, so reach is untouched).
# NOTE (BUG-3 baseline): the SDF skin used a second, larger multiplier table
# (_SDF_R_SCALE 1.30-1.58) on top of this one; it is removed in R1.2 and the
# SDF uses these same radii.
FING_R_SCALE = {1: 1.0, 2: 1.10, 3: 1.12, 4: 1.13, 5: 1.13}

# R1.2: MCP spacing shrink (fraction) applied to fingers 2-5 about the
# middle-finger base x.  2% closes the rest gap to [0, 0.15] cm together
# with the de-inflated SDF radii; bases move <= 0.26 cm (reach-safe: the
# root chases the fan mean and IK stays exact).
MCP_SHRINK = 0.02
_MCP_PIVOT_F = 3


def mcp_base(f):
    """MCP base (x, y, z) at hf=1 with MCP_SHRINK applied (fingers 2-5)."""
    bx, by, bz = FINGER_SPECS[f]["base"]
    if f in (2, 3, 4, 5) and MCP_SHRINK:
        px = FINGER_SPECS[_MCP_PIVOT_F]["base"][0]
        bx = px + (bx - px) * (1.0 - MCP_SHRINK)
    return (bx, by, bz)


def mcp_spacing():
    """Adjacent MCP gaps (hf=1 units) for fingers 2-5."""
    xs = sorted(mcp_base(f)[0] for f in (2, 3, 4, 5))
    return [b - a for a, b in zip(xs[:-1], xs[1:])]


def finger_radius(f, joint=1):
    """Visual half-width of finger `f` at joint 0..3 (hf=1 units).

    R1.2: the SDF skin uses this same radius (the old extra _SDF_R_SCALE
    inflation table is deleted); MCP radius stays <= 0.5 x MCP spacing.
    """
    return FINGER_SPECS[f]["radii"][joint] * FING_R_SCALE[f]


def thumb_index_ratio():
    """Thumb MCP->tip length over index MCP->tip length."""
    th = sum(FINGER_SPECS[1]["lens"])
    ix = sum(FINGER_SPECS[2]["lens"])
    return th / ix


# --------------------------------------------------------------------------- #
# arm + palm loft profile: (y, centre-x, half-width, half-thickness, p)
# p = 2 is a circle, larger is flatter-sided like the edge of a palm.
# --------------------------------------------------------------------------- #
ARM_PROFILE = (
    (-46.0, 0.00, 3.60, 2.70, 2.00),
    (-30.0, 0.00, 3.40, 2.60, 2.00),
    (-16.0, 0.00, 3.65, 2.30, 2.05),
    (-9.0, 0.00, 3.60, 1.90, 2.15),
    (-4.5, 0.00, 3.45, 1.55, 2.35),
    (-1.8, 0.00, 3.30, 1.40, 2.55),      # wrist (narrowest)
    (0.8, -0.35, 3.55, 1.36, 2.80),
    (3.2, -0.70, 4.05, 1.48, 3.00),
    (5.8, -0.10, 4.50, 1.46, 3.10),
    (8.2, 1.00, 4.55, 1.32, 3.05),      # knuckle line (widest)
    (9.2, 1.00, 3.90, 0.95, 2.80),
    (10.0, 1.00, 0.60, 0.55, 2.20),     # closed cap
)
ARM_PIN = 2            # first stations keep their y (out-of-frame arm)
THENAR = 0.28          # thumb-side bulge over the palm (R1.1: 0.50 -> 0.28)
HYPO = 0.08            # pinky-side bulge (R1.1: 0.12 -> 0.08)

# R1.1 target half-widths (hf=1), wrist narrowest, monotonic both ways over
# y in [-16, 8.2] (stations past 8.2 close the cap and are exempt).
PALM_TARGET_A1 = {
    -16.0: 3.65, -9.0: 3.60, -4.5: 3.45, -1.8: 3.30, 0.8: 3.55,
    3.2: 4.05, 5.8: 4.50, 8.2: 4.55,
}
WRIST_Y = -1.8
KNUCKLE_Y = 8.2


def check_palm_monotonic(profile=None):
    """Half-width strictly increasing both ways from the wrist.

    Returns (ok, message).  Only stations with -16 <= y <= 8.2 participate
    (far arm + end cap are exempt).
    """
    prof = ARM_PROFILE if profile is None else profile
    pts = sorted((y, a1) for (y, _c, a1, _a2, _p) in prof
                 if -16.0 <= y <= 8.2)
    if not pts:
        return False, "no stations in [-16, 8.2]"
    wi = min(range(len(pts)), key=lambda i: abs(pts[i][0] - WRIST_Y))
    for i in range(wi, 0, -1):
        if not pts[i - 1][1] > pts[i][1]:
            return False, ("proximal violation: a1(%g)=%.2f <= a1(%g)=%.2f"
                            % (pts[i - 1][0], pts[i - 1][1],
                               pts[i][0], pts[i][1]))
    for i in range(wi, len(pts) - 1):
        if not pts[i + 1][1] > pts[i][1]:
            return False, ("distal violation: a1(%g)=%.2f <= a1(%g)=%.2f"
                            % (pts[i + 1][0], pts[i + 1][1],
                               pts[i][0], pts[i][1]))
    return True, "monotonic both ways from wrist y=%g" % WRIST_Y


def wrist_palm_ratio():
    """Wrist half-width over knuckle half-width (target 0.68-0.80)."""
    w = dict((y, a1) for (y, _c, a1, _a2, _p) in ARM_PROFILE)
    return w[WRIST_Y] / w[KNUCKLE_Y]


# --------------------------------------------------------------------------- #
# wrist-root placement (scales with hf; tuned for test_hand_reach 70 cells)
# --------------------------------------------------------------------------- #
ROOT_CONTACT_Y = -3.0
ROOT_WRIST_GAP = 8.6
ROOT_REACH_FRAC = 0.69
ROOT_Z_BASE = 2.0
ROOT_Z_C0 = 0.50
ROOT_Z_HF_K = 0.60
THUMB_BASE_Y0 = 3.55
THUMB_FWD = 4.4
HF_XXS = 0.33
PINKY_CONTACT_DY = 0.80
# D1c (thumb twin): the shortened anatomical thumb contacts its key a touch
# nearer the player (world cm, both colours; still on the key).
THUMB_CONTACT_DY = 0.65


def thumb_base_y(hf):
    """Thumb MCP y (world cm): forward shift grows with hand size (D1b)."""
    return max(THUMB_BASE_Y0, THUMB_BASE_Y0 + THUMB_FWD * (hf - HF_XXS))


# --------------------------------------------------------------------------- #
# thumb CMC anatomy (R2.5 targets; IK keeps FINGER_SPECS lens until R5.3)
# NOTE (honest status): THUMB_CMC_LEN / THUMB_MCP_LEN / THUMB_IP_LEN
# and THENAR_ANGLE_DEG / THUMB_ABDUCT_DEG are anatomy targets for the
# articulated CMC (R5.3, deferred): IK and the SDF still use the single
# FINGER_SPECS[1] chain until then. They feed geom_hash() so retuning
# rebuilds the cache. THUMB_CMC_A (thenar origin) IS wired into the
# geometry (rest_chains + SDF thenar pair).
THUMB_CMC_A = (-3.0, -0.5, 0.3)  # thenar origin (hf=1, right-hand local)
THUMB_CMC_LEN = 4.4      # CMC -> MCP (metacarpal) x hf
THUMB_MCP_LEN = 3.3      # MCP -> IP (proximal phalanx) x hf
THUMB_IP_LEN = 2.8       # IP -> tip (distal phalanx) x hf
THENAR_ANGLE_DEG = 35.0  # thenar mound inclination along the metacarpal
THUMB_ABDUCT_DEG = 25.0  # natural thumb abduction (top view)

# thenar-web ellipsoid (palm frame radii x hf; centre y follows thumb MCP)
WEB_RADII = (1.20, 2.40, 1.10)
WEB_POS_X = -3.90
WEB_POS_Z = 0.70

# --------------------------------------------------------------------------- #
# rest pose = average playing pose (R2.6 targets)
#
# NOTE (R2.6 honest status): only REST_DIP_RATIO, REST_SPREAD,
# REST_REACH_FRAC and REST_LIFT_Z shape the geometry -- rest_chains()
# places a target (spread + REACH_FRAC of the chain length + lift) and
# solves it with solve_chain(coupling=REST_DIP_RATIO). REST_MCP_DEG /
# REST_PIP_DEG are the angle targets that pose is meant to approximate;
# they are NOT wired into the solver (a target-based solve cannot take
# joint angles as input) -- they pin the intent and feed geom_hash() so
# retuning them rebuilds the cache. Do not fake-wire them.
# --------------------------------------------------------------------------- #
REST_MCP_DEG = 20.0      # target MCP flexion of the rest pose (hash only)
REST_PIP_DEG = 35.0      # target PIP flexion of the rest pose (hash only)
REST_DIP_RATIO = 0.65    # DIP = 0.65 x PIP (wired: solve_chain coupling)
REST_SPREAD = 0.30       # lateral splay per finger step (x hf)
REST_REACH_FRAC = 0.92   # rest target distance as a fraction of chain length
REST_LIFT_Z = 0.15       # rest target lift above the MCP plane (x hf)

# --------------------------------------------------------------------------- #
# SDF v2 section parameters
# --------------------------------------------------------------------------- #
FING_Z_FLAT = (0.88, 0.84, 0.80)  # z-radius scale per segment (elliptical)
TIP_PAD_FLAT = 0.72      # fingertip pad ellipsoid flattening (z)
PIP_MOUND = 1.06         # PIP joint sphere radius multiplier
DIP_MOUND = 1.03         # DIP joint sphere radius multiplier
MCP_MOUND_R = 0.55       # MCP mound sphere radius as a fraction of spacing
PALM_ARCH = 0.18         # dorsal arch height factor (z(x) curve) --
                         # target only (hash): the loft/SDF approximate the
                         # arch via the station thickness profile (R-dev3),
                         # no explicit transverse arch term exists yet
WEB_K = 0.25             # smin k inside the interdigital web band
WEB_Y_LO = -1.6          # web band lower edge rel. knuckle line (x hf)
WEB_Y_HI = 1.2           # web band upper edge rel. knuckle line (x hf)
THUMB_WEB_K = 0.50       # thumb<->index web uses a wider blend
THUMB_WEB_FRAC = 0.40    # ... extending over this fraction of proximal

K_INTRA = 0.05           # within one finger: near-hard union
K_PALM = 0.35            # finger <-> palm/thenar: webbing bridges
K_CMC = 0.40             # thumb CMC <-> palm
K_BODY = 0.30            # forearm/palm ellipsoid chain

GRID_STEP = 0.15         # Deviation (H6): 0.12 -> 0.15, SDF ~3s not ~7s
GRID_Y0 = -24.0          # fixed proximal border (stub pole closes inside)
TRIS_MAX = 8000
HANDMESH_VERSION = "v4"  # v4: thumb MCP base matches the rig (was /hf off),
                         # rest_chains coupling 0.4 -> REST_DIP_RATIO 0.65.
                         # v3: SDF v2 (elliptical sections, pad tips, y-web)


# --------------------------------------------------------------------------- #
# F4a: chord span limits (cm at hf=1, right hand).  MAX_SPAN is the hard
# limit (HARD cost beyond), COMFORT_SPAN is where quadratic cost starts.
# Symmetric: MAX_SPAN[fa][fb] == MAX_SPAN[fb][fa].
MAX_SPAN = {
    1: {2: 7.0, 3: 9.0, 4: 11.0, 5: 13.0},
    2: {1: 7.0, 3: 5.5, 4: 8.0, 5: 11.0},
    3: {1: 9.0, 2: 5.5, 4: 5.5, 5: 8.0},
    4: {1: 11.0, 2: 8.0, 3: 5.5, 5: 5.5},
    5: {1: 13.0, 2: 11.0, 3: 8.0, 4: 5.5},
}
COMFORT_SPAN = {
    1: {2: 5.0, 3: 6.5, 4: 8.0, 5: 9.5},
    2: {1: 5.0, 3: 4.0, 4: 6.0, 5: 8.5},
    3: {1: 6.5, 2: 4.0, 4: 4.0, 5: 6.0},
    4: {1: 8.0, 2: 6.0, 3: 4.0, 5: 4.0},
    5: {1: 9.5, 2: 8.5, 3: 6.0, 4: 4.0},
}

# F5: crossing costs.  CROSS_BASE[fa][fb] for fa < fb (higher finger over
# lower).  Applied when (fb - fa) * xba < 0 (crossing).  Cost = CROSS_BASE *
# max(1, semitone_interval / 3) * max(1, 0.2 / dt_sec).  Thumb-under uses
# UNDER[fa] where fb=1 (thumb) passes under fa.
CROSS_BASE = {
    (2, 3): 1.5, (2, 4): 2.0, (2, 5): 2.5,
    (3, 4): 1.2, (3, 5): 1.8,
    (4, 5): 1.0,
}
UNDER = {2: 3.0, 3: 1.0, 4: 8.0, 5: 9.0}  # thumb-under cost after finger:
        # pedagogy prefers tucking after the middle finger (3); after the
        # ring (4) only when topography demands it (F major: 4 lands on
        # Bb); after 2/5 almost never. Margins must exceed ~2.5 cost units
        # or downstream strong-finger benefits (3 on black keys) outweigh
        # them (G-major gold regression). Halved (in _crossing_cost) when
        # departing from a black key onto white: the hand is already
        # raised, so the classic Bb->C tuck rides the hand position.

# F6: static black/white preference per finger (added to pair_cost, not
# multiplied by velocity).  Positive = penalty for black keys.
BLACK_COST = {1: 0.5, 2: 0.1, 3: 0.0, 4: 0.1, 5: 0.3}
WHITE_COST = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}


def geom_hash():
    """8-char sha1 over geometry + SDF constants + grid + version (R1.4).

    NOTE: the hash covers GEOMETRY constants only. Fingering cost tables
    (MAX_SPAN, COMFORT_SPAN, CROSS_BASE, UNDER, BLACK_COST, WHITE_COST)
    are deliberately EXCLUDED: retuning them must not invalidate the mesh
    cache. Geometry LOGIC changes (e.g. a rest_chains formula or coupling
    edit) are invisible to it -- bump HANDMESH_VERSION manually with every
    such change (v3 -> v4 above was exactly this case: the thumb formula
    reuses unhashed THUMB_BASE_Y0 / THUMB_FWD inputs differently, and the
    coupling change reuses the unchanged REST_DIP_RATIO value, so the hash
    alone rebuilt nothing).
    """
    rep = repr((FINGER_SPECS, FING_R_SCALE, MCP_SHRINK, ARM_PROFILE,
                ARM_PIN, THENAR, HYPO, WEB_RADII, THUMB_CMC_A,
                THUMB_CMC_LEN, THUMB_MCP_LEN, THUMB_IP_LEN,
                THUMB_BASE_Y0, THUMB_FWD, HF_XXS,  # thumb-MCP forward shift
                REST_MCP_DEG, REST_PIP_DEG, REST_DIP_RATIO, REST_SPREAD,
                REST_REACH_FRAC, REST_LIFT_Z, FING_Z_FLAT, TIP_PAD_FLAT,
                PIP_MOUND, DIP_MOUND, MCP_MOUND_R, PALM_ARCH, WEB_K,
                WEB_Y_LO, WEB_Y_HI, THUMB_WEB_K, THUMB_WEB_FRAC,
                K_INTRA, K_PALM, K_CMC, K_BODY, GRID_STEP, GRID_Y0,
                TRIS_MAX, HANDMESH_VERSION))
    return hashlib.sha1(rep.encode()).hexdigest()[:8]
