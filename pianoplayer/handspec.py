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
import math

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
            # v5: proximal radii 1.06/0.96 -> 0.98/0.88 (thumb/index groove:
            # the thumb is ungraded by gap50/pip_spacing, which only cover
            # pairs (2,3)/(3,4)/(4,5); slimmer thumb + narrower web below
            # carve the thenar groove instead of a melted blend).
            radii=(0.98, 0.88, 0.83, 0.60), pole=(0.15, 0.55, 1.00)),
    2: dict(base=(-2.1, 8.2, 1.85), lens=(3.9, 2.5, 2.0),
            radii=(0.96, 0.89, 0.77, 0.55), pole=(0.00, 0.35, 1.00)),
    3: dict(base=(0.15, 8.6, 1.95), lens=(4.3, 2.7, 2.1),
            radii=(0.98, 0.91, 0.79, 0.56), pole=(0.00, 0.35, 1.00)),
    4: dict(base=(2.4, 8.3, 1.80), lens=(3.9, 2.5, 2.0),
            radii=(0.93, 0.86, 0.74, 0.53), pole=(0.00, 0.35, 1.00)),
    5: dict(base=(4.5, 8.1, 1.60), lens=(3.2, 2.1, 1.8),
            radii=(0.86, 0.79, 0.68, 0.48), pole=(0.00, 0.30, 1.00)),
}

# visual finger-width multiplier (IK only uses lens, so reach is untouched).
# NOTE (BUG-3 baseline): the SDF skin used a second, larger multiplier table
# (_SDF_R_SCALE 1.30-1.58) on top of this one; it is removed in R1.2 and the
# SDF uses these same radii.
FING_R_SCALE = {1: 1.0, 2: 1.10, 3: 1.12, 4: 1.13, 5: 1.13}

# v5 (inter-finger gaps): distal taper per joint index (MCP, PIP, DIP, tip).
# The green spec gates pin the proximal section: gap50/pip_spacing are graded
# on joints 0-1 only, with gap50 headroom +0.015 cm (uniform thinning to 1.00
# would push gap50 to ~0.32 vs max 0.15 and pip_spacing to ~0.85 vs min 0.85),
# so only DIP/tip slim down. Real fingers taper distally; the PIP knuckle
# (PIP_MOUND reads max(R1, R2)) keeps its size since R1 is untouched.
FING_R_TAPER = {0: 1.0, 1: 1.0, 2: 0.85, 3: 0.80}

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
    v5: distal taper (FING_R_TAPER) slims DIP/tip so neighbours separate
    past the knuckles; joints 0-1 (the graded proximal section) untouched.
    """
    return (FINGER_SPECS[f]["radii"][joint] * FING_R_SCALE[f]
            * FING_R_TAPER[joint])


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
ROOT_Z_ARCH = 1.0          # extra arch lift, only above hf 0.5: small hands
                           # (short fingers) cannot reach down from a high
                           # wrist (solver returns bend=0, 23 cells fail at
                           # flat +50%); large hands get the full dome.
                           # 2.0 adds M-thumb-black + L/XL-pinky fails, so
                           # 1.0 (M +0.32): no NEW fails vs 67/70 baseline.
                           # ROOT_Z = BASE*(C0+K*hf) + ARCH*max(0,hf-0.5).
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
REST_MCP_DEG = 30.0      # target MCP flexion of the rest pose (hash only)
REST_PIP_DEG = 45.0      # target PIP flexion of the rest pose (hash only)
REST_DIP_RATIO = 0.65    # DIP = 0.65 x PIP (wired: solve_chain coupling)
REST_SPREAD = 0.30       # lateral splay per finger step (x hf)
REST_REACH_FRAC = 0.85   # rest target distance as a fraction of chain length
                         # (0.92 -> 0.85: folded dome, not a flat paddle)
REST_LIFT_Z = 0.45       # rest target lift above the MCP plane (x hf)
                         # (0.15 -> 0.45: knuckle dome like a playing hand)

# --------------------------------------------------------------------------- #
# SDF v2 section parameters
# --------------------------------------------------------------------------- #
FING_Z_FLAT = (0.88, 0.84, 0.80)  # z-radius scale per segment (elliptical)
TIP_PAD_FLAT = 0.72      # fingertip pad ellipsoid flattening (z)
PIP_MOUND = 1.03         # PIP joint sphere radius multiplier (was 1.06: the
                         # near-hard union groove rendered as a dark zigzag
                         # crack under side light; DIP already 1.03)
DIP_MOUND = 1.03         # DIP joint sphere radius multiplier
MCP_MOUND_R = 0.55       # MCP mound sphere radius as a fraction of spacing
PALM_ARCH = 0.18         # dorsal arch height factor (z(x) curve) --
                         # target only (hash): the loft/SDF approximate the
                         # arch via the station thickness profile (R-dev3),
                         # no explicit transverse arch term exists yet
WEB_K = 0.25             # smin k inside the interdigital web band
WEB_Y_LO = -1.6          # web band lower edge rel. knuckle line (x hf)
WEB_Y_HI = 0.6           # web band upper edge rel. knuckle line (x hf)
                         # (v5: 1.2 -> 0.6 -- the old band reached ~1 cm past
                         # the knuckles and bridged the distal phalanges;
                         # fusion now stays proximal like a real web)
THUMB_WEB_K = 0.35       # thumb<->index web uses a wider blend
                         # (v5: 0.50 -> 0.35 -- the old k filled the thenar
                         # groove completely; the web stays, a groove reads)
THUMB_WEB_FRAC = 0.40    # ... extending over this fraction of proximal

K_INTRA = 0.05           # within one finger: near-hard union
K_PALM = 0.35            # finger <-> palm/thenar: webbing bridges
K_CMC = 0.40             # thumb CMC <-> palm
K_BODY = 0.30            # forearm/palm ellipsoid chain

GRID_STEP = 0.15         # Deviation (H6): 0.12 -> 0.15, SDF ~3s not ~7s
GRID_Y0 = -24.0          # fixed proximal border (stub pole closes inside)
TRIS_MAX = 8000
HANDMESH_VERSION = "v5"  # v5: distal taper (FING_R_TAPER on DIP/tip) opens
                         # inter-finger gaps past the knuckles; WEB_Y_HI
                         # 1.2 -> 0.6 keeps the webbing proximal. Proximal
                         # radii/grading (gap50, pip_spacing) unchanged.
                         # v4: thumb MCP base matches the rig (was /hf off),
                         # rest_chains coupling 0.4 -> REST_DIP_RATIO 0.65.
                         # v3: SDF v2 (elliptical sections, pad tips, y-web)


# --------------------------------------------------------------------------- #
# Anatomy (step 3): the 19-bone hand under anatomical names (idea-port of
# pygame-piano-animator `pianist.py` DEFAULT_ANATOMY + its `wrist` slider).
#
# The 3D rig already HAS those 19 bones, only grouped differently (18 skinned
# bones in handmesh.rest_bones):
#
#   forearm     the radius -- not a hand bone, no source counterpart
#   palm        the five metacarpals mc1..mc5 merged into one bone
#   f1_cmc      the thumb's own thenar -> MCP span (mc1 gets its own bone)
#   f{2..5}_0   pp  proximal phalanx (MCP -> PIP)
#   f{2..5}_1   mp  middle phalanx   (PIP -> DIP)
#   f{2..5}_2   dp  distal phalanx   (DIP -> tip)
#   f1_0/1/2    pp1 / dp1 / the thumb pad past the IP joint
#
# => 2 rigid + 1 thumb CMC + 15 phalanx bones carry the source's 5 metacarpals
# + 14 phalanges (the thumb has no middle phalanx).
#
# NOTHING here adds a bone or a new geometry constant: every number below is
# DERIVED from what is already above -- FINGER_SPECS lens/radii/base,
# ARM_PROFILE, THUMB_CMC_*, thumb_base_y() -- i.e. from exactly the values
# geom_hash() already covers, so retuning the anatomy rebuilds the mesh cache
# and the defaults are the shipped rig's own numbers (bit-identical; see
# Deviations D11).  What this step adds is the NAMED VIEW: the source's 19
# numbers can be read, compared against a real hand and retuned without
# touching the solver.
# --------------------------------------------------------------------------- #
BONE_KIND = {"mc": "metacarpal", "pp": "proximal phalanx",
             "mp": "middle phalanx", "dp": "distal phalanx"}
# the source's 19 bones in its own order: id -> (finger, kind)
ANATOMY_BONES = (
    ("mc1", 1, "mc"), ("pp1", 1, "pp"), ("dp1", 1, "dp"),
    ("mc2", 2, "mc"), ("pp2", 2, "pp"), ("mp2", 2, "mp"), ("dp2", 2, "dp"),
    ("mc3", 3, "mc"), ("pp3", 3, "pp"), ("mp3", 3, "mp"), ("dp3", 3, "dp"),
    ("mc4", 4, "mc"), ("pp4", 4, "pp"), ("mp4", 4, "mp"), ("dp4", 4, "dp"),
    ("mc5", 5, "mc"), ("pp5", 5, "pp"), ("mp5", 5, "mp"), ("dp5", 5, "dp"),
)
# the source's own default hand, in its model units (1 unit ~= 1.09 cm there):
# reference only, for comparing bone_lengths() against.  Each bone there may
# be 0.70..2.50 x its default (metacarpals included); the wrist width 0.75..
# 2.00 is a separate slider.  This rig has no per-pianist store (one hand,
# scaled by hf), so the ranges are not knobs: the bone constants themselves
# are the knobs and they are hashed.
SOURCE_UNIT_CM = 1.09
SOURCE_ANATOMY = {
    "mc1": 4.7, "pp1": 3.3, "dp1": 2.8,
    "mc2": 6.684, "pp2": 4.0, "mp2": 2.3, "dp2": 1.8,
    "mc3": 6.626, "pp3": 4.5, "mp3": 2.7, "dp3": 1.9,
    "mc4": 6.332, "pp4": 4.2, "mp4": 2.6, "dp4": 1.9,
    "mc5": 5.974, "pp5": 3.3, "mp5": 1.8, "dp5": 1.7,
}


def bone_lengths():
    """Anatomical bone lengths [cm] at hf=1: this rig's DEFAULT_ANATOMY view.

    Keys are the source's bone ids (ANATOMY_BONES).  A metacarpal is the
    wrist -> MCP distance (`mcp_base(f).y` against WRIST_Y); the thumb's mc1 is
    the whole thenar -> MCP span (THUMB_CMC_A -> the thumb MCP at
    `thumb_base_y(1.0)`), because the 3D rig carries the CMC as its own bone
    (f1_cmc).  Phalanges are the FINGER_SPECS lens; the thumb's third segment
    (past the IP joint -- the rounded pad) has no source bone and is not
    listed.

    NOTE on the metacarpals: they read ~10 cm here against the source's ~7 cm
    (6.7 units x 1.09) because the rig's WRIST_Y (-1.8) is the *centre of the
    wrist loft*, ~2.7 cm proximal to the wrist joint the source measures
    metacarpals from.  The phalanges -- the part the solver actually flexes --
    line up within a few tenths of a centimetre.
    """
    out = {}
    for bid, f, kind in ANATOMY_BONES:
        if kind == "mc":
            if f == 1:
                bx, _by, bz = FINGER_SPECS[1]["base"]
                out[bid] = math.dist(THUMB_CMC_A,
                                     (bx, thumb_base_y(1.0), bz))
            else:
                _x, by, _z = mcp_base(f)
                out[bid] = abs(by - WRIST_Y)
            continue
        i = {"pp": 0, "mp": 1, "dp": 2}[kind]
        out[bid] = float(FINGER_SPECS[f]["lens"][i])
    return out


def segment_ratio(f):
    """Per-finger phalanx ratios MCP:PIP:DIP (== pp:mp:dp) over the chain
    length: the source's per-bone proportions, normalised to 1."""
    lens = FINGER_SPECS[f]["lens"]
    tot = float(sum(lens))
    return tuple(float(v) / tot for v in lens)


def knuckle_row():
    """MCP (knuckle) positions at hf=1: {finger: (x, y, z)} -- the heights of
    the knuckle dome (the middle finger rides 0.35 cm above the little one;
    REST_LIFT_Z domes the rest pose over it)."""
    return {f: mcp_base(f) for f in FINGERS}


def wrist_width():
    """Wrist half-width at hf=1 [cm]: the source's `wrist` slider scales this
    row of ARM_PROFILE (0.75..2.0 there); here it is hashed with the rest of
    the profile, so a retune rebuilds the mesh."""
    for (y, _c, a1, _a2, _p) in ARM_PROFILE:
        if y == WRIST_Y:
            return a1
    return float(PALM_TARGET_A1.get(WRIST_Y, 0.0))


# --------------------------------------------------------------------------- #
# F4a: chord span limits (cm at hf=1, right hand).  MAX_SPAN is the hard
# limit (HARD cost beyond), COMFORT_SPAN is where quadratic cost starts.
# Symmetric: MAX_SPAN[fa][fb] == MAX_SPAN[fb][fa].
# Thumb rows (Phase 1 step 1.2): old values (7/9/11/13 and 5/6.5/8/9.5)
# were below an octave (16.5 cm) for 1-5.  Max thumb-pinky = the 21*hf cm
# "relaxed span" printed by Hand.
MAX_SPAN = {
    1: {2: 11.0, 3: 14.0, 4: 17.0, 5: 21.0},
    2: {1: 11.0, 3: 5.5, 4: 8.0, 5: 11.0},
    3: {1: 14.0, 2: 5.5, 4: 5.5, 5: 8.0},
    4: {1: 17.0, 2: 8.0, 3: 5.5, 5: 5.5},
    5: {1: 21.0, 2: 11.0, 3: 8.0, 4: 5.5},
}
COMFORT_SPAN = {
    1: {2: 7.0, 3: 10.0, 4: 13.0, 5: 15.5},
    2: {1: 7.0, 3: 4.0, 4: 6.0, 5: 8.5},
    3: {1: 10.0, 2: 4.0, 4: 4.0, 5: 6.0},
    4: {1: 13.0, 2: 6.0, 3: 4.0, 5: 4.0},
    5: {1: 15.5, 2: 8.5, 3: 6.0, 4: 4.0},
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
# Figure-lite round (learned direction: thumb_black 1.92 -> 0.24, pinky up):
# thumb 0.5 -> 0.25, pinky 0.3 -> 0.35.
BLACK_COST = {1: 0.25, 2: 0.1, 3: 0.0, 4: 0.1, 5: 0.35}
WHITE_COST = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

# Figure-lite: one finger covering two neighbouring keys inside a chord
# (soft: the hand really can do it) -- HARD otherwise.
THUMB_DOUBLE = 7.0
PINKY_DOUBLE = 8.0
# Figure-lite: octave (or wider) on 1-4 with an inner note held down --
# 1-5 opens the hand, so 1-4 pays this (soft, physical limits still win).
INNER_ROOM_14 = 4.0

# Procedural skin texture (handmesh.build_skin_texture). 512 default;
# PIANOPLAYER_TEXSIZE=1024 selects the hi-res variant (own cache file).
TEX_VERSION = "v3"
TEX_SIZE_DEFAULT = 512

# Cheap skin material (hand3d._SKIN_*): softer wrap-style response --
# ambient up, diffuse down slightly, visible specular highlight.
SKIN_AMBIENT = 0.45
SKIN_DIFFUSE = 0.65
SKIN_SPECULAR = 0.15
SKIN_SHININESS = 24.0

# --------------------------------------------------------------------------- #
# Skins (step 2): idea-port of pygame-piano-animator `skins.py` styles onto
# the 3D material / texture / vertex-colour layers.  NOT a line port: that
# source draws 2D top-down polygons, here the same *looks* are expressed as
# a texture variant + vertex-colour bands + material coefficients.
#
# `cartoon` is the shipped look and MUST stay pixel-identical: it reuses the
# existing texture, SKIN colour, nail patch and material coefficients below
# unchanged.  `gloves` / `robot` recolour + texture-variant; `skeleton` is
# bones only (falls back to the tubes mesh with a dark colour).
#
# All look constants, EXCLUDED from geom_hash() like every other look knob:
# a style never changes geometry, it only changes how the geometry is
# painted (the cache filename carries the style instead).
# --------------------------------------------------------------------------- #
STYLE_ORDER = ("cartoon", "gloves", "robot", "skeleton")
STYLE_DEFAULT = "cartoon"

# material per style (ambient, diffuse, specular, shininess); cartoon is
# literally the SKIN_* values above so the default look cannot drift.
STYLE_MATERIAL = {
    "cartoon": dict(ambient=SKIN_AMBIENT, diffuse=SKIN_DIFFUSE,
                     specular=SKIN_SPECULAR, shininess=SKIN_SHININESS),
    # white latex: a touch more ambient (it is bright), soft spec
    "gloves": dict(ambient=0.52, diffuse=0.68, specular=0.22,
                   shininess=30.0),
    # painted shell plates: darker base response + a gloss highlight.
    # 0.30/40 (not 0.48/64): a tighter, stronger highlight blew the white
    # shell out to pure white on the whole dorsum and swallowed the dark
    # joint rings + tip caps the style is made of.
    "robot": dict(ambient=0.34, diffuse=0.62, specular=0.30,
                  shininess=40.0),
    # dark bones: mostly diffuse, barely any highlight
    "skeleton": dict(ambient=0.42, diffuse=0.74, specular=0.08,
                     shininess=16.0),
}

# vertex-colour palette per style: `base` (tubes + skinned fallback + the
# actor colour used when no texture), `press` (tube press tint), `nail`
# patch on the back of the distal segment + how far it is mixed in,
# `press_mul` (skinned mesh: scalars multiply the texture, so this is a
# multiply, not a colour) and the blob `shadow` colour.
STYLE_COLORS = {
    "cartoon": dict(base=(0.95, 0.80, 0.70), press=(0.96, 0.72, 0.64),
                    nail=(0.97, 0.82, 0.74), nail_mix=0.6,
                    press_mul=(1.00, 0.88, 0.90),
                    shadow=(0.055, 0.035, 0.035)),
    # gloves: white latex, no nails (the tip caps are the only dark patch),
    # press reads as a slight grey squeeze
    "gloves": dict(base=(0.97, 0.97, 0.97), press=(0.88, 0.88, 0.90),
                   nail=(0.99, 0.99, 0.99), nail_mix=0.0,
                   press_mul=(0.94, 0.94, 0.97),
                   shadow=(0.05, 0.05, 0.06)),
    # robot: light shell with dark fingertip caps (the "nail" patch is the
    # cap, mixed in almost fully)
    "robot": dict(base=(0.90, 0.90, 0.88), press=(0.80, 0.81, 0.84),
                  nail=(0.21, 0.21, 0.23), nail_mix=0.85,
                  press_mul=(0.90, 0.92, 0.95),
                  shadow=(0.06, 0.06, 0.07)),
    # skeleton: dark bones only (tubes mode; no texture, no nails)
    "skeleton": dict(base=(0.11, 0.11, 0.13), press=(0.17, 0.17, 0.20),
                     nail=(0.11, 0.11, 0.13), nail_mix=0.0,
                     press_mul=(1.0, 1.0, 1.0),
                     shadow=(0.03, 0.03, 0.035)),
}

# Vertex-colour bands painted onto the skinned mesh (they multiply the
# texture, so they are dark).  All lengths in cm x hf, like the rest of the
# geometry knobs.  Source mapping (skins.py):
#   gloves cuff  -> `sleeve` puffy cuff across the wrist + thin dark arm
#   robot joints  -> `joint` dark metal cylinders at MCP/PIP/DIP
#   robot tips    -> `tip` dark caps over the last phalanx
STYLE_BANDS = {
    "gloves": dict(cuff_y=-1.4, cuff_h=1.3,        # dark forearm beyond this
                   cuff_col=(0.10, 0.10, 0.12),
                   puff_col=(0.93, 0.93, 0.94),   # puffy white cuff band
                   puff_h=1.0),
    "robot": dict(shell=(0.90, 0.90, 0.88),
                  joint_col=(0.24, 0.25, 0.28),
                  joint_r=0.42,                   # half-length of a ring
                                                #   band along the bone
                  joint_band_r=1.15,              # radial reach of a ring
                  tip_col=(0.21, 0.21, 0.23),
                  tip_r=0.95),                    # radius of the end cap
}


def normalize_style(style=None):
    """Canonical style name (unknown/empty -> the default, 'cartoon')."""
    s = str(style or "").strip().lower()
    return s if s in STYLE_ORDER else STYLE_DEFAULT


def style_look(style=None):
    """Complete look dict for `style`: colors + material + vertex bands."""
    s = normalize_style(style)
    look = dict(STYLE_COLORS[s])
    look["style"] = s
    look.update(STYLE_MATERIAL[s])
    look["bands"] = STYLE_BANDS.get(s, {})
    # 'cartoon' must resolve to the module-level skin constants
    return look

# Cheap motion flavour (hand3d.update, all visual-only, IK stays exact):
# wrist bounce on quick re-strikes, idle wander, sympathy dip, white-key
# area ride and loudness (mean-press) wrist lift.
BOUNCE_LIFT = 0.25      # cm root-z kick on a quick re-press
BOUNCE_TAU = 0.18       # ... decay [s]
WANDER_AMP = 0.05       # cm idle-finger x wander amplitude
WANDER_FREQ = 0.23      # ... [Hz]
SYMPATHY_DIP = 0.10     # cm idle neighbour dips per adjacent press
WHITE_RIDE = 0.15       # cm idle fingers ride higher over black-key areas
LOUD_LIFT = 0.12        # cm root-z lift per mean press (effort)

# --------------------------------------------------------------------------- #
# Motion step 1 (ported from pygame-piano-animator hands.py + pianist.py
# BEHAVIORS at their defaults; idea-port, not line-port: the 2D solver works
# in pixels/top-down with a Gauss-Newton wrist fit, the 3D rig drives a root
# spring + yaw/pitch + per-finger IK targets instead).
# Units are adapted: inches/model-units -> cm x hf (rest offsets scale like
# hand3d._REST_REF_HF).  All visual-only: IK targets stay reachable.
# Deliberately EXCLUDED from geom_hash() (see it): retuning motion must not
# invalidate the mesh cache.
# --------------------------------------------------------------------------- #
# 1a -- anticipation / strike timeline (hands STRIKE_T/RELEASE_T/LINGER_T/
# RETURN_T/PRESS_T, pianist finger_lead(0.5) = 0.9 s head start, 0.6 travel
# share).  AIM_LEAD (vkeyboard, 0.25 s) is the short end of that head start:
# the pump only knows imminent notes, so PREP_LEAD_T caps how early a finger
# rises to its striking height.
STRIKE_T = 0.12        # s final drop onto the key (cut short when time is
                       # short; quadratic ease, like _tip_target's s*s)
RELEASE_T = 0.12       # s fingertip lift after the release (the 2D pose
                       # eases the lift; here it is a one-pole rise, RISE_TAU)
LINGER_T = 0.06        # s a released finger stays over its key ...
RETURN_T = 0.35        # s ... then drifts back to its resting spot
RISE_TAU = 0.045       # s one-pole the tip rises toward a NEW target (the
                       # release lift / the hover->PREP raise, hands
                       # RELEASE_T ~3 tau).  Only a rising target is
                       # smoothed: a descending one (the strike drop, the key
                       # going down) is taken exactly, so contact timing and
                       # tip_err never lag.
PREP_LEAD_T = 0.9      # s longest head start (finger_lead(0.5)); beyond it
                       # the finger keeps its idle height
PREP_ADD = {1: 0.45, 2: 1.30, 3: 1.30, 4: 1.30, 5: 1.10}  # cm x hf above
                       # HOVER when raised, ready to strike (hands PREP-HOVER
                       # in model units x ~1.09 cm/unit)
# 1b -- wrist fit, joint limits first (hands WRIST_DEV_DEG/YAW_PRIOR,
# SPLAY_LIMIT_DEG/SPLAY_COMFORT_DEG, REACH_MIN/REACH_COMFORT, CROSS_TURN_DEG).
# The hand turns at the wrist first (fast tau: turn leads) and slides second
# (root spring): yaw anticipation is clamped to the wrist's deviation range.
WRIST_DEV_DEG = (-26.0, 8.0)  # wrist deviation range about the forearm's
                              # natural turn (hands.py, clockwise positive
                              # toward the little finger there)
YAW_PRIOR = 0.25       # how strongly the hand prefers the forearm's natural
                       # turn (hands _fit_hand prior; here: the shoulder-arm
                       # yaw in hand3d.update plays this role)
YAW_SPLAY_GAIN = 0.05  # rad of anticipation yaw per cm the root still has to
                       # travel (turn-before-slide; 0 at settle, reach-safe).
                       # 0.02 rad/cm read as 1 deg on a 12 cm jump (invisible);
                       # 0.05 gives ~3.5 deg there and hits the WRIST_DEV_DEG
                       # clamp (-26 deg) past ~9 cm of travel, so a big shift
                       # visibly turns before it slides.
YAW_SPLAY_TAU = 0.04   # s smoothing of that yaw (faster than the root TAU
                       # 0.065, so the turn visibly leads the slide)
SPLAY_LIMIT_DEG = {1: (-85, 34), 2: (-36, 14), 3: (-26, 26), 4: (-22, 24),
                   5: (-12, 55)}     # full stretch (hands.py, held/struck keys)
SPLAY_COMFORT_DEG = {1: (-64, 34), 2: (-24, 10), 3: (-18, 18), 4: (-15, 16),
                     5: (-8, 24)}    # comfortable spread in the air
REACH_MIN = {1: 0.55, 2: 0.32, 3: 0.32, 4: 0.32, 5: 0.36}  # min knuckle->tip
REACH_COMFORT = {1: 0.97, 2: 0.98, 3: 0.98, 4: 0.98, 5: 0.98}  # ... as a
                       # share of the finger's length (a playing finger is
                       # long, gently arched)
CROSS_TURN_DEG = 14.0  # extra turn toward the little finger while the thumb
                       # crosses under (hands CROSS_TURN_DEG; folded into the
                       # yaw-anticipation clamp, not a separate state)
# 1c -- tendon link + idle fan + gestures (hands TENDON_FOLLOWERS/
# TENDON_FLOOR_IN/SHAPE_FALLOFF/RUN_*/TREM_*/GESTURE_REPEAT_T/BOUNCE_MAX_IN,
# pianist tendon_link 0.6 / wrist_bounce 0.5 / tremolo_rotation 0.5).
TENDON_FOLLOWERS = {3: (4,), 4: (3,), 5: (3, 4)}  # a curving leader takes
                       # these with it (middle<->ring both ways, little takes
                       # both) -- the thumb/index stay independent
TENDON_LINK = 0.6      # pianist default: share of the leader's depth passed on
TENDON_FLOOR = 0.38    # cm above the key tops a follower never goes below
                       # (TENDON_FLOOR_IN 0.15 in)
SHAPE_FALLOFF = 0.6    # idle fingers follow a busy neighbour by this much
                       # per finger of distance (hands._shaped_rests)
SHAPE_CAP_X = 2.1      # cm x hf cap on the shaped-rest x displacement
SHAPE_CAP_Y = 1.2      # ... and y (hands 0.9/0.5 white-key caps in inches)
                       # -- reference only, NOT wired: the y cap needs a
                       # shaped-rest y, which this rig has no slot for (the
                       # idle IDLE_PULL_Y/_IDLE_FWD own the y). Do not
                       # fake-wire it (same rule as REST_MCP_DEG).
SHAPE_RAMP = (0.15, 0.6)  # cm x hf: a busy finger's pull ramps in over this
                       # |displacement| range -- inside a comfortable grip
                       # the natural slots already fan correctly (and the
                       # silhouette stays put for the look test); a
                       # stretched grip (octaves, wide chords) fans fully
RUN_GLIDE_TAU = 0.12   # s root smoothing inside a scale run.  hands
                       # RUN_GLIDE_T = 0.25 s is a +-window the wrist is
                       # *averaged* over (a ~0.5 s box, one-pole equivalent
                       # ~0.11 s); a literal 0.25 s tau made the wrist trail a
                       # 19 cm/s run by ~4.8 cm and dragged held fingertips
                       # off their keys (0.4 cm tip_err in a legato probe)
RUN_RAMP_T = 0.15      # s a run's hold eases in/out over
TREM_GAP_T = 0.3       # s alternations quicker than this rotate the forearm
TREM_ROLL_MAX = 0.07   # rad forearm rock in a tremolo (~4 deg: the 2D
                       # ROLL_MAX 40 deg at 50% reads as top-down swing and
                       # does not map 1-1 onto 3D roll; IK stays exact)
TREM_MIN_NOTES = 4     # alternating onsets needed before the roll engages
BOUNCE_REPEAT_T = 0.45  # s repeated chords closer than this bounce (hands
                       # GESTURE_REPEAT_T; the in-rig re-strike window stays
                       # 0.25 s) -- reference only, NOT wired: the bounce is
                       # not scaled with chord size (the rig has no chord-size
                       # signal); kept so the port stays traceable


# --------------------------------------------------------------------------- #
# Flexibility / behaviour (step 3): idea-port of pygame-piano-animator
# `pianist.py` BEHAVIORS (the technique sliders) and its per-bone anatomy
# ranges.  The 2D model drives a Gauss-Newton wrist fit that takes per-bone
# lengths as input; the 3D rig solves a fixed 18-bone skeleton, so what a
# slider can still move here is (a) how far each finger may fold and how its
# DIP follows its PIP, and (b) how high a finger waits above the keys, plus
# the idle curl.
#
# Every default is the value the rig ALREADY used (HandRig.BEND_MAX 1.75,
# HandRig.COUPLING = REST_DIP_RATIO, HandRig.HOVER 0.55 and hand3d's idle
# lift/pull tables), so the shipped hand stays bit-identical: these are wired,
# not aspirational.
#
# All visual-only and deliberately EXCLUDED from geom_hash(): how flexibly a
# finger folds is not a shape, and a slider drag must not re-mesh the hand.
# The geometry half of the source's anatomy -- the bone lengths themselves --
# lives in the anatomy block above, inside the hash.
# --------------------------------------------------------------------------- #
FLEX_BEND = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}     # x HandRig.BEND_MAX
                        # (the PIP flexion limit solve_chain bisects up to;
                        # below 1 the chain stops short of a full fold, which
                        # is what a stiff 4th/5th reads like)
FLEX_COUPLING = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}  # x HandRig.COUPLING
                        # (DIP = coupling * PIP; the thumb's 2-joint chain
                        # follows the same rule).  The cached REST mesh keeps
                        # using REST_DIP_RATIO directly (handmesh.rest_chains):
                        # a rest curl is geometry, so it must not move with a
                        # visual slider -- retune REST_DIP_RATIO for that.
HOVER_GAP = 0.55        # cm above the key top a waiting fingertip rides
                        # (hand3d HandRig.HOVER; a WORLD cm like KEY_SINK, not
                        # scaled with hand size)
HOVER_ADD = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}  # cm x hf per finger on
                        # top of HOVER_GAP (PREP_ADD is the raised
                        # striking-ready counterpart, step 1a); 0 everywhere =
                        # the shipped look, so a re-tuned value only changes
                        # how high that finger waits
IDLE_LIFT = {1: 0.4, 2: 0.9, 3: 1.1, 4: 1.0, 5: 0.8}  # cm x hf: the height
                        # an idle finger curls at (hand3d reads it live in update(); NOT to
                        # be confused with REST_LIFT_Z above -- that is the
                        # rest-pose target lift of the mesh, a different thing)
IDLE_PULL_Y = {1: 1.0, 2: 0.35, 3: 0.35, 4: 0.35, 5: 0.5}  # cm x hf: how
                        # far an idle tip retreats toward the wrist (the thumb
                        # needs the full pull -- its MCP sits far behind the
                        # fingers -- the others only a hint)
IDLE_PULL_Z = 0.6       # ... and rises with it, so the finger arches instead
                        # of splaying (read live by hand3d.update)

# Where the source's sliders landed (the number in brackets is the source's
# own default; "not ported" entries are deliberate, with the reason).
BEHAVIOR_PORT = {
    "retraction": "IDLE_PULL_Y / IDLE_PULL_Z (0.1): an idle tip pulls back and "
                  "up toward the palm; the thumb only half as much "
                  "(hands.py 2045).",
    "finger_curve": "IDLE_LIFT + FLEX_BEND (0.4): the idle curl height, and "
                    "how far the chain may fold to reach a key.",
    "antic_hand": "YAW_SPLAY_GAIN / TAU (0.5): the whole hand moves for the "
                  "notes coming up -- step 1b.",
    "antic_fingers": "PREP_LEAD_T / STRIKE_T (0.5): the per-finger head start "
                     "and the final drop -- step 1a.",
    "tendon_link": "TENDON_LINK / TENDON_FOLLOWERS (0.6) -- step 1c.",
    "cross_height": "SPLAY_LIMIT_DEG[1] + _COINCIDE_LIFT (0.33): not ported "
                    "separately -- the IK pole already keeps a crossing finger "
                    "arched over the thumb exactly.",
    "lift_height": "PREP_ADD / HOVER_ADD (0.5): how high a finger rises before "
                   "it strikes.",
    "wrist_bounce": "BOUNCE_LIFT / BOUNCE_TAU (0.5) -- step 1c.",
    "tremolo_rotation": "TREM_ROLL_MAX (0.5) -- step 1c.",
    "gesture_finger_action": "not ported (0.25): splitting a gesture between "
                             "wrist and fingers needs a chord-size signal the "
                             "rig does not get.",
    "cross_turn": "CROSS_TURN_DEG (14 deg) -- folded into the yaw clamp, "
                  "step 1b.",
    "smoothness": "HandRig.TAU (0.07 -> 0.065 s): the root spring tau.",
    "early_release": "LINGER_T (0.22 -> 0.06 s): visual only -- the audible "
                     "early release belongs to the audio scheduler, not the "
                     "rig.",
    "max_speed": "not ported (3.0 m/s): a fingertip speed cap would need the "
                 "audio scheduler's release times to decide when to let go.",
    "key_area_near/far": "not ported (0.0/1.0): keys are struck at their own "
                         "contact y (utils.CONTACT_Y_*).",
    "roll_speed": "not ported (0.035 s): rolling a too-wide chord is a "
                  "fingering choice, not a hand pose.",
    "wrist": "wrist_width() / ARM_PROFILE (1.0): the wrist half-width, hashed "
             "with the palm profile.",
    "mc*/pp*/mp*/dp*": "bone_lengths() / segment_ratio() / knuckle_row() over "
                       "FINGER_SPECS -- hashed geometry.",
}


def geom_hash():
    """8-char sha1 over geometry + SDF constants + grid + version (R1.4).

    NOTE: the hash covers GEOMETRY constants only. Fingering cost tables
    (MAX_SPAN, COMFORT_SPAN, CROSS_BASE, UNDER, BLACK_COST, WHITE_COST,
    THUMB_DOUBLE, PINKY_DOUBLE, INNER_ROOM_14) and look/motion constants
    (TEX_*, SKIN_*, STYLE_*, BOUNCE_*, WANDER_*, SYMPATHY_DIP, WHITE_RIDE,
    LOUD_LIFT) are deliberately EXCLUDED: retuning them must not invalidate
    the mesh cache -- a hand *style* is painted, never re-meshed, so the
    skin-style marker lives in the texture filename instead. The same holds
    for the step-3 anatomy view (bone_lengths() / segment_ratio() /
    knuckle_row() / wrist_width() -- pure DERIVATIONS of the hashed constants,
    which is why this step needed no new hash input) and for the step-3
    flexibility knobs (FLEX_BEND / FLEX_COUPLING / HOVER_GAP / HOVER_ADD /
    IDLE_LIFT / IDLE_PULL_Y / IDLE_PULL_Z): how a finger folds is not a shape.
    Geometry LOGIC changes (e.g. a rest_chains formula or coupling edit) are
    invisible to
    it -- bump HANDMESH_VERSION manually with every such change (v3 -> v4
    above was exactly this case: the thumb formula reuses unhashed
    THUMB_BASE_Y0 / THUMB_FWD inputs differently, and the coupling change
    reuses the unchanged REST_DIP_RATIO value, so the hash alone rebuilt
    nothing).
    """
    rep = repr((FINGER_SPECS, FING_R_SCALE, FING_R_TAPER, MCP_SHRINK, ARM_PROFILE,
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
