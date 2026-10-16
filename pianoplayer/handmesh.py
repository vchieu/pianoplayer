"""Unified skinned hand mesh (Tier B, SDF v2).

One smooth-unioned mesh per hand, built once from a signed-distance field
(SDF) in the wrist-local rest pose and cached in the user cache dir::

    origin : wrist centre
    +y     : toward the fingers
    +x     : toward the pinky (right hand; mirrored for the left)
    +z     : up (dorsal side)

The rest pose (R2.6) is the average playing pose -- fingers closed, slight
curl solved with the existing ``solve_chain`` IK, no rig required.
Primitives are forearm/palm ellipsoids from the ``ARM_PROFILE`` stations, a
single thenar pair + small hypothenar, 4 MCP mounds, per-finger 3-segment
elliptical capsules (flatter dorsal, fuller pad) with PIP/DIP mounds and a
flattened pad tip, and a thumb CMC capsule.  Combination uses ``smin``
(``-k*logaddexp``) inside a finger / finger<->palm, y-dependent webbing
between neighbours (smin inside the web band, plain ``min()`` outside so
neighbours touch but never fuse), and plain ``min()`` finger<->finger
outside the band (R2.4).

Per-frame skinning (in ``hand3d``) reuses the already-solved IK ``q``
chains: per bone ``R_b = [X T Z] @ [X0 T0 Z0].T``,
``v' = P_b + R_b @ (v - P0_b)`` (linear blend skinning, vectorised).

All geometry constants live in ``pianoplayer.handspec``; the cache filename
carries their hash so edits rebuild automatically (R1.4).

No new dependencies: numpy + PIL + vtk/vedo only (no scipy).
"""

from __future__ import annotations

import os

import numpy as np

from pianoplayer import handspec as H

HANDMESH_VERSION = H.HANDMESH_VERSION
# Deviation (H6, plan-sanctioned fallback): GRID_STEP 0.12 -> 0.15. Timing
# evidence (size M, same machine): 0.12 SDF alone 5.4-8.1 s (total build
# 7-10 s incl. decimation); 0.15 SDF ~3.0-3.4 s, total ~4.5 s. Decimation to
# <=8000 tris dominates the final surface detail either way, so 0.15 loses
# nothing visible.
GRID_STEP = H.GRID_STEP
GRID_Y0 = H.GRID_Y0     # fixed proximal border: the stub pole (~-23.5) always
                        # closes inside it (app frame bottom is -18, so the
                        # cap never shows)
TEX_NAME = "handtex_v2.png"
TRIS_MAX = H.TRIS_MAX

_K_INTRA = H.K_INTRA  # within one finger: near-hard union
_K_PALM = H.K_PALM    # finger <-> palm/thenar: webbing bridges
_K_CMC = H.K_CMC      # thumb CMC <-> palm
_K_BODY = H.K_BODY    # forearm/palm ellipsoid chain


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def cache_dir():
    """User cache dir for meshes + textures (created on demand)."""
    d = os.path.join(os.path.expanduser("~"), ".cache", "pianoplayer")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def cache_path(hf, side):
    """Hashed cache file (R1.4): geometry/SDF/grid/version changes rebuild."""
    s = "left" if str(side).lower().startswith("l") else "right"
    return os.path.join(cache_dir(),
                        "handmesh_%s_%.2f_%s.npz" % (s, float(hf),
                                                    H.geom_hash()))


def _load_cache(path):
    try:
        if os.path.isfile(path):
            return np.load(path, allow_pickle=False)
    except Exception:
        pass
    return None


def texture_path(hf=None, side=None, size=None):
    """Per-(hf, side) texture (R1.5); bare call returns the legacy file.

    `size` selects the resolution variant (512 default, 1024 via
    PIANOPLAYER_TEXSIZE); each size has its own cache file carrying
    TEX_VERSION, so look upgrades rebuild once and then load fast."""
    if hf is None or side is None:
        return os.path.join(cache_dir(), TEX_NAME)
    s = "left" if str(side).lower().startswith("l") else "right"
    sz = int(size or H.TEX_SIZE_DEFAULT)
    return os.path.join(cache_dir(),
                        "handtex_%s_%.2f_%s_%s%d.png"
                        % (s, float(hf), H.geom_hash(), H.TEX_VERSION, sz))


def _cache_hash_of(fn):
    """Hash embedded in a cache filename, or None for legacy names.

    Current mesh names end in _<hash8>; current texture names carry
    _<hash8>_<TEX_VERSION><size> (see cache_path/texture_path); anything
    else (pre-R1.4 version-suffixed meshes, the bare handtex_v2.png, and
    pre-v3 unversioned handtex_<side>_<hf>_<hash>.png) is legacy and
    therefore stale by definition (BUG-4)."""
    stem = fn.rsplit(".", 1)[0]
    tail = stem.rsplit("_", 1)[-1]
    if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
        return tail
    # versioned texture: handtex_<side>_<hf>_<hash8>_v<N><size>
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and len(parts[1]) == 8 and \
            all(c in "0123456789abcdef" for c in parts[1]):
        return parts[1]
    return None


def clear_cache(prune_only=False):
    """Remove cached meshes + textures (R1.4 `--clear-cache`).

    Default clears all (unchanged); prune_only=True (`--prune-cache`)
    deletes only stale files -- hashed names whose hash != current
    geom_hash() plus legacy unhashed names -- and keeps current caches.
    Texture files must also carry the current TEX_VERSION marker
    (pre-v3 unversioned handtex files are stale look, orphaned by the
    versioned texture_path).
    """
    n = 0
    try:
        d = cache_dir()
        cur = H.geom_hash()
        for fn in os.listdir(d):
            if (fn.startswith("handmesh_") and fn.endswith(".npz")) or \
                    (fn.startswith("handtex_") and fn.endswith(".png")):
                if prune_only and _cache_hash_of(fn) == cur and \
                        (fn.startswith("handmesh_") or
                         ("_%s" % H.TEX_VERSION) in fn):
                    continue  # current geometry + look: keep
                try:
                    os.remove(os.path.join(d, fn))
                    n += 1
                except Exception:
                    pass
    except Exception:
        pass
    return n


# --------------------------------------------------------------------------- #
# SDF helpers (pure numpy)
# --------------------------------------------------------------------------- #
def _smin(a, b, k):
    a = np.asarray(a)
    b = np.asarray(b)
    return -k * np.logaddexp(-a / k, -b / k)


def _seg_proj(p, a, b):
    """Distance of points `p` (N,3) to segment a->b; returns (d, t, closest)."""
    dt = np.asarray(p).dtype
    a = np.asarray(a, dtype=dt)
    b = np.asarray(b, dtype=dt)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        d = np.linalg.norm(p - a[None, :], axis=1)
        return d, np.zeros(p.shape[0]), np.repeat(a[None, :], p.shape[0],
                                                  axis=0)
    t = np.clip((p - a[None, :]) @ ab / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    d = np.linalg.norm(p - closest, axis=1)
    return d, t, closest


def _capsule(p, a, b, r0, r1):
    d, t, _ = _seg_proj(p, a, b)
    return d - (float(r0) + (float(r1) - float(r0)) * t)


def _capsule_ell(p, a, b, r0, r1, az):
    """R2.3 elliptical capsule: x/y radius r, z radius az*r (flatter back).

    The offset from the segment is measured with z compressed, so the
    zero level sits at az*r above/below the axis instead of r.
    """
    dt = np.asarray(p).dtype
    a = np.asarray(a, dtype=dt)
    b = np.asarray(b, dtype=dt)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        off = (p - a[None, :]) * np.array([1.0, 1.0, 1.0 / az], dtype=dt)
        return np.linalg.norm(off, axis=1) - float(r0)
    t = np.clip((p - a[None, :]) @ ab / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    off = (p - closest) * np.array([1.0, 1.0, 1.0 / max(float(az), 0.3)],
                                   dtype=dt)
    d = np.linalg.norm(off, axis=1)
    return d - (float(r0) + (float(r1) - float(r0)) * t)


def _smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blend(w, s, m):
    """Blend fields `s` (weight w) and `m` (weight 1-w), NaN-safe.

    Plain `w*s + (1-w)*m` yields 0*inf = NaN where w hits exactly 0/1
    while a field is +inf (far from every primitive); select instead.
    Single precision throughout: the grid is float32 and the blend needs
    no more (saves ~1/3 of the combine traffic).
    """
    w = np.asarray(w, np.float32)
    s = np.asarray(s, np.float32)
    out = np.array(m, dtype=np.float32, copy=True)
    mid = (w > 0) & (w < 1)
    out[mid] = w[mid] * s[mid] + (1.0 - w[mid]) * out[mid]
    out[w >= 1] = s[w >= 1]
    return out


def _ellipsoid(p, c, r):
    """Approximate SDF of an axis-aligned ellipsoid (stable, no scipy)."""
    dt = np.asarray(p).dtype
    c = np.asarray(c, dtype=dt)
    r = np.asarray(r, dtype=dt)
    r = np.maximum(r, 1e-6)
    q = (p - c[None, :]) / r[None, :]
    k0 = np.linalg.norm(q, axis=1)
    denom = np.linalg.norm((p - c[None, :]) / (r[None, :] ** 2), axis=1)
    denom = np.maximum(denom, 1e-9)
    out = k0 * (k0 - 1.0) / denom
    # deep inside: fall back to the scaled radius so the field stays finite
    out = np.where(k0 < 1e-9, -float(np.min(r)), out)
    return out


# --------------------------------------------------------------------------- #
# rest pose (F1) + bones
# --------------------------------------------------------------------------- #
def rest_chains(hf):
    """R2.6 rest joints: {finger: q (4,3)} + thumb CMC start, right-hand local.

    The average playing pose: fingers closed (REST_SPREAD), slight curl
    (REST_REACH_FRAC of the chain length, small lift), solved with the
    existing solve_chain -- no rig required.
    """
    from pianoplayer.hand3d import solve_chain
    hf = float(hf)
    chains = {}
    for f in H.FINGERS:
        spec = H.FINGER_SPECS[f]
        bx, by, bz = H.mcp_base(f)
        if f == 1:
            # thumb MCP in hf=1 units (same convention as the rig:
            # HandRig._thumb_by = thumb_base_y(hf), scaled by hf in _L
            # below); must match the rig exactly (reach-tested authority).
            by = H.thumb_base_y(hf)
        base = np.array([bx * hf, by * hf, bz * hf])
        lens = [l * hf for l in spec["lens"]]
        L = sum(lens)
        spread = (f - 3) * H.REST_SPREAD * hf
        target = base + np.array([spread, L * H.REST_REACH_FRAC,
                                  H.REST_LIFT_Z * hf])
        pole = np.array(spec["pole"], float)
        # R2.6: same DIP coupling as the rig (HandRig.COUPLING ==
        # REST_DIP_RATIO), so the rest curl matches the playing curl.
        pts, _ = solve_chain(base, target, lens, pole,
                             coupling=H.REST_DIP_RATIO)
        chains[f] = np.array(pts, float)
    cmc_a = np.array(H.THUMB_CMC_A) * hf
    return chains, cmc_a


def rest_bones(hf, side="right"):
    """Bone segments (a, b) in side-local coords.

    Order (B=18): forearm, palm, thumb-CMC, then 3 per finger (f1..f5).
    The first three are rigid with the palm (same root/R transform); the
    15 finger bones articulate with the IK chains.
    """
    hf = float(hf)
    s = -1.0 if str(side).lower().startswith("l") else 1.0
    chains, cmc_a = rest_chains(hf)
    if s < 0:
        cmc_a = cmc_a * np.array([-1.0, 1.0, 1.0])
    bones = []
    names = []
    # forearm: far arm (out of frame) -> wrist
    bones.append((np.array([0.0, -20.0, 0.0]),
                  np.array([0.0, -1.8 * hf, 0.2 * hf])))
    names.append("forearm")
    # palm: wrist -> knuckle line
    bones.append((np.array([0.0, -1.8 * hf, 0.2 * hf]),
                  np.array([1.0 * s * hf, 8.2 * hf, 1.0 * hf])))
    names.append("palm")
    # thumb CMC: thenar -> thumb MCP
    q1 = chains[1][0].copy()
    if s < 0:
        q1 = q1 * np.array([-1.0, 1.0, 1.0])
    bones.append((cmc_a, q1))
    names.append("f1_cmc")
    for f in (1, 2, 3, 4, 5):
        q = chains[f]
        if s < 0:
            q = q * np.array([-1.0, 1.0, 1.0])
        for i in range(3):
            bones.append((np.array(q[i]), np.array(q[i + 1])))
            names.append("f%d_%d" % (f, i))
    return bones, names


def _bone_frames(dirs):
    """Rest/current bone rotation columns: R[b] = [X T Z] (B,3,3).

    Same construction as hand3d._frames (X ~ world +x, Z = X x T).
    """
    from pianoplayer.hand3d import _frames
    T = np.array(dirs, float)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.maximum(n, 1e-9)
    X, Z = _frames(T)
    R = np.stack([X, T, Z], axis=2)
    return R


def rest_frames(hf, side="right"):
    """Per-bone rest (P0, R0): rigid bones get P0=0/R0=I (F3)."""
    bones, names = rest_bones(hf, side)
    B = len(bones)
    P0 = np.zeros((B, 3))
    R0 = np.repeat(np.eye(3)[None], B, axis=0)
    dirs = []
    for (a, b) in bones:
        d = np.array(b, float) - np.array(a, float)
        n = np.linalg.norm(d)
        dirs.append(d / max(n, 1e-9))
    Rf = _bone_frames(np.array(dirs))
    for i, nm in enumerate(names):
        if nm in ("forearm", "palm", "f1_cmc"):
            continue  # rigid: P0=0, R0=I
        P0[i] = np.array(bones[i][0], float)
        R0[i] = Rf[i]
    return bones, names, P0, R0


# --------------------------------------------------------------------------- #
# SDF field
# --------------------------------------------------------------------------- #
def _body_ellipsoids(hf):
    """Forearm->knuckle ellipsoid chain from ARM_PROFILE + mounds (R2.1/2).

    The two farthest stations are replaced by five unscaled forearm-stub
    ellipsoids (the gap-based ry of the -30 station would otherwise poke
    through any reasonable grid border and leave the mesh open).  The chain
    ends in a rounded pole at y~-23.5, safely inside the grid (app frame
    bottom is -18, so the cap never shows).  Station ry is capped at 4.0;
    distal gaps (~2-3 cm) never hit the cap.

    R1.1: the thenar/hypothenar eminences are small (no outer-edge ears);
    the thenar is a single pair along the metacarpal (~35 deg); 4 MCP
    mounds round the knuckle line (R2.2).
    """
    hf = float(hf)
    prof = H.ARM_PROFILE
    n = len(prof)
    ys = np.array([(y if k < H.ARM_PIN else y * hf) for k, (y, *_r) in
                   enumerate(prof)])
    ell = []
    for k, (y, c, a1, a2, _p) in enumerate(prof):
        if k < H.ARM_PIN:
            continue  # far out-of-frame stations: replaced by stubs below
        gaps = []
        if k > 0:
            gaps.append(ys[k] - ys[k - 1])
        if k + 1 < n:
            gaps.append(ys[k + 1] - ys[k])
        ry = min(0.75 * max(gaps) if gaps else 3.0, 4.0)
        ell.append((np.array([c * hf, ys[k], 0.0]),
                    np.array([a1 * hf, max(float(ry), 1.2 * hf),
                              a2 * hf])))
    # unscaled forearm stubs: constant slim tube ending rounded at y ~ -23.5
    for sy in (-21.5, -18.0, -14.5, -11.0, -7.5):
        ell.append((np.array([0.0, sy, 0.0]),
                    np.array([3.3 * hf, 2.0, 2.3 * hf])))
    # thenar pair along the metacarpal + small hypothenar (R1.1/R2.5).
    # All mounds stay inside the loft silhouette (outer-edge defect: the
    # +x extent must not exceed the profile by more than ~0.3 cm).
    ell.append((np.array([-2.6 * hf, 1.2 * hf, 0.4 * hf]),
                np.array([1.2 * hf, 2.0 * hf, 1.0 * hf])))
    ell.append((np.array([-3.3 * hf, 3.2 * hf, 0.5 * hf]),
                np.array([0.8 * hf, 1.5 * hf, 0.9 * hf])))
    ell.append((np.array([3.0 * hf, 4.2 * hf, 0.3 * hf]),
                np.array([0.7 * hf, 1.5 * hf, 0.8 * hf])))
    # MCP mounds (R2.2): the knuckle line reads as 4 bumps, not a bar
    spac = H.mcp_spacing()
    for f, sp in zip((2, 3, 4, 5), [spac[0]] + spac):
        bx, by, bz = H.mcp_base(f)
        ell.append((np.array([bx * hf, (by - 0.3) * hf, bz * hf]),
                    np.array([H.MCP_MOUND_R * sp * hf, 0.9 * hf,
                              0.9 * hf])))
    return ell


def _finger_capsules(hf, chains):
    """R2.3 fingers: 3 elliptical capsules + PIP/DIP mounds + pad tip.

    Each segment is an elliptical section (x/y radius r, z radius az*r,
    flatter on the back); the PIP mound and a slight DIP mound read as
    knuckles; the fingertip is a flattened pad ellipsoid, not a hemisphere.
    Radii come from handspec (R1.2: no inflation); the MCP radius is capped
    at half the local MCP spacing so neighbours can never overlap.
    """
    hf = float(hf)
    spac = H.mcp_spacing()
    adj = {2: min(spac[0], spac[1]), 3: min(spac[0], spac[1]),
           4: min(spac[1], spac[2]), 5: spac[2], 1: 2.0}
    out = {}
    for f in H.FINGERS:
        q = np.asarray(chains[f], float)
        R = [H.finger_radius(f, j) * hf for j in range(4)]
        R[0] = min(R[0], 0.5 * adj[f] * hf)
        az0, az1, az2 = H.FING_Z_FLAT
        segs = [(q[0], q[1], R[0], R[1], az0),
                (q[1], q[2], R[1], R[2], az1),
                (q[2], q[3], R[2], R[3] * 0.94, az2)]
        spheres = [(q[0] + 0.13 * (q[1] - q[0]),
                    R[0] * 1.10),                            # MCP knuckle
                   (q[1], max(R[1], R[2]) * H.PIP_MOUND),     # PIP mound
                   (q[2], max(R[2], R[3]) * H.DIP_MOUND)]    # DIP hint
        d = q[3] - q[2]
        n = float(np.linalg.norm(d)) or 1.0
        tip = q[3] - 0.15 * R[3] * d / n
        pads = [(tip, np.array([R[3], 1.35 * R[3],
                                H.TIP_PAD_FLAT * R[3]]))]
        out[f] = (segs, spheres, pads)
    return out


# Baked AO parameters
_AO_EPS = 0.3      # cm offset along normal for AO sampling
_AO_SCALE = 0.5    # scale for AO contrast (tuned so max raw ~0.47 maps to ~1.0)
_AO_CLAMP_LO = 0.75
_AO_CLAMP_HI = 1.0


def _eval_sdf_at_points(pts, hf):
    """Evaluate the hand SDF at arbitrary points (not on grid).
    
    Uses the same primitive combination as _build_sdf but evaluates
    directly at the given points. Returns SDF values (negative = inside).
    """
    from pianoplayer import handspec as H
    hf = float(hf)
    chains, cmc_a = rest_chains(hf)
    body_ell = _body_ellipsoids(hf)
    raw_caps = _finger_capsules(hf, chains)
    fing_caps = {}
    for f, (segs, spheres, pads) in raw_caps.items():
        fing_caps[f] = (segs, spheres, pads)
    
    # Evaluate all primitives at the query points
    pts = np.asarray(pts, np.float32)
    N = pts.shape[0]
    INF = np.full(N, np.inf, np.float32)
    
    # Body ellipsoids
    acc_body = INF.copy()
    for (c, r) in body_ell:
        c = np.asarray(c, np.float32)
        r = np.asarray(r, np.float32)
        d = _ellipsoid(pts, c, r)
        acc_body = _smin(acc_body, d, _K_BODY)
    
    # Finger capsules/spheres/pads
    acc_f = {}
    for f in H.FINGERS:
        acc = INF.copy()
        segs, spheres, pads = fing_caps[f]
        for (a, b, r0, r1, az) in segs:
            a = np.asarray(a, np.float32)
            b = np.asarray(b, np.float32)
            d = _capsule_ell(pts, a, b, float(r0), float(r1), float(az))
            acc = _smin(acc, d, _K_INTRA)
        for (c, r) in spheres:
            c = np.asarray(c, np.float32)
            d = np.linalg.norm(pts - c[None, :], axis=1).astype(np.float32) - np.float32(r)
            acc = _smin(acc, d, _K_INTRA)
        for (c, r) in pads:
            c = np.asarray(c, np.float32)
            r = np.asarray(r, np.float32)
            d = _ellipsoid(pts, c, r)
            acc = _smin(acc, d, _K_INTRA)
        acc_f[f] = acc
    
    # Thumb CMC
    q1 = chains[1][0]
    r_cmc = 1.15 * hf
    r_mcp1 = H.finger_radius(1, 0) * hf
    acc_cmc = _capsule(pts, cmc_a, q1, r_cmc, r_mcp1)
    
    # Webbing
    y_kn = 8.2 * hf
    w_web = _smoothstep(y_kn + H.WEB_Y_LO * hf - 0.5 * hf,
                        y_kn + H.WEB_Y_LO * hf, pts[:, 1]) * \
            (1.0 - _smoothstep(y_kn + H.WEB_Y_HI * hf,
                               y_kn + H.WEB_Y_HI * hf + 0.5 * hf, pts[:, 1]))
    d2345_min = np.minimum(np.minimum(acc_f[2], acc_f[3]),
                           np.minimum(acc_f[4], acc_f[5]))
    d2345_s = _smin(_smin(acc_f[2], acc_f[3], H.WEB_K),
                    _smin(acc_f[4], acc_f[5], H.WEB_K), H.WEB_K)
    d_2345 = _blend(w_web, d2345_s, d2345_min)
    
    # Thumb-index web
    w_th = _smoothstep(2.0 * hf, 3.5 * hf, pts[:, 1]) * \
        (1.0 - _smoothstep(8.0 * hf, 9.5 * hf, pts[:, 1]))
    d1i_s = _smin(acc_f[1], d2345_min, H.THUMB_WEB_K)
    d_1 = _blend(w_th, d1i_s, np.minimum(acc_f[1], d2345_min))
    
    # Combine all
    d_hand = _smin(_smin(acc_body, acc_cmc, _K_CMC), d_2345, _K_PALM)
    d_hand = _smin(d_hand, d_1, _K_PALM)
    
    return d_hand.astype(np.float32)


def _compute_baked_ao(pts, nrm, hf):
    """Compute baked ambient occlusion for mesh vertices.
    
    Samples SDF at v + eps*n and v - eps*n along normals.
    Note: mesh normals point INWARD (see _tube_faces), so:
    - pts_out = v - eps*n goes OUTSIDE (d > 0)
    - pts_in = v + eps*n goes INSIDE (d < 0)
    SDF is negative inside, positive outside.
    ao = clamp((d_out - d_in) / scale, 0.75, 1.0)
    Returns per-vertex AO factor in [0.75, 1.0].
    """
    eps = _AO_EPS
    # Outside (opposite normal, since normals point inward)
    pts_out = pts - eps * nrm
    # Inside (along normal)
    pts_in = pts + eps * nrm
    
    d_out = _eval_sdf_at_points(pts_out, hf)
    d_in = _eval_sdf_at_points(pts_in, hf)
    
    # AO factor: higher = more occluded (darker)
    # d_out (outside) > 0, d_in (inside) < 0 -> d_out - d_in = large positive
    ao = (d_out - d_in) / _AO_SCALE
    ao = np.clip(ao, _AO_CLAMP_LO, _AO_CLAMP_HI)
    return ao.astype(np.float32)


def _build_sdf(hf):
    chains, cmc_a = rest_chains(hf)
    f32 = lambda a: np.asarray(a, np.float32)
    chains = {f: f32(q) for f, q in chains.items()}
    cmc_a = f32(cmc_a)
    body_ell = [(f32(c), f32(r)) for (c, r) in _body_ellipsoids(hf)]
    raw_caps = _finger_capsules(hf, {f: np.asarray(q, float) for f, q in
                                     chains.items()})
    fing_caps = {}
    for f, (segs, spheres, pads) in raw_caps.items():
        fing_caps[f] = ([(f32(a), f32(b), r0, r1, az) for (a, b, r0, r1, az)
                         in segs],
                        [(f32(c), float(r)) for (c, r) in spheres],
                        [(f32(c), f32(r)) for (c, r) in pads])
    # Deviation (H6): adaptive y extent. The plan's fixed y[-20,12] clips the
    # F1 rest fingertips (chains reach y~14 at M, ~21 at XXL). Per-hf extent
    # (max rest tip y + 2.0 margin for radii + smoothing) is exact and cheap.
    # The proximal border is fixed (GRID_Y0): the stub pole (~-23.5) always
    # closes inside it. x/z likewise adapt to the primitive extents (+1.2
    # margin + max finger radius), so small hands build small grids.
    tipmax = max(float(q[:, 1].max()) for q in chains.values())
    y1 = max(tipmax + 2.0, GRID_Y0 + 4.0)
    ys = np.arange(GRID_Y0, y1 + 1e-9, GRID_STEP)
    xlo = min([float(np.min([q[:, 0].min() for q in chains.values()]))] +
              [float(c[0] - r[0]) for (c, r) in body_ell]) - 1.0
    xhi = max([float(np.max([q[:, 0].max() for q in chains.values()]))] +
              [float(c[0] + r[0]) for (c, r) in body_ell]) + 1.0
    zlo = min([float(c[2] - r[2]) for (c, r) in body_ell]) - 1.0
    zhi = max([float(np.max([q[:, 2].max() for q in chains.values()]))] +
              [float(c[2] + r[2]) for (c, r) in body_ell]) + 1.0
    # finger radii extend past the joint positions themselves
    rmax = max(H.finger_radius(f, 0) * float(hf) for f in H.FINGERS)
    xlo -= rmax
    xhi += rmax
    zhi += rmax
    xs = np.arange(xlo, xhi + 1e-9, GRID_STEP)
    zs = np.arange(zlo, zhi + 1e-9, GRID_STEP)
    nx, ny, nz = xs.size, ys.size, zs.size
    # Slice-major flat coordinates (idx = (j*nx+i)*nz+k): one boolean mask
    # per primitive selects exactly the slices the per-slice culling would
    # keep, so the field is bit-identical to the slice loop while each
    # primitive is evaluated in a single big numpy call.
    X = np.tile(np.repeat(xs, nz), ny).astype(np.float32)
    Y = np.repeat(ys, nx * nz).astype(np.float32)
    Z = np.tile(zs, ny * nx).astype(np.float32)
    INF = np.full(nx * ny * nz, np.inf, np.float32)

    def _block(mask):
        return np.stack([X[mask], Y[mask], Z[mask]], axis=1)

    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        acc_body = INF.copy()
        for (c, r) in body_ell:
            m = (Y >= c[1] - r[1] - 2.0) & (Y <= c[1] + r[1] + 2.0)
            if not m.any():
                continue
            d = _ellipsoid(_block(m), c, r)
            acc_body[m] = _smin(acc_body[m], d, _K_BODY)
        acc_f = {}
        for f, (segs, spheres, pads) in fing_caps.items():
            acc = INF.copy()
            for (a, b, r0, r1, az) in segs:
                rm = max(float(r0), float(r1))
                m = (Y >= min(a[1], b[1]) - rm - 0.5) & \
                    (Y <= max(a[1], b[1]) + rm + 0.5)
                if not m.any():
                    continue
                acc[m] = _smin(acc[m],
                               _capsule_ell(_block(m), a, b, r0, r1, az),
                               _K_INTRA)
            for (c, r) in spheres:
                m = (Y >= c[1] - r - 0.5) & (Y <= c[1] + r + 0.5)
                if not m.any():
                    continue
                d = np.linalg.norm(_block(m) - c[None, :], axis=1).astype(
                    np.float32) - np.float32(r)
                acc[m] = _smin(acc[m], d, _K_INTRA)
            for (c, r) in pads:
                m = (Y >= c[1] - r[1] - 0.5) & (Y <= c[1] + r[1] + 0.5)
                if not m.any():
                    continue
                acc[m] = _smin(acc[m], _ellipsoid(_block(m), c, r),
                               _K_INTRA)
            acc_f[f] = acc
        q1 = chains[1][0]
        # thumb CMC capsule (radii from the metacarpal scale, no inflation)
        r_cmc = 1.15 * hf
        r_mcp1 = H.finger_radius(1, 0) * hf
        rm_cmc = max(float(r_cmc), float(r_mcp1))
        m_cmc = (Y >= min(cmc_a[1], q1[1]) - rm_cmc - 0.5) & \
            (Y <= max(cmc_a[1], q1[1]) + rm_cmc + 0.5)
        acc_cmc = INF.copy()
        if m_cmc.any():
            acc_cmc[m_cmc] = _capsule(_block(m_cmc), cmc_a, q1,
                                      r_cmc, r_mcp1)
        # R2.4 y-dependent webbing: smin inside the web band (bridges the
        # 0-0.15 cm proximal gaps), plain min outside (tips stay separate).
        y_kn = 8.2 * hf
        w_web = _smoothstep(y_kn + H.WEB_Y_LO * hf - 0.5 * hf,
                            y_kn + H.WEB_Y_LO * hf, Y) * \
            (1.0 - _smoothstep(y_kn + H.WEB_Y_HI * hf,
                               y_kn + H.WEB_Y_HI * hf + 0.5 * hf, Y))
        d2345_min = np.minimum(np.minimum(acc_f[2], acc_f[3]),
                               np.minimum(acc_f[4], acc_f[5]))
        d2345_s = _smin(_smin(acc_f[2], acc_f[3], H.WEB_K),
                        _smin(acc_f[4], acc_f[5], H.WEB_K), H.WEB_K)
        # NOTE: _blend(), not arithmetic: w*s+(1-w)*m with w in {0,1}
        # and a +inf field is 0*inf = NaN far from every primitive.
        d_2345 = _blend(w_web, d2345_s, d2345_min)
        # thumb<->index web: wider band, larger k (R2.4)
        w_th = _smoothstep(2.0 * hf, 3.5 * hf, Y) * \
            (1.0 - _smoothstep(8.0 * hf, 9.5 * hf, Y))
        d1i_s = _smin(acc_f[1], d2345_min, H.THUMB_WEB_K)
        d_1 = _blend(w_th, d1i_s, np.minimum(acc_f[1], d2345_min))
        d_hand = _smin(_smin(acc_body, acc_cmc, _K_CMC),
                       d_2345, _K_PALM)
        d_hand = _smin(d_hand, d_1, _K_PALM)
    vol = d_hand.reshape(ny, nx, nz).transpose(1, 0, 2).astype(np.float32)
    return vol, (xs, ys, zs)


# --------------------------------------------------------------------------- #
# mesh extraction + cache
# --------------------------------------------------------------------------- #
def _pd_to_tris(pd):
    import vtkmodules.util.numpy_support as _ns
    polys = pd.GetPolys()
    data = _ns.vtk_to_numpy(polys.GetData())
    if len(data) and data[0] == 3 and (data[0::4] == 3).all():
        return np.ascontiguousarray(data.reshape(-1, 4)[:, 1:].astype(
            np.int64))
    tris = []
    i = 0
    while i < len(data):
        m = int(data[i])
        cell = [int(v) for v in data[i + 1:i + 1 + m]]
        if m == 3:
            tris.append(cell)
        elif m == 4:
            tris.extend(([cell[0], cell[1], cell[2]],
                         [cell[0], cell[2], cell[3]]))
        elif m > 4:
            for k in range(1, m - 1):
                tris.append([cell[0], cell[k], cell[k + 1]])
        i += 1 + m
    return np.array(tris, np.int64)


def _extract_mesh(vol, grid):
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import (
        vtkPolyDataNormals,
        vtkQuadricDecimation,
        vtkWindowedSincPolyDataFilter,
    )
    try:
        from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D as _FE
    except Exception:  # pragma: no cover - older VTK keeps the serial filter
        from vtkmodules.vtkFiltersCore import vtkMarchingCubes as _FE
    xs, ys, zs = grid
    img = vtkImageData()
    img.SetDimensions(xs.size, ys.size, zs.size)
    img.SetOrigin(xs[0], ys[0], zs[0])
    img.SetSpacing(GRID_STEP, GRID_STEP, GRID_STEP)
    import vtkmodules.util.numpy_support as _ns
    arr = _ns.numpy_to_vtk(vol.ravel(order="F"), deep=1)
    arr.SetName("sdf")
    img.GetPointData().SetScalars(arr)
    fe = _FE()
    fe.SetInputData(img)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOff()
    fe.Update()
    sm = vtkWindowedSincPolyDataFilter()
    sm.SetInputConnection(fe.GetOutputPort())
    sm.SetNumberOfIterations(15)
    sm.NormalizeCoordinatesOn()
    sm.Update()
    ntri = sm.GetOutput().GetNumberOfPolys()
    if ntri > TRIS_MAX:
        dec = vtkQuadricDecimation()
        dec.SetInputConnection(sm.GetOutputPort())
        dec.SetTargetReduction(1.0 - TRIS_MAX / float(ntri))
        dec.Update()
        pd = dec.GetOutput()
    else:
        pd = sm.GetOutput()
    nr = vtkPolyDataNormals()
    nr.SetInputData(pd)
    nr.ComputePointNormalsOn()
    nr.ComputeCellNormalsOff()
    nr.SplittingOff()
    nr.Update()
    pd = nr.GetOutput()
    import vtkmodules.util.numpy_support as _ns2
    pts = np.array(_ns2.vtk_to_numpy(pd.GetPoints().GetData()), float)
    nrm = np.array(_ns2.vtk_to_numpy(
        pd.GetPointData().GetNormals()), float)
    tris = _pd_to_tris(pd)
    return pts, tris, nrm


def build_rest_mesh(hf, side="right"):
    """Build (or load) the rest mesh; returns (pts, faces, nrm).

    Right-hand local SDF; mirrored x + reversed faces for the left hand
    (same convention as hand3d._arm_surface).  Cached under the hashed
    name (R1.4); pre-R1.4 files are never loaded (no hash to verify).
    """
    s = "left" if str(side).lower().startswith("l") else "right"
    path = cache_path(hf, s)
    z = _load_cache(path)
    if z is not None:
        try:
            if {"rest_pts", "faces", "rest_nrm"} <= set(z.files):
                return (np.array(z["rest_pts"], float),
                        [tuple(int(v) for v in row)
                         for row in np.array(z["faces"], np.int64)],
                        np.array(z["rest_nrm"], float))
        except Exception:
            pass
    print("building hand mesh (first run, size %.2f %s)..." % (float(hf), s),
          flush=True)
    vol, grid = _build_sdf(float(hf))
    pts, tris, nrm = _extract_mesh(vol, grid)
    faces = [tuple(int(v) for v in row) for row in tris]
    if s == "left":
        pts = pts * np.array([-1.0, 1.0, 1.0])
        nrm = nrm * np.array([-1.0, 1.0, 1.0])
        faces = [tuple(reversed(f)) for f in faces]
    # R4: baked AO - sample SDF along normals
    ao = _compute_baked_ao(np.asarray(pts, float), np.asarray(nrm, float), float(hf))
    try:
        np.savez_compressed(path, rest_pts=np.asarray(pts, np.float32),
                            faces=np.asarray(faces, np.int64),
                            rest_nrm=np.asarray(nrm, np.float32),
                            rest_ao=np.asarray(ao, np.float32))
    except Exception:
        pass
    return pts, faces, nrm


# --------------------------------------------------------------------------- #
# skinning weights (H7)
# --------------------------------------------------------------------------- #
def compute_weights(rest_pts, bones, rigid_names=("forearm", "palm"),
                    names=None, hf=None, web_yrange=None):
    """R2.7 finger-wise skinning weights.

    Base falloff: w = exp(-((d-d_min)/0.45)**2), top-4 renormalised.  A
    vertex whose nearest bone belongs to finger `f` blends ONLY the bones
    of finger `f` (+ rigid palm/forearm/CMC) -- verts near finger i get 0%
    from finger j != i, so the H7 <=5% leakage criterion holds exactly.
    The exception is the interdigital web band (`web_yrange`, side-local y):
    there the two nearest bones may be adjacent fingers and the standard
    top-4 blend applies (web verts move with both neighbours).

    Verts whose nearest bone is a rigid (forearm/palm) bone and whose
    runner-up is >1.0 cm further away get a hard 1.0 (exact rigid motion).
    Returns (idx (N,4) int16, w (N,4) float32).
    """
    pts = np.asarray(rest_pts, float)
    N = pts.shape[0]
    B = len(bones)
    dmat = np.empty((N, B), np.float32)
    for b, (a, bb) in enumerate(bones):
        d, _, _ = _seg_proj(pts, np.asarray(a, float), np.asarray(bb, float))
        dmat[:, b] = d.astype(np.float32)
    dmin = dmat.min(axis=1)
    order = np.argsort(dmat, axis=1)
    best = order[:, 0]

    def finger_of(bi):
        if names is None or bi >= len(names):
            return None
        nm = names[bi]
        if len(nm) == 4 and nm[0] == "f" and nm[2] == "_" and \
                nm[1] in "12345" and nm[3] in "012":
            return int(nm[1])
        return None

    f_of_bone = np.array([finger_of(b) for b in range(B)], object)
    rigid = set(rigid_names or ())
    is_rigid = np.array([(names[b] in rigid) if names is not None and
                         b < len(names) else False for b in range(B)])
    is_cmc = np.array([(names[b] == "f1_cmc") if names is not None and
                       b < len(names) else False for b in range(B)])
    best_f = np.array([f_of_bone[b] for b in best], object)
    in_web = np.zeros(N, bool)
    if web_yrange is not None:
        ylo, yhi = web_yrange
        second = order[:, 1]
        second_f = np.array([f_of_bone[b] for b in second], object)
        adj = {(2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4),
               (1, 2), (2, 1)}
        pair_ok = np.array([(a is not None and c is not None
                             and (a, c) in adj)
                            for a, c in zip(best_f.tolist(),
                                            second_f.tolist())])
        in_web = pair_ok & (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi)

    w = np.exp(-((dmat - dmin[:, None]) / 0.45) ** 2).astype(np.float32)
    if names is not None and len(names) == B and \
            any(v is not None for v in f_of_bone):
        # R2.7: mask out other fingers' bones, but only for verts whose
        # nearest bone is itself a finger bone (palm/rigid transitions at
        # the finger base keep their smooth blend) and outside the web.
        best_is_finger = np.array([v is not None for v in best_f.tolist()])
        allow = np.ones((N, B), bool)
        for b in range(B):
            fb = f_of_bone[b]
            if fb is None:
                continue  # rigid / CMC: always allowed
            mask_out = best_is_finger & (best_f != fb) & (~in_web)
            allow[mask_out, b] = False
        # never strand a vertex: the best bone is always allowed
        allow[np.arange(N), best] = True
        w = np.where(allow, w, 0.0)
    idx = np.argsort(-w, axis=1)[:, :4]
    w4 = np.take_along_axis(w, idx, axis=1)
    w4 /= np.maximum(w4.sum(axis=1, keepdims=True), 1e-12)
    if rigid and names is not None and len(names) == B:
        second = dmat[np.arange(N), order[:, 1]]
        hard = is_rigid[best] & ((second - dmin) > 1.0)
        w4[hard] = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
        # keep the best bone first so the hard row is exact
        idx[hard, 1:] = idx[hard, 0:1]
    return idx.astype(np.int16), w4.astype(np.float32)


def get_rest_data(hf, side="right"):
    """Full cached rest set: pts/faces/nrm/idx/w/bones/names/P0/R0/ao.

    Load order:
    1. Package data (prebuilt meshes bundled with pip install)
    2. User cache dir (~/.cache/pianoplayer/)
    3. Build from scratch (first run without prebuilt data)
    """
    s = "left" if str(side).lower().startswith("l") else "right"
    bones, names = rest_bones(hf, s)
    _, _, P0, R0 = rest_frames(hf, s)
    hf_s = f"{float(hf):.2f}"
    hash8 = H.geom_hash()[:8]

    # 1. Try package data first (importlib.resources, works for installed pkg)
    pkg_mesh = f"handmesh_{s}_{hf_s}_{hash8}.npz"
    try:
        from importlib.resources import files as _files
        pkg_path = _files("pianoplayer.data").joinpath(pkg_mesh)
        if pkg_path.is_file():
            with pkg_path.open("rb") as f:
                z = np.load(f, allow_pickle=False)
                if {"rest_pts", "faces", "rest_nrm", "idx", "w"} <= set(z.files):
                    ao = np.array(z.get("rest_ao", np.ones_like(z["rest_pts"][:, 0])), float)
                    return dict(
                        pts=np.array(z["rest_pts"], float),
                        faces=[tuple(int(v) for v in row)
                               for row in np.array(z["faces"], np.int64)],
                        nrm=np.array(z["rest_nrm"], float),
                        idx=np.array(z["idx"], np.int16),
                        w=np.array(z["w"], np.float32),
                        ao=ao,
                        bones=bones, names=names, P0=P0, R0=R0)
    except Exception:
        pass

    # 2. Try user cache dir
    path = cache_path(hf, s)
    z = _load_cache(path)
    if z is not None:
        try:
            if {"rest_pts", "faces", "rest_nrm", "idx", "w"} <= set(z.files):
                ao = np.array(z.get("rest_ao", np.ones_like(z["rest_pts"][:, 0])), float)
                return dict(
                    pts=np.array(z["rest_pts"], float),
                    faces=[tuple(int(v) for v in row)
                           for row in np.array(z["faces"], np.int64)],
                    nrm=np.array(z["rest_nrm"], float),
                    idx=np.array(z["idx"], np.int16),
                    w=np.array(z["w"], np.float32),
                    ao=ao,
                    bones=bones, names=names, P0=P0, R0=R0)
        except Exception:
            pass

    # 3. Build from scratch
    pts, faces, nrm = build_rest_mesh(hf, s)
    # Load AO from the cache we just wrote
    z2 = _load_cache(path)
    ao = np.ones(pts.shape[0], float)
    if z2 is not None and "rest_ao" in z2:
        ao = np.array(z2["rest_ao"], float)
    y_kn = 8.2 * float(hf)
    idx, w = compute_weights(np.asarray(pts, float), bones, names=names,
                             hf=hf,
                             web_yrange=(y_kn + H.WEB_Y_LO * float(hf),
                                         y_kn + H.WEB_Y_HI * float(hf)))
    try:
        np.savez_compressed(path, rest_pts=np.asarray(pts, np.float32),
                            faces=np.array(faces, np.int64),
                            rest_nrm=np.asarray(nrm, np.float32),
                            idx=np.asarray(idx, np.int16),
                            w=np.asarray(w, np.float32),
                            rest_ao=np.asarray(ao, np.float32))
    except Exception:
        pass
    return dict(pts=np.array(pts, float), faces=faces,
                nrm=np.array(nrm, float), idx=idx, w=w,
                ao=ao,
                bones=bones, names=names, P0=P0, R0=R0)


# --------------------------------------------------------------------------- #
# planar UV + procedural skin texture (H9)
# --------------------------------------------------------------------------- #
def rest_uv(rest_pts):
    """Planar top-down UV from the rest bounds (static through skinning)."""
    p = np.asarray(rest_pts, float)
    xmin, xmax = p[:, 0].min(), p[:, 0].max()
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    W = max(float(xmax - xmin), 1e-9)
    H = max(float(ymax - ymin), 1e-9)
    uv = np.stack([(p[:, 0] - xmin) / W, (p[:, 1] - ymin) / H], axis=1)
    return uv.astype(np.float32), (xmin, xmax, ymin, ymax)


def _value_noise(size, cells, seed):
    """Smooth value noise: small random grid upscaled with PIL (no scipy)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    small = rng.random((cells, cells)).astype(np.float32)
    img = Image.fromarray((small * 255.0 + 0.5).astype(np.uint8), "L")
    return np.asarray(img.resize((size, size),
                                 Image.BILINEAR)).astype(np.float32) / 255.0


def build_skin_texture(rest_pts, bones, out_png, names=None, hf=0.82,
                       side="right", size=None):
    """Procedural skin PNG: tone noise, joint/palm creases, nails.

    Rendered from side-local bones, so creases/nails align on every (hf,
    side) pair (R1.5); the pore speckle seed differs per side so the files
    are never accidentally shared. `size` is the edge in px (512 default,
    1024 via PIANOPLAYER_TEXSIZE=1024 env or explicit arg) -- 0 frame cost,
    ~4x VRAM for the hi-res variant.
    """
    from PIL import Image, ImageDraw
    if size is None:
        try:
            size = int(os.environ.get("PIANOPLAYER_TEXSIZE", "512"))
        except ValueError:
            size = 512
    S = int(size) if int(size) in (512, 1024) else 512
    base = np.array([0.95, 0.80, 0.70], float)
    n1 = _value_noise(S, 24, 7)
    n2 = _value_noise(S, 96, 13)
    tone = 1.0 + 0.03 * (0.65 * (n1 - 0.5) * 2.0 + 0.35 * (n2 - 0.5) * 2.0)
    img = np.empty((S, S, 3), float)
    for i in range(3):
        img[..., i] = base[i] * tone
    pil = Image.fromarray((np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    dr = ImageDraw.Draw(pil, "RGBA")
    p = np.asarray(rest_pts, float)
    xmin, xmax = p[:, 0].min(), p[:, 0].max()
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    W = max(float(xmax - xmin), 1e-9)
    H = max(float(ymax - ymin), 1e-9)

    def to_px(x, y):
        return ((float(x) - xmin) / W * (S - 1),
                (float(y) - ymin) / H * (S - 1))

    # finger joints + tips from the bone segments
    joints = {}
    tips = {}
    if names is not None:
        by_name = {nm: (np.asarray(a, float), np.asarray(b, float))
                   for nm, (a, b) in zip(names, bones)}
        for f in (1, 2, 3, 4, 5):
            try:
                a0, b0 = by_name["f%d_0" % f]
                _a1, b1 = by_name["f%d_1" % f]
                _a2, b2 = by_name["f%d_2" % f]
            except KeyError:
                continue
            joints[f] = (a0, b0, b1)
            tips[f] = b2
    else:
        for f in (1, 2, 3, 4, 5):
            try:
                a, b = bones[(f - 1) * 3]
                _a, b1 = bones[(f - 1) * 3 + 1]
                _a, b2 = bones[(f - 1) * 3 + 2]
            except Exception:
                continue
            joints[f] = (np.asarray(a, float), np.asarray(b, float),
                         np.asarray(b1, float))
            tips[f] = np.asarray(b2, float)
    # metacarpal tendon valleys: soft dark longitudinal lines on the dorsum
    # between the finger bases (the sample photo shows clear tendon shading;
    # flat tone noise cannot produce it). Positions from the MCP bases.
    try:
        mcps = [joints[f][0] for f in (2, 3, 4, 5) if f in joints]
        if len(mcps) == 4:
            xs = sorted([float(m[0]) for m in mcps])
            y1 = max(float(m[1]) for m in mcps)
            y0 = y1 - 8.0 * float(hf)
            for xa, xb in zip(xs, xs[1:]):
                xm = 0.5 * (xa + xb)
                x0, yy0 = to_px(xm, y0)
                x1, yy1 = to_px(xm, y1)
                for dx, col in ((-2, (150, 100, 90, 34)),
                                (0, (150, 100, 90, 44)),
                                (2, (150, 100, 90, 34))):
                    dr.line([(x0 + dx, yy0), (x1 + dx, yy1)],
                            fill=col, width=4)
    except Exception:
        pass
    # joint-crease bands (~12% dark + light ridge) across each finger u-span
    for f, (qm, qp, qd) in joints.items():
        r = 0.95 * float(hf)
        for j in (qm, qp, qd):
            x0, y0 = to_px(j[0] - r, j[1])
            x1, y1 = to_px(j[0] + r, j[1])
            for dy, col in ((-3, (120, 70, 60, 60)), (-1, (255, 235, 225, 70)),
                            (1, (120, 70, 60, 46))):
                dr.line([(x0, y0 + dy), (x1, y1 + dy)], fill=col, width=3)
    # palm-crease arcs
    for (x0, y0, x1, y1) in ((-2.5, 3.0, 3.5, 5.5), (-3.0, 1.0, 2.5, 3.0),
                             (0.5, 0.0, 4.0, 4.0)):
        try:
            ax0, ay0 = to_px(x0 * float(hf), y0 * float(hf))
            ax1, ay1 = to_px(x1 * float(hf), y1 * float(hf))
            dr.arc([ax0, ay0 - 30, ax1, ay1 + 30], 200, 340,
                   fill=(130, 75, 65, 55), width=3)
        except Exception:
            pass
    # nails: pinkish ellipse with a whiter tip + cuticle arc (R4: boost contrast)
    for f, t in tips.items():
        try:
            r = 0.50 * float(hf)
            x0, y0 = to_px(t[0] - r, t[1] - 1.15 * float(hf))
            x1, y1 = to_px(t[0] + r, t[1] - 0.10 * float(hf))
            # Base nail color (within 8% of skin tone)
            skin_r, skin_g, skin_b = 247, 209, 189  # base skin-ish
            # Brighter rim (lunula area) - 8% lighter
            rim_r, rim_g, rim_b = min(255, int(skin_r * 1.08)), min(255, int(skin_g * 1.08)), min(255, int(skin_b * 1.08))
            # Whiter tip - 3% lighter than rim (was 6%: blew out to white
            # blobs on curled fingertips in side views)
            tip_r, tip_g, tip_b = min(255, int(rim_r * 1.03)), min(255, int(rim_g * 1.03)), min(255, int(rim_b * 1.03))
            # Cuticle - 10% darker
            cut_r, cut_g, cut_b = int(skin_r * 0.9), int(skin_g * 0.9), int(skin_b * 0.9)
            
            dr.ellipse([x0, y0, x1, y1], fill=(skin_r, skin_g, skin_b, 255))
            # Lunula (proximal half-moon) - slightly lighter
            dr.ellipse([x0 + 2, y0 + 2, x1 - 2, y0 + (y1 - y0) * 0.55],
                       fill=(rim_r, rim_g, rim_b, 255))
            # Tip highlight - brightest
            dr.ellipse([x0 + 4, y0 + (y1 - y0) * 0.55, x1 - 4, y1 - 2],
                       fill=(tip_r, tip_g, tip_b, 255))
            # Cuticle arc - darker
            dr.arc([x0, y0, x1, y1], 100, 260, fill=(cut_r, cut_g, cut_b, 180),
                   width=2)
            # Subtle rim highlight
            dr.arc([x0 + 1, y0 + 1, x1 - 1, y1 - 1], 270, 90, fill=(tip_r, tip_g, tip_b, 100),
                   width=1)
        except Exception:
            pass
    arr = np.asarray(pil).astype(np.float32) / 255.0
    # fingertip / palm-edge blush (warm, subtle)
    yy, xx = np.mgrid[0:S, 0:S]
    blush = np.zeros((S, S), float)
    for f, t in tips.items():
        cx, cy = to_px(t[0], t[1])
        blush += np.exp(-(((xx - cx) / 26.0) ** 2 + ((yy - cy) / 30.0) ** 2))
    blush = np.clip(blush, 0, 1) * 0.5
    arr[..., 0] = np.clip(arr[..., 0] + 0.015 * blush, 0, 1)
    arr[..., 1] = np.clip(arr[..., 1] - 0.010 * blush, 0, 1)
    arr[..., 2] = np.clip(arr[..., 2] - 0.008 * blush, 0, 1)
    # pore speckle (seed per side: the files must never be shared, R1.5)
    rng = np.random.default_rng(29 if str(side).lower().startswith("r")
                                else 31)
    arr *= (1.0 + (rng.random((S, S)) - 0.5) * 0.018)[..., None]
    out = Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        out.save(out_png)
    except Exception:
        pass
    return out_png


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys as _sys
    if "--prune-cache" in _sys.argv[1:]:
        print("pruned %d stale file(s) in %s"
              % (clear_cache(prune_only=True), cache_dir()))
    elif "--clear-cache" in _sys.argv[1:]:
        print("cleared %d file(s) in %s" % (clear_cache(), cache_dir()))
    else:
        print("usage: python -m pianoplayer.handmesh --clear-cache "
              "[--prune-cache]")
        _sys.exit(2)
