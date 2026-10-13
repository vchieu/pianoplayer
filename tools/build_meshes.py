#!/usr/bin/env python3
"""
Prebuild hand meshes and textures for all 7 sizes × 2 sides.

Writes to pianoplayer/data/ so they are included in the package.
Run from repo root: python tools/build_meshes.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pianoplayer.handmesh import (
    build_rest_mesh, build_skin_texture, compute_weights,
    rest_bones, rest_frames
)
from pianoplayer.handspec import geom_hash, HANDMESH_VERSION
from pianoplayer.utils import handSizeFactor
import numpy as np

# 7 sizes and their hand factor (hf)
SIZES = {
    "XXS": 0.33,
    "XS": 0.46,
    "S": 0.64,
    "M": 0.82,
    "L": 1.0,
    "XL": 1.1,
    "XXL": 1.2,
}

SIDES = ("right", "left")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pianoplayer", "data")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    hash8 = geom_hash()[:8]
    
    print(f"Building meshes for HANDMESH_VERSION={HANDMESH_VERSION}, hash={hash8}")
    print(f"Output directory: {DATA_DIR}")
    
    for side in SIDES:
        for size_name, hf in SIZES.items():
            print(f"\n=== {side} {size_name} (hf={hf:.2f}) ===")
            
            # Build mesh from scratch (bypass package data cache)
            s = "left" if str(side).lower().startswith("l") else "right"
            pts, faces, nrm = build_rest_mesh(hf, s)
            
            # Load AO from cache (saved by build_rest_mesh)
            from pianoplayer.handmesh import cache_path, _load_cache
            path = cache_path(hf, s)
            z = _load_cache(path)
            ao = np.ones(pts.shape[0], np.float32)
            if z is not None and "rest_ao" in z:
                ao = np.array(z["rest_ao"], np.float32)
            
            bones, names = rest_bones(hf, s)
            _, _, P0, R0 = rest_frames(hf, s)
            y_kn = 8.2 * float(hf)
            from pianoplayer import handspec as H
            idx, w = compute_weights(np.asarray(pts, float), bones, names=names,
                                     hf=hf,
                                     web_yrange=(y_kn + H.WEB_Y_LO * float(hf),
                                                 y_kn + H.WEB_Y_HI * float(hf)))
            
            # Target package data paths
            mesh_fn = f"handmesh_{side}_{hf:.2f}_{hash8}.npz"
            tex_fn = f"handtex_{side}_{hf:.2f}_{hash8}.png"
            mesh_path = os.path.join(DATA_DIR, mesh_fn)
            tex_path = os.path.join(DATA_DIR, tex_fn)
            
            # Save the full rest data (pts, faces, nrm, idx, w, bones, names, P0, R0, ao)
            try:
                np.savez_compressed(
                    mesh_path,
                    rest_pts=np.asarray(pts, np.float32),
                    faces=np.array(faces, np.int64),
                    rest_nrm=np.asarray(nrm, np.float32),
                    idx=np.asarray(idx, np.int16),
                    w=np.asarray(w, np.float32),
                    rest_ao=np.asarray(ao, np.float32),
                )
                print(f"  Saved mesh: {mesh_fn}")
            except Exception as e:
                print(f"  ERROR saving mesh: {e}")
            
            # Build and save texture
            try:
                build_skin_texture(
                    np.asarray(pts, float),
                    bones,
                    tex_path,
                    names=names,
                    hf=hf,
                    side=side
                )
                print(f"  Saved texture: {tex_fn}")
            except Exception as e:
                print(f"  ERROR saving texture: {e}")
    
    print("\n=== Done ===")
    print(f"All files written to {DATA_DIR}")
    print("Run `pip install -e .` to include them in the package.")

if __name__ == "__main__":
    main()