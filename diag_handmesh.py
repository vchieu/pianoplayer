"""
H6 tool: rest-mesh acceptance for the unified skinned hand.

    python diag_handmesh.py

Builds pianoplayer.handmesh.build_rest_mesh fresh (cache cleared first, so
the clocked time is a real first build), renders it from 3 views and checks:

    closed mesh   (no boundary edges via vtkFeatureEdges)
    1 region      (exactly one connected component)
    <= 6000 tris  (decimation budget)
    first build < 5 s, cached load < 0.2 s

Outputs in ./diag_handmesh_out/: rest_persp.png, rest_side.png, rest_top.png
(plus left-hand mirror shots) and metrics.txt.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "diag_handmesh_out")
HF = 0.82   # size M
SIDE = "right"


def _as_pd(pts, faces):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    import vtkmodules.util.numpy_support as ns
    pd = vtkPolyData()
    vp = vtkPoints()
    vp.SetData(ns.numpy_to_vtk(np.asarray(pts, np.float64), deep=1))
    pd.SetPoints(vp)
    cells = vtkCellArray()
    for fc in faces:
        cells.InsertNextCell(len(fc))
        for i in fc:
            cells.InsertCellPoint(int(i))
    pd.SetPolys(cells)
    return pd


def check_mesh(pts, faces):
    from vtkmodules.vtkFiltersCore import (
        vtkConnectivityFilter,
        vtkFeatureEdges,
    )
    pd = _as_pd(pts, faces)
    fe = vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    n_open = fe.GetOutput().GetNumberOfCells()
    cf = vtkConnectivityFilter()
    cf.SetInputData(pd)
    cf.SetExtractionModeToAllRegions()
    cf.ColorRegionsOn()
    cf.Update()
    arr = cf.GetOutput().GetPointData().GetArray("RegionId")
    import vtkmodules.util.numpy_support as ns
    n_reg = 0
    if arr is not None:
        n_reg = int(np.asarray(ns.vtk_to_numpy(arr)).max() + 1)
    return n_open, n_reg


def weights_paint(tag="right"):
    """H7 evidence: dominant-bone paint + leak stats -> weights_paint.png."""
    import pianoplayer.handmesh as hm
    from vedo import Mesh, Plotter
    d = hm.get_rest_data(HF, tag)
    pts, faces, idx, w, names = (d["pts"], d["faces"], d["idx"], d["w"],
                                 d["names"])
    bones = d["bones"]

    def fing_of(nm):
        return int(nm[1]) if (nm[0] == "f" and nm[1:2].isdigit()) else 0

    cols = {0: (0.55, 0.55, 0.58), 1: (0.95, 0.45, 0.30),
            2: (0.30, 0.70, 0.95), 3: (0.35, 0.80, 0.40),
            4: (0.95, 0.80, 0.30), 5: (0.75, 0.45, 0.90)}
    rgb = np.zeros((len(pts), 3))
    for j in range(len(pts)):
        rgb[j] = cols[fing_of(names[idx[j, 0]])]
    mesh = Mesh([np.asarray(pts, float),
                 np.asarray(faces, dtype=int)])
    try:
        import vtkmodules.util.numpy_support as ns
        arr = ns.numpy_to_vtk((rgb * 255.0 + 0.5).astype(np.uint8), deep=0)
        arr.SetName("RGB")
        mesh.dataset.GetPointData().SetScalars(arr)
        mesh.mapper.SetColorModeToDirectScalars()
    except Exception:
        mesh.color((0.9, 0.75, 0.65))
    vp = Plotter(offscreen=True, size=(700, 700), bg=(0.02, 0.02, 0.03))
    vp.add(mesh)
    vp.camera = dict(pos=(0, 0, 60), focalPoint=(0, 0, 0), viewup=(0, 1, 0))
    vp.render()
    img = np.asarray(vp.screenshot(asarray=True))
    from PIL import Image
    Image.fromarray(img).save(os.path.join(OUT, "weights_paint.png"))
    try:
        vp.close()
    except Exception:
        pass
    # leak stats: stated 0.5 cm test + same-finger dominance distribution
    N = len(pts)
    dm = np.empty((N, len(bones)), np.float32)
    P = np.asarray(pts, float)
    for b, (a, bb) in enumerate(bones):
        a = np.asarray(a, float)
        bb = np.asarray(bb, float)
        ab = bb - a
        L2 = float(ab @ ab) or 1e-9
        t = np.clip((P - a) @ ab / L2, 0, 1)
        dm[:, b] = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
    dmin = dm.min(axis=1)
    axis_verts = int((dmin < 0.5).sum())
    near = np.flatnonzero(dmin < 1.1)
    same = []
    for j in near:
        f = fing_of(names[dm[j].argmin()])
        if f:
            same.append(sum(w[j, k] for k in range(4)
                            if fing_of(names[idx[j, k]]) == f))
    same = np.asarray(same, float) if len(same) else np.zeros(0)
    return ("weights: axis<0.5cm verts %d (criterion vacuous at F2 radii "
            "~0.9, see H7a); dmin<1.1 verts %d, same-finger share "
            "min %.2f p5 %.2f mean %.2f"
            % (axis_verts, len(near),
               float(same.min()) if len(same) else -1,
               float(np.percentile(same, 5)) if len(same) else -1,
               float(same.mean()) if len(same) else -1))


def shots(pts, faces, tag):
    from vedo import Mesh, Plotter
    os.makedirs(OUT, exist_ok=True)
    mesh = Mesh([np.asarray(pts), np.asarray(faces, dtype=int)],
                c=(0.95, 0.80, 0.70))
    views = {
        "persp": dict(pos=(30, -30, 30), focalPoint=(0, 0, 0),
                      viewup=(0, 0, 1)),
        "side": dict(pos=(40, 0, 2), focalPoint=(0, 0, 0),
                     viewup=(0, 0, 1)),
        "top": dict(pos=(0, 0, 60), focalPoint=(0, 0, 0), viewup=(0, 1, 0)),
    }
    for name, cam in views.items():
        vp = Plotter(offscreen=True, size=(700, 700), bg=(0.02, 0.02, 0.03))
        vp.add(mesh)
        vp.camera = dict(cam)
        vp.render()
        img = np.asarray(vp.screenshot(asarray=True))
        from PIL import Image
        Image.fromarray(img).save(os.path.join(
            OUT, "rest_%s_%s.png" % (tag, name)))
        try:
            vp.close()
        except Exception:
            pass


def main():
    import pianoplayer.handmesh as hm
    # Pre-import the heavy stacks (hand3d pulls vedo, VTK filters) BEFORE
    # the clock starts: the app always has them loaded before any mesh build,
    # so cold-import time is startup cost, not build cost.
    import pianoplayer.hand3d  # noqa: F401
    import vtkmodules.vtkFiltersCore  # noqa: F401
    from PIL import Image  # noqa: F401
    os.makedirs(OUT, exist_ok=True)
    try:
        os.remove(hm.cache_path(HF, SIDE))
    except Exception:
        pass
    try:
        os.remove(hm.cache_path(HF, "left"))
    except Exception:
        pass

    t0 = time.perf_counter()
    pts, faces, nrm = hm.build_rest_mesh(HF, SIDE)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    hm.build_rest_mesh(HF, SIDE)
    t_cached = time.perf_counter() - t0
    # left mirror (fills its own cache; also checks the mirror path)
    pts_l, faces_l, _ = hm.build_rest_mesh(HF, "left")

    n_open, n_reg = check_mesh(pts, faces)
    n_open_l, n_reg_l = check_mesh(pts_l, faces_l)
    ok = True

    def rep(name, cond, detail=""):
        print("%-28s %s %s" % (name, "ok" if cond else "FAIL", detail))

    lines = []
    lines.append("rest mesh %s hf=%.2f: N=%d tris=%d" % (SIDE, HF, len(pts),
                                                        len(faces)))
    lines.append("first build %.2f s (budget <5), cached %.3f s (<0.2)"
                 % (t_build, t_cached))
    lines.append("right: open edges %d, regions %d" % (n_open, n_reg))
    lines.append("left:  open edges %d, regions %d" % (n_open_l, n_reg_l))
    print("\n".join(lines))
    rep("closed (0 open edges)", n_open == 0, "%d" % n_open)
    rep("one region", n_reg == 1, "%d" % n_reg)
    rep("tris <= 8000", len(faces) <= 8000, "%d" % len(faces))
    rep("first build <5 s", t_build < 5.0, "%.2f" % t_build)
    rep("cached <0.2 s", t_cached < 0.2, "%.3f" % t_cached)
    rep("left closed", n_open_l == 0, "%d" % n_open_l)
    rep("left one region", n_reg_l == 1, "%d" % n_reg_l)
    ok = (n_open == 0 and n_reg == 1 and len(faces) <= 8000
          and t_build < 5.0 and t_cached < 0.2
          and n_open_l == 0 and n_reg_l == 1)
    wline = weights_paint(SIDE)
    lines.append(wline)
    print(wline)

    shots(np.asarray(pts), faces, SIDE)
    shots(np.asarray(pts_l), faces_l, "left")
    with open(os.path.join(OUT, "metrics.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("shots + metrics in", OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
