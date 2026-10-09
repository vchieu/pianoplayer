"""Does the vtkShadowMapPass pipeline work (and produce shadows)?"""
from vedo import Box, Plotter, Sphere, Text3D
from vtkmodules.vtkRenderingCore import vtkLight
from vtkmodules.vtkRenderingOpenGL2 import (
    vtkCameraPass,
    vtkRenderPassCollection,
    vtkSequencePass,
    vtkShadowMapBakerPass,
    vtkShadowMapPass,
)


def make_shadow_pass(renderer):
    shadows = vtkShadowMapPass()
    passes = vtkRenderPassCollection()
    baker = shadows.GetShadowMapBakerPass()
    if baker is None:
        baker = vtkShadowMapBakerPass()
        shadows.SetShadowMapBakerPass(baker)
    passes.AddItem(baker)
    passes.AddItem(shadows)
    seq = vtkSequencePass()
    seq.SetPasses(passes)
    cam = vtkCameraPass()
    cam.SetDelegatePass(seq)
    renderer.SetPass(cam)
    return shadows


vp = Plotter(offscreen=True, size=(900, 560), bg=(0.9, 0.92, 0.95))
plane = Box(pos=(0, 0, -0.5), length=14, width=14, height=1,
            c=(0.82, 0.82, 0.86))
ball = Sphere(pos=(0, 0, 3.5), r=2.0, c=(0.9, 0.35, 0.3))
vp += plane, ball
vp += Text3D("shadow test", pos=(-6, -7, 0.1), s=1.0, c='k')

lt = vtkLight()
lt.SetLightTypeToSceneLight()
lt.SetPosition(7.0, -9.0, 13.0)
lt.SetFocalPoint(0, 0, 0)
lt.SetIntensity(1.0)
lt.SetColor(1.0, 0.98, 0.95)
vp.renderer.AddLight(lt)
vp.renderer.LightFollowCameraOff()

try:
    make_shadow_pass(vp.renderer)
    print("shadow pass installed")
except Exception as exc:
    print("shadow pass FAILED:", type(exc).__name__, exc)

vp.camera = dict(pos=(0, -22, 15), focalPoint=(0, 0, 1), viewup=(0, 0, 1))
try:
    vp.show(resetcam=0, interactive=0)
    vp.screenshot("test_shadow_out.png")
    print("rendered ok")
except Exception as exc:
    print("render FAILED:", type(exc).__name__, exc)
