"""Vendored 2D hand model from pygame-piano-animator (hands.py + skins.py),
drawn offscreen with pygame and composited into the 3D vedo/VTK scene.

Only the *hand rendering* comes from the 2D project; music, fingering,
audio, keys, falling notes and HUD stay pianoplayer's.

The vendored modules are byte-identical to
D:\\Project\\pygame-piano-animator except for import lines, which point at
this package.  `midi_loader`, `common` and `paths` are small shims (the
originals drag in pretty_midi, the whole 2D keyboard and the user's data
folders).  `render.Hands2D` is the adapter: it owns the offscreen surface
and the vtkImageActor that shows it.
"""
from .render import Hands2D

__all__ = ["Hands2D"]