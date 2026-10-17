"""
paths.py (shim) - only DATA_DIR, for the vendored pianist.py.

The source module also owns the 2D app's bundled-assets and user-data
folders (a `pianists/` folder next to the code, %APPDATA% when compiled).
pianoplayer keeps nothing there - the 2D hands are drawn with the built-in
`Pianist("Default", ...)` anatomy and behaviour - so DATA_DIR just points
into pianoplayer's own cache folder. Nothing is ever created there.
"""
import os

DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pianoplayer", "hands2d")