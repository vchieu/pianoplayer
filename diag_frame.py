"""
Profile VirtualKeyboard._frame() and plotter.render() separately, under the
same conditions as a real playback.

    python diag_frame.py [score.xml] [frames]
"""
import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(fn="scores/bach_invention4.xml", nframes=60):
    from music21 import converter
    from pianoplayer.hand import Hand
    from pianoplayer.scorereader import reader
    from pianoplayer.utils import score_tempo
    from pianoplayer.vkeyboard import VirtualKeyboard

    sf = converter.parse(os.path.join(HERE, fn))
    tempo = score_tempo(sf)
    hands = []
    for beam, sign in ((0, 1.0), (1, -1.0)):
        h = Hand("right" if sign > 0 else "left", "M")
        h.verbose = False
        h.noteseq = reader(sf, beam=beam)
        if not h.noteseq:
            continue
        h.generate(1, 1000)
        hands.append((h, sign))

    vk = VirtualKeyboard(songname=fn, tempo=tempo, offscreen=True)
    vk.verbose = False
    vk.volume = 0.0
    for h, sign in hands:
        (vk.build_RH if sign > 0 else vk.build_LH)(h)
    vk._prepare()

    dt = 1.0 / 60.0
    t = -0.5
    for _ in range(5):                     # warm up
        vk._frame(t, dt)
        vk.vp.render()
        t += dt

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(nframes):
        vk._frame(t, dt)
        t += dt
    pr.disable()
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    st.print_stats(22)
    print(buf.getvalue())

    py, gl = [], []
    for _ in range(60):
        t0 = time.perf_counter()
        vk._frame(t, dt)
        t1 = time.perf_counter()
        vk.vp.render()
        t2 = time.perf_counter()
        py.append(1000 * (t1 - t0))
        gl.append(1000 * (t2 - t1))
        t += dt
    print("_frame(): mean %.1f ms  max %.1f" % (np.mean(py), np.max(py)))
    print("render() : mean %.1f ms  max %.1f" % (np.mean(gl), np.max(gl)))
    vk._stop_audio()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scores/bach_invention4.xml",
         int(sys.argv[2]) if len(sys.argv) > 2 else 60)
