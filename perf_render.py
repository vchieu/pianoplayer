"""Measure the real windowed frame cost under different settings.

    python perf_render.py [score.xml] [frames]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vedo
from vedo import settings

from music21 import converter

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import score_tempo
from pianoplayer.vkeyboard import FPS_MS, VirtualKeyboard

fn = sys.argv[1] if len(sys.argv) > 1 else "scores/bach_invention4.xml"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40

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

CONFIGS = tuple((label, msaa, live)
                for msaa in (0, 2, 4, 8)
                for live in (False, True)
                for label in ("msaa%d %s" % (msaa, "live" if live else "idle"),))

for label, msaa, live in CONFIGS:
    settings.multi_samples = msaa
    vk = VirtualKeyboard(songname=fn, tempo=tempo, offscreen=False)
    vk.verbose = False
    vk.volume = 0.0
    for h, sign in hands:
        (vk.build_RH if sign > 0 else vk.build_LH)(h)
    if live:
        vk._prepare()
        vk._t0 = time.time()
        vk._last = vk._t0
    for _ in range(4):
        if live:
            vk._frame(0.0, FPS_MS / 1000.0)
        vk.vp.render()
    tf = tr = 0.0
    t_end = time.perf_counter() + N * FPS_MS / 1000.0
    n = 0
    t0 = time.perf_counter()
    while n < N:
        f0 = time.perf_counter()
        if live:
            now = time.time() - vk._t0
            vk._frame(now, FPS_MS / 1000.0)
            tf += 1000.0 * (time.perf_counter() - f0)
        r0 = time.perf_counter()
        vk.vp.render()
        tr += 1000.0 * (time.perf_counter() - r0)
        n += 1
        slack = FPS_MS / 1000.0 - (time.perf_counter() - f0)
        if slack > 0:
            time.sleep(slack)
    wall = time.perf_counter() - t0
    ur = vk.engine.underruns if vk.engine else 0
    print("%-11s fps %5.1f | frame %6.1f ms | _frame %5.1f | render %6.1f | "
          "underruns %d" % (label, n / wall,
                            1000.0 * wall / n, tf / n, tr / n, ur))
    sys.stdout.flush()
    try:
        vk.engine.stop()
    except Exception:
        pass
    del vk
print("done")
