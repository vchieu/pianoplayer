"""Is the audio feeder thread really starved, or is the measurement wrong?

Inside the feeder's own loop we time two things while the 3D thread runs:
  * micro : a pure-Python busy loop (only needs the GIL + CPU)
  * render: SynthCore.render_block (needs the GIL + numpy)

If micro's wall/cpu is ~1 but render's is huge, the time goes into numpy;
if both are huge, the thread is waiting for the interpreter lock.

    python diag_cpu.py [score.xml] [seconds]
"""
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

M = {"micro_wall": 0.0, "micro_cpu": 0.0, "micro_n": 0,
     "rend_wall": 0.0, "rend_cpu": 0.0, "rend_n": 0, "rend_max": 0.0}


def probe_thread_time():
    w0 = time.perf_counter()
    c0 = time.thread_time()
    x = 0
    while time.perf_counter() - w0 < 0.5:
        x += 1
    w = time.perf_counter() - w0
    c = time.thread_time() - c0
    print("thread_time probe: wall %.3f s  cpu %.3f s  -> %s"
          % (w, c, "OK" if c > 0.9 * w else "UNDER-REPORTS"))


def install_probe():
    from pianoplayer.audio import _CushionBackend, SynthCore
    orig_chunk = _CushionBackend._render_chunk
    orig_render = SynthCore.render_block

    def render(self, n):
        w0 = time.perf_counter()
        c0 = time.thread_time()
        out = orig_render(self, n)
        M["rend_wall"] += time.perf_counter() - w0
        M["rend_cpu"] += time.thread_time() - c0
        M["rend_n"] += 1
        M["rend_max"] = max(M["rend_max"], time.perf_counter() - w0)
        return out

    def chunk(self):
        w0 = time.perf_counter()
        c0 = time.thread_time()
        x = 0
        for _ in range(20000):        # ~1-2 ms of pure Python
            x += 1
        M["micro_wall"] += time.perf_counter() - w0
        M["micro_cpu"] += time.thread_time() - c0
        M["micro_n"] += 1
        return orig_chunk(self)

    SynthCore.render_block = render
    _CushionBackend._render_chunk = chunk
    return orig_chunk, orig_render


def main(fn="scores/bach_invention4.xml", seconds=8.0):
    probe_thread_time()
    from music21 import converter
    from pianoplayer.hand import Hand
    from pianoplayer.scorereader import reader
    from pianoplayer.utils import score_tempo
    from pianoplayer.vkeyboard import VirtualKeyboard

    install_probe()

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
    print("backend:", vk.engine.backend_name if vk.engine else "(none)")

    dt = 1.0 / 60.0
    t = -0.5
    t0 = time.time()
    while time.time() - t0 < seconds:
        vk._frame(t, dt)
        vk.vp.render()
        t += dt
    wall = time.time() - t0
    engine = vk.engine.now if vk.engine else 0.0
    vk._stop_audio()

    print("--- feeder thread (measured inside its own loop) ------------")
    if M["micro_n"]:
        print("micro (pure python): %d x | wall %.3f s  cpu %.3f s  wall/cpu %.1f"
              % (M["micro_n"], M["micro_wall"], M["micro_cpu"],
                 M["micro_wall"] / max(M["micro_cpu"], 1e-9)))
    if M["rend_n"]:
        print("render_block       : %d x | wall %.3f s  cpu %.3f s  wall/cpu %.1f  max %.0f ms"
              % (M["rend_n"], M["rend_wall"], M["rend_cpu"],
                 M["rend_wall"] / max(M["rend_cpu"], 1e-9),
                 1000.0 * M["rend_max"]))
    print("engine %.2f s rendered in %.2f s wall (need 1.0x)" % (engine, wall))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scores/bach_invention4.xml",
         float(sys.argv[2]) if len(sys.argv) > 2 else 8.0)
