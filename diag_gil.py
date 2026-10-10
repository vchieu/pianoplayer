"""Is the interpreter lock held while VTK renders, and where is the audio
feeder thread stuck while it should be rendering?

Two probes:
  1. a spinning thread measures how much of the GIL+CPU it can get during
     _frame() / vp.render() / both, against an idle baseline;
  2. faulthandler dumps every thread every 2 s while the real playback loop
     runs, showing the exact line the feeder is blocked on.

    python diag_gil.py [score.xml]
"""
import faulthandler
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Spinner(threading.Thread):
    """Tight Python loop: it can only advance while it owns the GIL."""

    def __init__(self):
        super().__init__(daemon=True)
        self.phase = "idle"
        self.stop = False
        self.cpu, self.wall, self.iters = {}, {}, {}

    def run(self):
        while not self.stop:
            p = self.phase
            c0 = time.thread_time()
            w0 = time.perf_counter()
            it = 0
            while self.phase == p and not self.stop:
                it += 1
            self.cpu[p] = self.cpu.get(p, 0.0) + time.thread_time() - c0
            self.wall[p] = self.wall.get(p, 0.0) + time.perf_counter() - w0
            self.iters[p] = self.iters.get(p, 0) + it


def report(sp, name, wall):
    cpu = sp.cpu.get(name, 0.0)
    sampled = sp.wall.get(name, 0.0)
    it = sp.iters.get(name, 0)
    sp.cpu[name] = sp.wall[name] = sp.iters[name] = 0.0
    print("  %-14s wall %.3f s | spinner got GIL %.1f%% (%.3f s cpu, %.1fM iters)"
          % (name, wall, 100.0 * cpu / max(sampled, 1e-9), cpu, it / 1e6))


def main(fn="scores/bach_invention4.xml", seconds=8.0):
    from music21 import converter
    from pianoplayer.hand import Hand
    from pianoplayer.scorereader import reader
    from pianoplayer.utils import score_tempo
    from pianoplayer.vkeyboard import VirtualKeyboard

    print("python %s | cpus %s | gil %s"
          % (sys.version.split()[0], os.cpu_count(),
             sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else "?"))

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

    sp = Spinner()
    sp.start()
    time.sleep(0.5)

    def frames(n=60):
        nonlocal t
        for _ in range(n):
            vk._frame(t, dt)
            t += dt

    def renders(n=60):
        for _ in range(n):
            vk.vp.render()

    def both(n=60):
        nonlocal t
        for _ in range(n):
            vk._frame(t, dt)
            vk.vp.render()
            t += dt

    print("--- GIL available to a second thread -----------------------")
    sp.phase = "settle"
    time.sleep(0.3)
    for name, fn2 in (("idle", lambda: time.sleep(1.0)),
                      ("frame()", frames), ("render()", renders),
                      ("both", both)):
        sp.phase = "settle"
        time.sleep(0.3)
        sp.phase = name
        w0 = time.perf_counter()
        fn2()
        wall = time.perf_counter() - w0
        sp.phase = "settle"
        time.sleep(0.3)
        report(sp, name, wall)

    # ---- phase 2: real playback load + thread dumps -------------------
    print("--- feeder thread under load (dump every 2 s) ---------------")
    faulthandler.dump_traceback_later(2.0, repeat=True, file=sys.stdout)
    t0 = time.time()
    t = -0.5
    while time.time() - t0 < seconds:
        vk._frame(t, dt)
        vk.vp.render()
        t += dt
    faulthandler.cancel_dump_traceback_later()

    print("engine %.2f s rendered in %.2f s wall"
          % (vk.engine.now if vk.engine else 0.0, time.time() - t0))
    vk._stop_audio()
    sp.stop = True


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scores/bach_invention4.xml",
         float(sys.argv[2]) if len(sys.argv) > 2 else 8.0)
