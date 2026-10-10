"""
Same real-time measurement as diag_live.py, but with the REAL load: the 3D
scene is being rendered by vedo/VTK on the main thread while the audio thread
has to keep up.  This is what happens when you run `pianoplayer ... -v`.

Silent (volume = 0).

    python diag_load.py [score.xml] [seconds]
"""
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pianoplayer.audio import SynthCore, _CushionBackend
from pianoplayer.vkeyboard import FPS_MS, LEAD_IN

S = {"cb": 0, "frames": 0, "render": 0.0, "max_render": 0.0, "cpu": 0.0,
     "late": 0, "late_ms": [], "sizes": {},
     "take": 0, "take_n": 0, "take_max": 0.0, "take_max_n": 0}
_orig = SynthCore.render_block


def timed(self, n):
    t0 = time.perf_counter()
    c0 = time.thread_time()
    out = _orig(self, n)
    dt = time.perf_counter() - t0
    S["cb"] += 1
    S["frames"] += n
    S["render"] += dt
    S["cpu"] += time.thread_time() - c0
    S["max_render"] = max(S["max_render"], dt)
    if dt > n / float(self.sr):
        S["late"] += 1
        S["late_ms"].append(1000.0 * dt)
    S["sizes"][n] = S["sizes"].get(n, 0) + 1
    return out


SynthCore.render_block = timed


# --- device callback probe, installed on the CLASS so that even the very
# --- first pulls done inside SoundEngine.start() are visible
_raw_take = _CushionBackend._take


def timed_take(self, frames):
    now = time.perf_counter()
    t_ref = S.setdefault("_t0", now)
    u0 = self.underruns
    out = _raw_take(self, frames)
    dt = time.perf_counter() - now
    if self.underruns > u0:                # _take padded the gap with zeros
        S.setdefault("ur", []).append((now - t_ref, frames))
    if len(S.setdefault("first", [])) < 8:
        S["first"].append((now - t_ref, frames, self._buffered(),
                           self.underruns, threading.current_thread().name))
    S["take_n"] += 1
    S["take"] += dt
    S["take_frames"] = S.get("take_frames", 0) + frames
    gap = (now - S["last_take"]) * 1000.0 if S.get("last_take") else 0.0
    S["last_take"] = now
    if gap:
        S.setdefault("gaps", []).append(gap)
    if dt > S["take_max"]:
        S["take_max"] = dt
        S["take_max_n"] = frames
    return out


_CushionBackend._take = timed_take


def main(fn="scores/bach_invention4.xml", seconds=8.0):
    from music21 import converter
    from pianoplayer.hand import Hand
    from pianoplayer.scorereader import reader
    from pianoplayer.utils import score_tempo
    from pianoplayer.vkeyboard import VirtualKeyboard

    sf = converter.parse(os.path.join(HERE, fn) if not os.path.isabs(fn) else fn)
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
    vk.volume = 0.0                      # silent: we only measure the load
    for h, sign in hands:
        (vk.build_RH if sign > 0 else vk.build_LH)(h)

    vk._prepare()
    print("backend:", vk.engine.backend_name if vk.engine else "(none)")
    be = getattr(vk.engine, "_backend", None) if vk.engine else None
    if be is not None:
        print("output_latency: %.3f s (device period %s ms)"
              % (be.output_latency, _CushionBackend.BUFFER_MS))
    t0 = time.time()

    # score time follows the wall clock, exactly like _on_timer does: a
    # hand-rolled "t += dt" loop would let the synth race ahead of the score
    dt = FPS_MS / 1000.0
    t = -LEAD_IN
    frame_ms, py_ms, gl_ms = [], [], []
    ring = []                          # (elapsed s, free ring frames)
    slow = []                          # frames that took > 100 ms
    while t < seconds:
        f0 = time.perf_counter()
        vk._frame(t, dt)
        f1 = time.perf_counter()
        vk.vp.render()
        f2 = time.perf_counter()
        frame_ms.append(1000.0 * (f2 - f0))
        py_ms.append(1000.0 * (f1 - f0))
        gl_ms.append(1000.0 * (f2 - f1))
        if be is not None:
            ring.append((time.time() - t0, be._buffered()))
        if f2 - f0 > 0.10:
            slow.append((time.time() - t0, 1000.0 * (f2 - f0)))
        tn = time.time() - vk._t0
        dt = tn - t
        if dt <= 0.0 or dt > 0.25:
            dt = FPS_MS / 1000.0
        t = tn

    wall = time.time() - t0
    engine_now = vk.engine.now if vk.engine else 0.0
    live_un = getattr(be, "underruns", "?") if be is not None else "?"
    lat_live = be.output_latency if be is not None else 0.0
    vk._stop_audio()

    print("--- audio thread -------------------------------------------")
    if S["cb"]:
        print("feeder renders %d | wall %.2f s  cpu %.2f s  (%.2f x realtime)"
              % (S["cb"], S["render"], S["cpu"],
                 (S["frames"] / 44100.0) / max(S["cpu"], 1e-9)))
        print("   wall/cpu = %.1f  (1 = running, >>1 = waiting for the GIL/CPU)"
              % (S["render"] / max(S["cpu"], 1e-9)))
    if S["take_n"]:
        period = 1000.0 * S.get("take_frames", 0) / S["take_n"] / 44100.0
        gaps = sorted(S.get("gaps", []))
        print("device callbacks %d | period %.1f ms | mean %.2f ms | max %.1f ms"
              % (S["take_n"], period, 1000.0 * S["take"] / S["take_n"],
                 1000.0 * S["take_max"]))
        if gaps:
            print("   callback gaps: median %.0f ms  p95 %.0f ms  max %.0f ms"
                  "   (period %.0f ms -> late by up to %.0f ms)"
                  % (gaps[len(gaps) // 2], gaps[int(0.95 * len(gaps))], gaps[-1],
                     period, max(0.0, gaps[-1] - period)))
        print("   ring underruns while playing: %s   (after stop: %s)"
              % (live_un, getattr(be, "underruns", "?")))
        if be is not None:
            print("   output_latency after playback: %.3f s" % lat_live)
        for when, want in S.get("ur", []):
            print("      underrun at +%.3f s: %d frames of silence"
                  % (when, want))
        if ring:
            lv = sorted(r for _, r in ring)
            print("   free ring: min %.0f ms  p01 %.0f ms  median %.0f ms"
                  "  (cushion %d ms)"
                  % (1000.0 * lv[0] / 44100.0, 1000.0 * lv[len(lv) // 100] / 44100.0,
                     1000.0 * lv[len(lv) // 2] / 44100.0,
                     int(_CushionBackend.CUSHION * 1000)))
        if slow:
            print("   frames > 100 ms: %d   worst %d"
                  % (len(slow), int(max(ms for _, ms in slow))))
        print("   first takes (+s, frames, ring, underruns, thread):")
        for row in S.get("first", []):
            print("      +%7.3f  n=%-5d ring=%-6d ur=%d  %s" % row)
    print("engine clock vs wall: %+.3f s  (engine %.2f s rendered in %.2f s)"
          % (engine_now - wall, engine_now, wall))
    print("--- main (3D) thread ---------------------------------------")
    print("frame: mean %.1f ms  max %.1f ms  | over 16.7 ms: %d/%d"
          % (np.mean(frame_ms), np.max(frame_ms),
             sum(1 for m in frame_ms if m > 16.7), len(frame_ms)))
    print("   python _frame(): mean %.1f ms | vp.render(): mean %.1f ms"
          % (np.mean(py_ms), np.mean(gl_ms)))


if __name__ == "__main__":
    score = sys.argv[1] if len(sys.argv) > 1 else "scores/bach_invention4.xml"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    main(score, secs)
