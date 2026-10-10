"""
Real-time diagnostic for the audio engine: runs the REAL device in the
background and measures everything that can make live playback differ from
an offline render (underruns, buffer sizes, clock drift, dropouts).

Silent: volume is 0, so nothing is heard.

    python diag_live.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pianoplayer.audio import SoundEngine, SynthCore

DUR = 6.0
SR = 44100

S = {"cb": 0, "frames": 0, "render": 0.0, "max_render": 0.0,
     "late": 0, "sizes": {}, "out": []}

_orig = SynthCore.render_block


def timed(self, n):
    t0 = time.perf_counter()
    out = _orig(self, n)
    dt = time.perf_counter() - t0
    budget = n / float(self.sr)
    S["cb"] += 1
    S["frames"] += n
    S["render"] += dt
    S["max_render"] = max(S["max_render"], dt)
    if dt > budget:
        S["late"] += 1
    S["sizes"][n] = S["sizes"].get(n, 0) + 1
    S["out"].append(out.copy())
    return out


SynthCore.render_block = timed


def main():
    eng = SoundEngine(volume=0.0, enabled=True, backend="auto")
    eng.start()
    print("backend:", eng.backend_name, "| started:", eng.now)

    # schedule exactly like the player does: ahead of the *synth* clock
    # (which leads the wall clock by the playback cushion), never in its past.
    # The 1 s lead-in mirrors LEAD_IN: engine time 0 has already been
    # pre-rendered by start(), so a note at t=0 would be clamped to it.
    base = 1.0
    notes = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79]
    ev = []
    t = 0.0
    i = 0
    t0 = time.time()
    while time.time() - t0 < DUR:
        eng.pump()                      # what the animation thread does
        now = eng.now                   # engine clock, not the wall clock
        while i < 60 and (base + i * 0.25) - now <= 0.60:
            m = notes[i % len(notes)]
            eng.note_on(m, 0.7, when=base + i * 0.25, pan=0.3)
            eng.note_off(m, when=base + i * 0.25 + 0.20)
            ev.append((base + i * 0.25, 'on', m, 0.7, 0.3))
            ev.append((base + i * 0.25 + 0.20, 'off', m, 0.0, 0.3))
            i += 1
        time.sleep(0.005)
    wall = time.time() - t0
    eng.now_end = eng.now
    drift = eng.now - wall
    time.sleep(0.4)                      # let the tail render
    eng.stop()

    live = np.concatenate(S["out"]) if S["out"] else np.zeros((0, 2))
    dur_rendered = len(live) / float(SR)
    print("wall %.2f s | engine rendered %.2f s | drift %+.3f s"
          % (wall, dur_rendered, drift))
    print("callbacks %d | frames %d | avg buffer %.1f ms | sizes %s"
          % (S["cb"], S["frames"], 1000.0 * S["frames"] / max(S["cb"], 1) / SR,
             dict(sorted(S["sizes"].items()))))
    print("render time: mean %.2f ms, max %.2f ms | over budget: %d/%d"
          % (1000.0 * S["render"] / max(S["cb"], 1), 1000.0 * S["max_render"],
             S["late"], S["cb"]))

    # ---- offline reference with the same events -------------------------
    # (volume=1.0 only for the maths: the offline path never opens a device)
    ref = SoundEngine(volume=1.0, backend="none").render(ev, DUR + 1.0)
    n = min(len(live), len(ref))
    win = int(0.02 * SR)
    a = live[:n].mean(axis=1)
    b = ref[:n].mean(axis=1)
    gaps = 0
    silent_live = 0
    for i in range(0, n - win, win):
        ra = np.sqrt((a[i:i + win] ** 2).mean())
        rb = np.sqrt((b[i:i + win] ** 2).mean())
        if rb > 0.01 and ra < 1e-4:
            gaps += 1
        if rb > 0.01:
            silent_live += 1
    print("20 ms blocks where the offline render has sound: %d" %
          sum(1 for i in range(0, n - win, win)
              if np.sqrt((b[i:i + win] ** 2).mean()) > 0.01))
    print("  ...of those, silent in the LIVE render (dropouts): %d" % gaps)
    k = min(len(a), len(b))
    denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))
    print("correlation live vs offline: %.3f" %
          (float((a[:k] * b[:k]).sum()) / max(denom, 1e-12)))


if __name__ == "__main__":
    main()
