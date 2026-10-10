"""
Real GUI measurement: opens the actual window (offscreen=False) and runs the
real vedo timer loop, while recording the audio-thread and per-frame stats.

Same probes as diag_load.py, but this is what `pianoplayer score -v` really
does: a window, a compositor, and VTK rendering on the thread that also pumps
the audio ring.

Silent (volume = 0).

    python diag_gui.py [score.xml] [seconds]
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import diag_load                      # installs the SynthCore/_take probes

from music21 import converter

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import score_tempo
from pianoplayer.vkeyboard import VirtualKeyboard

S = diag_load.S
STATS = {"t0": None, "frame": [], "ring": []}


def report():
    import numpy as np
    wall = time.time() - (STATS["t0"] or time.time())
    fm = np.array(STATS["frame"]) if STATS["frame"] else np.array([0.0])
    print("\n--- main (windowed) thread -------------------------------")
    print("frames %d | wall %.1f s | mean %.1f ms | max %.1f ms | "
          "> 16.7 ms: %d | > 100 ms: %d"
          % (len(fm), wall, fm.mean(), fm.max(),
             int((fm > 16.7).sum()), int((fm > 100).sum())))

    print("--- audio ------------------------------------------------")
    print("ring underruns: %d" % len(S.get("ur", [])))
    for when, want in S.get("ur", [])[:40]:
        print("   underrun at +%.3f s: %d frames of silence" % (when, want))
    if STATS["ring"]:
        lv = sorted(r for _, r in STATS["ring"])
        n = len(lv)
        print("free ring: min %.0f ms | p01 %.0f ms | median %.0f ms"
              % (1000.0 * lv[0] / 44100.0,
                 1000.0 * lv[max(0, n // 100)] / 44100.0,
                 1000.0 * lv[n // 2] / 44100.0))
        drained = [(t, 1000.0 * r / 44100.0) for t, r in STATS["ring"]
                   if r < 0.15 * 44100]
        print("frames with ring < 15%% of cushion: %d" % len(drained))
        for t, r in drained[:20]:
            print("   t=%7.2f s  ring=%.0f ms" % (t, r))
    if S["cb"]:
        print("feeder renders %d | cpu %.2f s | wall %.2f s | wall/cpu %.1f"
              % (S["cb"], S["cpu"], S["render"],
                 S["render"] / max(S["cpu"], 1e-9)))
    fr = STATS.get("fr")
    if fr and fr["n"]:
        tot = fr["frame"] + fr["render"]
        print("split: _frame %.1f ms | vp.render %.1f ms | other %.1f ms | "
              "n=%d" % (fr["frame"] / fr["n"], fr["render"] / fr["n"],
                        max(0.0, tot / fr["n"] - fr["frame"] / fr["n"]
                            - fr["render"] / fr["n"]), fr["n"]))
    rows = STATS.get("rows", [])
    if rows:
        print("  t      gap    render   ring")
        for t, gap, r, ring in rows[:40]:
            print("%6.2f  %6.1f  %6.1f  %5.0fms" % (t or 0, gap, r, ring / 44.1))
    parts = STATS.get("parts", {})
    if parts:
        n = max(fr["n"] if fr and fr["n"] else 1, 1)
        print("--- per-frame breakdown (ms) -----------------------------")
        for k in sorted(parts, key=lambda k: -parts[k]):
            print("  %-16s %7.2f" % (k, parts[k] / n))
    sys.stdout.flush()


def watchdog(seconds):
    """Let the piece play, then report and leave (VTK's own stop is flaky)."""
    t0 = time.time()
    while STATS["t0"] is None and time.time() - t0 < 90.0:
        time.sleep(0.2)
    time.sleep(seconds)
    report()
    time.sleep(1.0)
    print("done", flush=True)
    os._exit(0)


def main(fn="scores/bach_invention4.xml", seconds=40.0):
    path = fn if os.path.isabs(fn) else os.path.join(HERE, fn)
    sf = converter.parse(path)
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

    vk = VirtualKeyboard(songname=fn, tempo=tempo, offscreen=False)
    vk.verbose = False
    vk.volume = 0.0                      # silent: we only measure the load
    for h, sign in hands:
        (vk.build_RH if sign > 0 else vk.build_LH)(h)

    orig = vk._on_timer
    fr = {"frame": 0.0, "render": 0.0, "n": 0}
    STATS["rows"] = []

    orig_frame = vk._frame
    orig_render = vk.vp.render
    _t = {"last": 0.0}
    P = STATS["parts"] = {}
    _done = {"n": False}

    def spy(name, fn):
        def w(*a, **k):
            t0 = time.perf_counter()
            out = fn(*a, **k)
            P[name] = P.get(name, 0.0) + 1000.0 * (time.perf_counter() - t0)
            return out
        return w

    def instrument():
        # _prepare() (inside play()) creates the engine and the players, so
        # the spies can only be installed once they exist.
        if _done["n"]:
            return
        if getattr(vk, "engine", None) is not None and getattr(vk, "_players", None):
            try:
                vk.engine.pump = spy("pump", vk.engine.pump)
            except Exception:
                pass
            vk._sync_keys = spy("sync_keys", vk._sync_keys)
            vk._hud_update = spy("hud", vk._hud_update)
            vk._progress = spy("progress", vk._progress)
            for side, pl in vk._players.items():
                pl.rig.update = spy("rig." + side, pl.rig.update)
            _done["n"] = True
        if not _done.get("vtk"):
            _done["vtk"] = True
            try:
                vk.vp.window.Render = spy("vtk.Render", vk.vp.window.Render)
                vk.vp.initialize_interactor = spy(
                    "init.interactor", vk.vp.initialize_interactor)
                vk.vp.interactor.ProcessEvents = spy(
                    "proc.events", vk.vp.interactor.ProcessEvents)
                _done["vtk"] = True
            except Exception:
                pass

    def timed_frame(now, dt):
        t0 = time.perf_counter()
        orig_frame(now, dt)
        fr["frame"] += 1000.0 * (time.perf_counter() - t0)
        fr["n"] += 1

    def timed_render(*a, **k):
        t0 = time.perf_counter()
        out = orig_render(*a, **k)
        d = 1000.0 * (time.perf_counter() - t0)
        fr["render"] += d
        now = time.perf_counter()
        gap = 1000.0 * (now - _t["last"]) if _t["last"] else 0.0
        _t["last"] = now
        be = getattr(vk.engine, "_backend", None) if vk.engine else None
        STATS["rows"].append(((time.time() - STATS["t0"]) if STATS["t0"]
                              else 0.0,
                              gap, d, be._buffered() if be else 0))
        return out

    vk._frame = timed_frame
    vk.vp.render = timed_render
    STATS["fr"] = fr

    def timed(evt=None):
        try:
            instrument()
            f0 = time.perf_counter()
            orig(evt)
            STATS["frame"].append(1000.0 * (time.perf_counter() - f0))
            if STATS["t0"] is None:
                STATS["t0"] = time.time()
            be = getattr(vk.engine, "_backend", None) if vk.engine else None
            if be is not None and STATS["t0"] is not None:
                STATS["ring"].append((time.time() - STATS["t0"],
                                      be._buffered()))
        except Exception:
            import traceback
            traceback.print_exc()

    # play() wires self._on_timer into the vedo timer; shadow it with timed()
    vk._on_timer = timed
    threading.Thread(target=watchdog, args=(seconds,), daemon=True).start()
    print("entering the window", flush=True)
    if "--manual" in sys.argv:
        _run_manual(vk, seconds)
    else:
        vk.play()                        # blocks until the window is closed


def _run_manual(vk, seconds):
    """Same work as play(), but pump the event loop ourselves.

    Tells us whether the Windows/VTK repeating timer (and not the render
    itself) is what sets the frame rate.
    """
    from pianoplayer.vkeyboard import FPS_MS
    if vk.vp is None:
        vk.build_keyboard()
    try:
        vk.vp.render()
    except Exception:
        pass
    vk._prepare()
    vk._playing = True
    vk._lock_camera()
    if "--noshow" in sys.argv:
        vk.vp.initialize_interactor()
        vk.vp.window.Render()
    else:
        vk.vp.show(interactive=0, resetcam=0)
    t_end = time.time() + seconds + 5.0
    period = FPS_MS / 1000.0
    next_t = time.perf_counter()
    while vk._playing and time.time() < t_end:
        try:
            vk.vp.interactor.ProcessEvents()
        except Exception:
            break
        vk._on_timer()
        next_t += period
        slack = next_t - time.perf_counter()
        if slack > 0:
            time.sleep(slack * 0.5)
        else:
            next_t = time.perf_counter()
    vk._playing = False
    vk._teardown()


if __name__ == "__main__":
    score = sys.argv[1] if len(sys.argv) > 1 else "scores/bach_invention4.xml"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    main(score, secs)
