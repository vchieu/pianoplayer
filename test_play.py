"""
Headless playback test.

Drives the VirtualKeyboard frame loop offline (no window, no audio device)
and writes screenshots + sanity checks in ./test_play_out/.

    python test_play.py [score.xml] [seconds]
"""
import os
import sys

from music21 import converter

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import score_tempo
from pianoplayer.vkeyboard import VirtualKeyboard

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_play_out")


def build(fn, size="M", nmeasures=1000):
    sf = converter.parse(fn)
    tempo = score_tempo(sf)
    rh = Hand("right", size)
    rh.verbose = False
    rh.noteseq = reader(sf, beam=0)
    rh.generate(0, nmeasures)

    lh = Hand("left", size)
    lh.verbose = False
    lh.noteseq = reader(sf, beam=1)
    lh.generate(0, nmeasures)
    return sf, tempo, rh, lh


def main(fn="scores/test_scales.xml", seconds=7.0, nmeasures=1000):
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".png"):
            os.remove(os.path.join(OUT, f))

    sf, tempo, rh, lh = build(fn, nmeasures=nmeasures)
    print("tempo", tempo, "RH notes", len(rh.noteseq), "fingers",
          len(rh.fingerseq), "| LH notes", len(lh.noteseq),
          "fingers", len(lh.fingerseq))

    vk = VirtualKeyboard(songname=fn, tempo=tempo, offscreen=True)
    vk.verbose = False
    vk.speedfactor = 1.0
    vk.build_RH(rh)
    vk.build_LH(lh)

    stats = {"on": 0, "off": 0}
    sched_err = []          # (note score time) - (frame time) for every note_on
    clock = {"t": -0.5}
    vk._prepare()
    if vk.engine is not None:
        raw_on, raw_off = vk.engine.note_on, vk.engine.note_off

        def on(*a, **k):
            stats["on"] += 1
            w = k.get("when")
            if w is not None:
                # the engine works on its own absolute clock: translate the
                # scheduled time back to score time and compare with the frame
                sched_err.append((w + vk._audio_epoch) - clock["t"])
            return raw_on(*a, **k)

        def off(*a, **k):
            stats["off"] += 1
            return raw_off(*a, **k)

        vk.engine.note_on = on
        vk.engine.note_off = off

    # ---- drive the loop ---------------------------------------------------
    dt = 1.0 / 60.0
    shots = {1.0: "t1.0", 3.0: "t3.0", 6.0: "t6.0", 9.0: "t9.0"}
    t = -0.5
    pressed_peak = 0
    keys_down = set()
    rig_moves = []
    while t < seconds:
        clock["t"] = t
        vk._frame(t, dt)
        vk.vp.render()
        pressed_peak = max(pressed_peak,
                           sum(1 for v in vk._key_tgt.values() if v > 0))
        for name, amt in vk._key_amt.items():
            if amt > 0.5:
                keys_down.add(name)
        if abs(t - 3.0) < dt / 2:
            rig_moves.append(tuple(vk.RH.root.round(2)))
        for mark, tag in list(shots.items()):
            if t >= mark:
                vk.vp.screenshot(os.path.join(OUT, "%s.png" % tag))
                del shots[mark]
        t += dt

    vk.vp.screenshot(os.path.join(OUT, "t_end.png"))

    # ---- checks -----------------------------------------------------------
    print("note_on scheduled:", stats["on"], " note_off:", stats["off"])
    print("peak of keys targeted down:", pressed_peak)
    print("keys seen pressed:", sorted(keys_down))
    print("RH root at t=3:", rig_moves)
    print("RH finger_x:", {k: round(v, 2)
                           for k, v in vk.RH._targets_x.items()})
    print("LH finger_x:", {k: round(v, 2)
                           for k, v in vk.LH._targets_x.items()})
    print("HUD sample:", vk._hud.text() if vk._hud else "(none)")

    ok = True
    if sched_err:
        lo, hi = min(sched_err), max(sched_err)
        print("note_on lead time: %.3f .. %.3f s" % (lo, hi))
        if lo < -0.05 or hi > 0.33:
            print("FAIL: audio events are not aligned with the animation")
            ok = False
    elif rh.noteseq:
        print("FAIL: no note carried a scheduling time"); ok = False
    if stats["on"] < 5:
        print("FAIL: not enough audio events scheduled"); ok = False
    if pressed_peak < 1 and len(rh.noteseq):
        print("FAIL: no key ever pressed"); ok = False
    if not keys_down:
        print("FAIL: no key animation"); ok = False
    if len(rh.noteseq) and not vk.RH._targets_x:
        print("FAIL: right hand never got a target"); ok = False
    vk._stop_audio()
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    score = sys.argv[1] if len(sys.argv) > 1 else "scores/test_scales.xml"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
    meas = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    sys.exit(main(score, secs, meas))
