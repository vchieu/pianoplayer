#!/usr/bin/env python3
"""Repeated same-pitch notes must visibly release the key and the finger
between two strikes (no VTK window needed)."""
import math
import sys

from pianoplayer.vkeyboard import (PRESS_LEAD, TAU_PRESS, TAU_RELEASE, VIS_LIFT,
                                   VirtualKeyboard, _Ev, _Player,
                                   _apply_visual_release)

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def ev(t_on, dur, midi=60, finger=3):
    return _Ev(t_on, t_on + dur, midi, 0.8, finger, False, 1, "C4", None, 0.0)


def simulate(evs, T, fps=60):
    """Run the real _pump/_press/_release; return per-frame key and finger amount."""
    vk = VirtualKeyboard.__new__(VirtualKeyboard)
    vk._key_cnt, vk._key_tgt, vk._key_acc, vk._touched = {}, {}, {}, set()
    vk._key_by_midi = {e.midi: "k%d" % e.midi for e in evs if e.midi is not None}
    vk.KB = {}
    vk._audio_epoch = 0.0
    p = _Player("R", 1, 0.0, None, None)
    p.notes = list(evs)
    dt, key_amt, fin_amt, t = 1.0 / fps, 0.0, 0.0, 0.0
    keys, fingers = [], []
    while t < T:
        vk._pump(p, t, None)
        tgt = vk._key_tgt.get("k60", 0.0)
        tau = TAU_PRESS if tgt > key_amt else TAU_RELEASE
        key_amt = tgt + (key_amt - tgt) * math.exp(-dt / tau)
        tgt = 1.0 if p.active.get(3) else 0.0
        tau = TAU_PRESS if tgt > fin_amt else TAU_RELEASE
        fin_amt = tgt + (fin_amt - tgt) * math.exp(-dt / tau)
        keys.append(key_amt)
        fingers.append(fin_amt)
        t += dt
    return keys, fingers, vk, p


def min_between(series, a, b, fps=60):
    """Smallest value between 30 ms after strike a and strike b."""
    return min(series[int((a.t_on + 0.03) * fps):int(b.t_on * fps) + 1])


def main():
    for label, step in (("0.50 s", 0.5), ("0.25 s", 0.25), ("0.15 s", 0.15)):
        evs = [ev(0.5 + step * i, step) for i in range(6)]       # abutting, same finger
        _apply_visual_release(evs)
        keys, fingers, vk, p = simulate(evs, 0.5 + step * 6 + 0.5)
        worst_k = max(min_between(keys, a, b) for a, b in zip(evs, evs[1:]))
        worst_f = max(min_between(fingers, a, b) for a, b in zip(evs, evs[1:]))
        check("key rises between repeats @%s" % label, worst_k < 0.5, "min=%.2f" % worst_k)
        check("finger rises between repeats @%s" % label, worst_f < 0.5, "min=%.2f" % worst_f)
        check("counters back to 0 @%s" % label,
              vk._key_cnt.get("k60") == 0 and not any(p.active.values()))

    # audio times must not change
    evs = [ev(0.5 + 0.5 * i, 0.5) for i in range(3)]
    before = [e.t_off for e in evs]
    _apply_visual_release(evs)
    check("audio t_off untouched", [e.t_off for e in evs] == before)
    check("v_off earlier than the next strike by >= VIS_LIFT",
          all(a.v_off <= b.t_on - PRESS_LEAD - 0.9 * VIS_LIFT for a, b in zip(evs, evs[1:])))
    check("last note keeps its full length", evs[-1].v_off == evs[-1].t_off)

    # chord mates (same onset) and different pitches are left alone
    chord = [ev(1.0, 1.0, midi=60, finger=1), ev(1.0, 1.0, midi=64, finger=3)]
    _apply_visual_release(chord)
    check("chord untouched", all(e.v_off == e.t_off for e in chord))

    # a long held note is not shortened by a different finger on another key
    held = [ev(1.0, 3.0, midi=60, finger=1), ev(2.0, 0.5, midi=64, finger=3)]
    _apply_visual_release(held)
    check("held note keeps its length", held[0].v_off == held[0].t_off)

    # v_off must never fall before the press itself
    evs = [ev(0.5 + 0.06 * i, 0.06) for i in range(5)]
    _apply_visual_release(evs)
    check("v_off after the press fires",
          all(e.v_off > e.t_on - PRESS_LEAD for e in evs))

    print("\n%d failure(s)" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
