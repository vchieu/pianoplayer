"""Space pause + tempo popup tests (plan 2026-10-16-bug5-pause-tempo, B/C).

Headless only: offscreen Plotter, playsounds=False, fake key objects --
no real window, no dialog, no audio device.

    python test_pause_tempo.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import note, stream, tempo

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.vkeyboard import VirtualKeyboard

FAILS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class FakeKey:
    """Stands in for the VTK interactor in _on_keypress (B4: no window)."""
    def __init__(self, sym):
        self._sym = sym

    def GetKeySym(self):
        return self._sym


def make_score(n=8, ql=0.5, hold=0.5, bpm=120):
    pitches = [60, 64, 67, 72, 67, 64, 60, 64]
    p = stream.Part()
    p.insert(0, tempo.MetronomeMark(number=bpm))
    for i in range(n):
        p.insert(i * ql, note.Note(pitches[i % len(pitches)],
                                   quarterLength=hold))
    s = stream.Score()
    s.insert(0, p)
    return s


def make_vk(n=8, tempo_bpm=120.0):
    h = Hand("right", "M")
    h.verbose = False
    h.noteseq = reader(make_score(n, bpm=tempo_bpm), beam=0)
    h.generate(1, 1000)
    vk = VirtualKeyboard(songname="pausetest", tempo=tempo_bpm, offscreen=True)
    vk.verbose = False
    vk.playsounds = False
    vk.build_RH(h)
    vk._prepare()
    vk.engine = None          # no audio device; _pump skips scheduling
    vk._playing = True
    return vk


# ---- B: pause -----------------------------------------------------------
def test_space_toggles_pause():
    vk = make_vk()
    check(vk._paused is False, "starts unpaused")
    vk._on_keypress(FakeKey("space"))
    check(vk._paused is True, "space pauses")
    vk._on_keypress(FakeKey("space"))
    check(vk._paused is False, "space resumes")
    vk._playing = False


def test_other_keys_untouched():
    vk = make_vk()
    vk._on_keypress(FakeKey("q"))       # quit keys fall through to vedo
    vk._on_keypress(FakeKey("Escape"))
    check(vk._paused is False, "q/Esc must not toggle pause")
    vk._playing = False


def test_v_still_flips_view():
    vk = make_vk()
    before = getattr(vk, "_view_persp", False)
    vk._on_keypress(FakeKey("v"))
    check(getattr(vk, "_view_persp", False) is not before, "v must still flip")
    vk._playing = False


def test_pause_shifts_t0():
    """Resume shifts _t0 (and _audio_epoch) by the paused span: score time
    freezes, so `now = t - _t0` continues where it stopped (B2)."""
    vk = make_vk()
    t0, epoch = vk._t0, vk._audio_epoch
    vk.toggle_pause()
    check(vk._paused is True, "paused flag")
    at_pause = time.perf_counter()     # score time the pause began at
    time.sleep(0.3)                    # a real paused span
    vk.toggle_pause()
    check(vk._paused is False, "resumed flag")
    d = vk._t0 - t0
    check(abs(d - 0.3) < 0.10, "_t0 shifted by paused span: %g" % d)
    check(abs((epoch - vk._audio_epoch) - d) < 1e-9,
          "epoch shifted with _t0")
    # score time is continuous across the pause
    s_before = at_pause - t0
    s_after = time.perf_counter() - vk._t0
    check(abs(s_after - s_before) < 0.05,
          "now continuous across pause: %.3f -> %.3f" % (s_before, s_after))
    vk._playing = False


def test_pause_no_double_fire():
    """Frozen frames (same `now`, as _run_frames renders while paused) fire
    nothing new; after resume every note fires exactly once (B2)."""
    vk = make_vk()
    p = vk._players["R"]
    total = len(p.notes)
    check(total > 0, "test score has notes")
    now, dt = -2.4, 1.0 / 60.0
    while now < 1.0:                   # play a little
        vk._frame(now, dt)
        now += dt
    fired_mid = sum(1 for e in p.notes if e.fired)
    fire_mid = p.fire
    check(fired_mid > 0, "some notes fired before pause")
    vk.toggle_pause()
    for _ in range(10):                # frozen frame, score time stands still
        vk._frame(now, dt)
    check(p.fire == fire_mid, "frozen frame fires nothing new")
    check(sum(1 for e in p.notes if e.fired) == fired_mid, "no double-fire")
    vk._pause_started -= 0.25
    vk.toggle_pause()
    while now < vk._total + 0.2:       # resume to the end
        vk._frame(now, dt)
        now += dt
    check(p.fire == total, "all %d notes fired, got %d" % (total, p.fire))
    check(all(e.fired for e in p.notes), "every note fired exactly once")
    vk._playing = False


def test_hud_paused_and_bpm():
    vk = make_vk()
    vk._hud_t = -1e9
    vk._hud_update(0.0)
    txt = vk._hud.text() if vk._hud else ""
    check("120 bpm" in txt, "HUD shows score BPM, got %r" % txt)
    check("PAUSED" not in txt, "no PAUSED while playing, got %r" % txt)
    vk.toggle_pause()
    vk._hud_t = -1e9
    vk._hud_update(0.0)
    txt = vk._hud.text() if vk._hud else ""
    check("PAUSED" in txt, "HUD shows PAUSED, got %r" % txt)
    vk.toggle_pause()
    vk._playing = False


# ---- C: tempo -----------------------------------------------------------
def test_tempo_math():
    vk = make_vk(tempo_bpm=120.0)
    got = vk.set_tempo_bpm(60.0, 120.0)
    check(abs(vk.speedfactor - 0.5) < 1e-12, "entered/original ratio")
    check(abs(got - 60.0) < 1e-9 and abs(vk.effective_bpm() - 60.0) < 1e-9,
          "effective BPM 60, got %r" % got)
    got = vk.set_tempo_bpm(180.0)      # default original = score tempo
    check(abs(vk.speedfactor - 1.5) < 1e-12, "original defaults to score BPM")
    for bad in (0, -10, None, "x"):
        sf = vk.speedfactor
        check(vk.set_tempo_bpm(bad, 120.0) is None, "reject %r" % (bad,))
        check(vk.speedfactor == sf, "rate unchanged on %r" % (bad,))
    vk._playing = False


def test_tempo_flows_through_speedfactor():
    """No parallel tempo variable: _beat_seconds (the A/V sync path) slows
    down exactly by the ratio (C4)."""
    vk = make_vk(tempo_bpm=120.0)
    before = vk._beat_seconds(4.0)
    vk.set_tempo_bpm(60.0, 120.0)
    check(abs(vk._beat_seconds(4.0) - 2.0 * before) < 1e-9,
          "half tempo = double beat seconds")
    vk._playing = False


def test_popup_skipped_offscreen():
    vk = make_vk()                     # offscreen=True
    sf = vk.speedfactor
    check(vk.prompt_tempo() is None, "offscreen skips popup")
    check(vk.speedfactor == sf, "rate unchanged when skipped")
    vk._playing = False


def test_popup_skipped_env():
    """PIANOPLAYER_TEMPO_POPUP=0 skips before any Tk object is touched
    (safe with offscreen temporarily off: still no window)."""
    vk = make_vk()
    old = os.environ.get("PIANOPLAYER_TEMPO_POPUP")
    os.environ["PIANOPLAYER_TEMPO_POPUP"] = "0"
    vk.offscreen = False
    try:
        sf = vk.speedfactor
        check(vk.prompt_tempo() is None, "env=0 skips popup")
        check(vk.speedfactor == sf, "rate unchanged when skipped")
    finally:
        vk.offscreen = True
        if old is None:
            del os.environ["PIANOPLAYER_TEMPO_POPUP"]
        else:
            os.environ["PIANOPLAYER_TEMPO_POPUP"] = old
    vk._playing = False


def test_cli_tempo_flag():
    """--tempo parses (help lists it) and a --tempo CLI run without -v
    needs no window (C2 smoke)."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "pianoplayer"),
                        "--help"], capture_output=True, text=True,
                       cwd=HERE, timeout=120)
    check(r.returncode == 0, "--help exits 0")
    check("--tempo" in r.stdout, "--tempo documented in --help")
    score = os.path.join(HERE, "scores", "test_scales.xml")
    r = subprocess.run([sys.executable, os.path.join(HERE, "bin", "pianoplayer"),
                        score, "-n1", "--tempo", "90", "--quiet"],
                       capture_output=True, text=True, cwd=HERE, timeout=300)
    check(r.returncode == 0, "--tempo CLI run exits 0: %s" % r.stderr[-500:])


# ---------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception:
            FAILS.append(t.__name__)
            print("FAIL", t.__name__)
            import traceback
            traceback.print_exc()
    print("\n%d/%d passed" % (len(tests) - len(FAILS), len(tests)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
