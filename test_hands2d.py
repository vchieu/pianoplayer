"""
Headless acceptance test for the 2D hand render backend (`--hands 2d`).

Builds both hands from a score, prepares the player in 2D mode (no `HandRig`
is created), drives ~4 s of frames and checks:

  1. `vk.RH is None and vk.LH is None`              - no 3D rig in 2D mode
  2. the `Hands2D` overlay exists and is in the renderer
  3. the overlay really paints the hands (non-zero alpha) while playing
  4. alignment: a fingertip on a pressed key sits on painted pixels and its
     world x equals the key's world x (< 0.35 * KEY_WB)
  5. both skins (`cartoon`, `skeleton`) draw without error
  6. `toggle_view()` hides / shows the overlay
  7. screenshots in ./test_player2d_out/
  8. the overlay band crop never clips: the top rows of the band stay
     transparent on every driven frame, and the mapping helpers round-trip

No audio device, no window: `playsounds=False`, `offscreen=True`.

    python test_hands2d.py [score.xml] [seconds]
"""
import os
import sys
import time

from music21 import converter

from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import KEY_WB, score_tempo
from pianoplayer.vkeyboard import FPS_MS, LEAD_IN, VirtualKeyboard, _key_x

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_player2d_out")
DEFAULT_SCORES = ("scores/test_scales.xml", "scores/bach_invention4.xml")
STYLES = ("cartoon", "skeleton")


def pick_score():
    for s in DEFAULT_SCORES:
        if os.path.exists(os.path.join(HERE, s)):
            return s
    return DEFAULT_SCORES[0]


def build(fn, size="M", nmeasures=1000):
    sf = converter.parse(fn)
    tempo = score_tempo(sf)
    out = []
    for beam, side in ((0, "right"), (1, "left")):
        h = Hand(side, size)
        h.verbose = False
        h.noteseq = reader(sf, beam=beam)
        h.generate(0, nmeasures)
        out.append(h)
    return sf, tempo, out[0], out[1]


def main(fn=None, seconds=4.0):
    fn = fn or pick_score()
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".png"):
            os.remove(os.path.join(OUT, f))

    sf, tempo, rh, lh = build(fn)
    print("score", fn, "tempo", tempo, "| RH notes", len(rh.noteseq),
          "fingers", len(rh.fingerseq), "| LH notes", len(lh.noteseq),
          "fingers", len(lh.fingerseq))

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            ok = False
        return bool(cond)

    # ---- A9: the default backend is 2d, PIANOPLAYER_HANDS / --hands override
    vk0 = VirtualKeyboard(songname="defaults", offscreen=True)
    check(vk0.hands == "2d", "default hand backend is 2d (got %r)" % vk0.hands)
    vk0.hands = "3d"
    check(vk0.hands == "3d", "--hands 3d selects the 3D backend")

    vk = VirtualKeyboard(songname=fn, tempo=tempo, offscreen=True, hands="2d")
    vk.verbose = False
    vk.playsounds = False          # no audio device in this test
    vk.falling = False             # isolate the hand layer
    vk._fall = None
    vk.volume = 0.0
    vk.speedfactor = 1.0
    vk.build_RH(rh)
    vk.build_LH(lh)

    # ---- check 1: no 3D rig in 2D mode
    check(vk.RH is None and vk.LH is None,
          "2D mode builds no HandRig (RH=%r LH=%r)" % (vk.RH, vk.LH))

    t_prep = time.perf_counter()
    vk._prepare()
    print("prepare took %.2f s (cold pose/draw is paid here)" % (time.perf_counter() - t_prep))

    # ---- check 2: the overlay is there and in the renderer
    h2 = vk._hands2d
    check(h2 is not None, "a Hands2D overlay was built")
    if h2 is None:
        print("RESULT: FAILED")
        return 1
    check(h2.in_renderer(), "the overlay actor is in the scene renderer")
    check(h2.actor is not None and h2.actor.GetVisibility() == 1,
          "the overlay is visible in the top-down view")
    check(sorted(h2.anims) == ["L", "R"],
          "one 2D animator per hand that has notes (%s)" % sorted(h2.anims))

    # ---- drive the frame loop, wall-clock paced exactly like _on_timer -----
    kb = h2.kb
    geom = h2.geom
    band_h = h2.band_top_y - geom["y0"]
    print("overlay: %dx%d px, %.2f px/cm, world x %.1f..%.1f, band y %.1f..%.1f "
          "cm (%.0f%% of the %.0f cm window)"
          % (h2.size[0], h2.size[1], geom["ppc"], geom["x0"], geom["x1"],
             geom["y0"], h2.band_top_y, 100.0 * band_h / (geom["y1"] - geom["y0"]),
             geom["y1"] - geom["y0"]))
    # the band's top row must stay transparent on every driven frame, or the
    # crop in Hands2D (BAND_TOP_Y) would be clipping a hand
    margin_px = max(4, int(round(0.02 * h2.size[1])))
    top_margin_px = h2.size[1]
    alpha_frames = 0
    checked_align = []
    shots = {1.0: "t1", 2.5: "t3"}
    t = -LEAD_IN
    dt = FPS_MS / 1000.0
    frames = 0
    while t < seconds:
        vk._frame(t, dt)
        vk.vp.render()
        frames += 1
        alpha = h2.alpha_pixels()
        lit = int((alpha > 0).sum())
        if lit > 0:
            alpha_frames += 1
            rows = (alpha > 0).any(axis=1).nonzero()[0]
            top_margin_px = min(top_margin_px, int(rows[0]))
        # ---- check 4: a fingertip on a pressed key, on painted pixels
        now = t
        for p in vk._players.values():
            for e in p.held:
                if e.midi is None or not e.finger:
                    continue
                anim = h2.anims.get(p.side)
                if anim is None:
                    continue
                pose = anim.pose(now, kb)
                tips = [(j[0], j[1]) for j in pose["joints"] if j[1] == "tip"]
                if not tips:
                    continue
                key_px = kb.key_rects[e.midi].centerx
                tip = min(tips, key=lambda j: abs(j[0][0] - key_px))[0]
                # the tip's own surface pixel: the 2D model draws a point q at
                # (q[0], front_y - q[1]), so these are band column / row
                col = int(round(tip[0]))
                row = int(round(pose["front_y"] - tip[1]))
                if not (0 <= col < h2.size[0] and 0 <= row < h2.size[1]):
                    continue
                wx = h2.world_x_of_col(col)              # its world x
                wy = h2.world_y_of_row(row)             # and its world y
                # anti-aliased edges: the exact pixel or a 3x3 window around it
                win = alpha[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
                checked_align.append({
                    "side": p.side, "midi": e.midi, "finger": e.finger,
                    "dx_cm": abs(wx - _key_x(e.midi)),
                    "wy_cm": wy,
                    "alpha_px": int(win.max()),
                    "px": (col, row),
                })
        for mark, tag in list(shots.items()):
            if t >= mark:
                vk.vp.screenshot(os.path.join(OUT, "%s.png" % tag))
                del shots[mark]
        # next frame: score time follows the wall clock (see _on_timer)
        tn = time.perf_counter() - vk._t0
        dt = tn - t
        if dt <= 0.0 or dt > 0.25:
            dt = FPS_MS / 1000.0
        t = tn
    vk.vp.screenshot(os.path.join(OUT, "t_end.png"))

    print("frames driven: %d in %.1f s of score (%.1f fps)"
          % (frames, seconds, frames / max(seconds, 1e-6)))

    # ---- check 3: the overlay paints the hands
    check(alpha_frames > 0, "the overlay painted hands on %d/%d sampled frames"
          % (alpha_frames, frames))
    check(alpha_frames >= max(1, frames // 4),
          "the overlay painted hands on most frames")

    # ---- check 4: alignment
    if not check(len(checked_align) > 0,
                 "sampled at least one pressed fingertip"):
        checked_align = []
    worst = 0.0
    bad = 0
    for rec in checked_align:
        worst = max(worst, rec["dx_cm"])
        if rec["dx_cm"] > 0.35 * KEY_WB or rec["alpha_px"] <= 0:
            bad += 1
    if checked_align:
        print("fingertip samples: %d, worst |tip_x - key_x| %.3f cm "
              "(limit %.3f), painted samples %d"
              % (len(checked_align), worst, 0.35 * KEY_WB,
                 sum(1 for r in checked_align if r["alpha_px"] > 0)))
        for rec in checked_align[:4]:
            print("   side %s midi %d finger %d dx %.3f cm wy %+.2f cm "
                  "alpha %d at col/row %s"
                  % (rec["side"], rec["midi"], rec["finger"], rec["dx_cm"],
                     rec["wy_cm"], rec["alpha_px"], rec["px"]))
    check(worst < 0.35 * KEY_WB,
          "every sampled fingertip is on its key (worst %.3f cm < %.3f)"
          % (worst, 0.35 * KEY_WB))
    check(bad == 0, "every sampled fingertip sits on painted overlay pixels "
                    "(%d bad)" % bad)

    # ---- check 8: the band crop never clips a hand
    check(top_margin_px > margin_px,
          "the top %d rows of the band stayed transparent on every frame "
          "(hands start at row %d of %d = world y %+.2f cm, band top %.1f cm)"
          % (margin_px, top_margin_px, h2.size[1], h2.world_y_of_row(top_margin_px),
             h2.band_top_y))
    # the mapping helpers must be exact inverses (the test indexes pixels with
    # them, so a stale formula would silently check the wrong pixel)
    probe = (-18.0, -8.0, 0.0, 4.0, h2.band_top_y)
    check(all(abs(h2.world_y_of_row(h2.py_of_world_y(y)) - y) < 1e-6
              for y in probe),
          "py_of_world_y / world_y_of_row round-trip exactly")
    check(abs(h2.px_of_world_x(h2.world_x_of_col(500)) - 500) < 1e-6,
          "px_of_world_x / world_x_of_col round-trip exactly")

    # ---- check 6: the perspective view hides the flat overlay
    vk.toggle_view()
    check(h2.actor.GetVisibility() == 0,
          "toggle_view() hides the overlay in the perspective view")
    vk.toggle_view()
    check(h2.actor.GetVisibility() == 1, "toggling back shows the overlay")

    # ---- check 5: both skins draw
    notes = {side: p.notes for side, p in vk._players.items()}
    for style in STYLES:
        try:
            h2.build(notes, style=style)
            h2.update(t, dt)
            lit = int((h2.alpha_pixels() > 0).sum())
            check(lit > 0, "style %r draws the hands (%d lit pixels)" % (style, lit))
        except Exception as exc:                 # pragma: no cover
            import traceback
            traceback.print_exc()
            check(False, "style %r raised %s: %s" % (style, type(exc).__name__, exc))
    h2.build(notes, style="cartoon")

    # ---- check 7: screenshots
    shots_written = sorted(f for f in os.listdir(OUT) if f.endswith(".png"))
    check("t1.png" in shots_written and "t3.png" in shots_written
          and "t_end.png" in shots_written,
          "screenshots written to test_player2d_out/: %s" % shots_written)

    vk._stop_audio()
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    score = sys.argv[1] if len(sys.argv) > 1 else None
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    sys.exit(main(score, secs))