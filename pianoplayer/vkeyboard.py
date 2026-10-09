#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Name:         VirtualKeyboard
# Purpose:      Animated 3D piano player: procedural hands + piano synthesiser
# URL:          https://github.com/marcomusy/pianoplayer
# Author:       Marco Musy
# -----------------------------------------------------------------------------
"""
`VirtualKeyboard` renders the keyboard and plays a score with animated,
anatomically posed hands while a built-in additive piano synth renders the
sound (no external sample files are needed).

The scene is driven by a vedo timer: every frame the play head advances,
pending note events are fired (key dip, finger press, scheduled sound) and
both hands are re-posed with their analytic IK rigs.

Typical use::

    vk = VirtualKeyboard(songname="bach.xml")
    vk.build_RH(rh)          # a pianoplayer.hand.Hand with .generate() run
    vk.build_LH(lh)
    vk.speedfactor = 1.5
    vk.play()                # blocks until the window is closed
"""
from __future__ import annotations

import math
import os
import time as _time

try:
    from vedo import Box, Disc, Plotter, Text2D, Text3D
except Exception:  # pragma: no cover - vedo is mandatory at runtime
    pass

try:
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkRenderingCore import vtkLight
except Exception:  # pragma: no cover
    vtkMatrix4x4 = None
    vtkLight = None

from pianoplayer import __version__
from pianoplayer.audio import SoundEngine, midi_from_name, midi_of
from pianoplayer.hand3d import FINGERS, HandRig
from pianoplayer.utils import (
    BLACK_KEY_PRESS_ANGLE,
    KEYBSIZE,
    KEY_PIVOT_Y,
    KEY_PRESS_ANGLE,
    KEY_WB,
    key_is_black,
    nameof,
)

__all__ = ["VirtualKeyboard"]


# --------------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------------- #
NR_OCTAVES = 7
KEY_TOL = 0.12
FPS_MS = 16               # animation period [ms]
LEAD_IN = 0.8             # s before the first note: hands settle in place
AIM_LEAD = 0.25           # s ahead at which the hand aims at the next note
AUDIO_LOOKAHEAD = 0.30    # s of audio events scheduled in advance
BLACK_AIM = 0.45          # s before a black-key note the hand slides forward
TAU_PRESS = 0.012         # key/finger easing while pressing [s]
TAU_RELEASE = 0.032       # ... while releasing [s]
HUD_PERIOD = 0.25         # s between HUD refreshes

KEY_WHITE = (0.94, 0.94, 0.91)
KEY_BLACK = (0.055, 0.055, 0.065)
ACCENT_R = (1.00, 0.45, 0.18)    # flash of a key played by the right hand
ACCENT_L = (0.72, 0.40, 1.00)    # ... left hand
WOOD = (0.46, 0.29, 0.17)
WOOD_DARK = (0.33, 0.20, 0.12)
DESK = (0.88, 0.85, 0.77)
BAR_COLOR = (1.00, 0.74, 0.20)
HUD_COLOR = (0.22, 0.17, 0.12)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _fmt_time(t):
    t = max(0.0, float(t))
    return "%d:%02d" % (int(t // 60), int(t % 60))


def _lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def _wood_texture(w=512, h=160, base=WOOD):
    """Procedural plank texture, no image files required."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return None
    rng = np.random.default_rng(11)
    X, Y = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    n = rng.normal(0, 1, (h, w))
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    k /= k.sum()
    for _ in range(3):
        n = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, n)
    rings = np.sin(X * 0.085 + 5.0 * np.sin(Y * 0.045 + 0.02 * n) + 0.30 * n)
    streak = np.sin(Y * 0.6 + 0.8 * n)
    shade = 0.86 + 0.12 * rings + 0.05 * streak
    img = np.empty((h, w, 3), dtype=float)
    for i in range(3):
        img[..., i] = base[i] * shade
    return (np.clip(img, 0.0, 1.0) * 255.0).astype("uint8")


def _write_key_matrix(m4, amount, black):
    """Matrix hinging a key around its back edge (line y = KEY_PIVOT_Y)."""
    ang = math.radians((BLACK_KEY_PRESS_ANGLE if black else KEY_PRESS_ANGLE)
                       * amount)
    c, s = math.cos(ang), math.sin(ang)
    py = KEY_PIVOT_Y
    pz = 1.0 if black else 0.0
    rows = ((1.0, 0.0, 0.0, 0.0),
            (0.0, c, -s, py - c * py + s * pz),
            (0.0, s, c, pz - s * py - c * pz),
            (0.0, 0.0, 0.0, 1.0))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            m4.SetElement(i, j, float(v))
    m4.Modified()


# --------------------------------------------------------------------------- #
# per note / per hand bookkeeping
# --------------------------------------------------------------------------- #
class _Ev:
    __slots__ = ("t_on", "t_off", "midi", "vel", "finger", "black",
                 "measure", "name", "fx", "key", "fired")

    def __init__(self, t_on, t_off, midi, vel, finger, black, measure,
                 name, fx):
        self.t_on = t_on
        self.t_off = t_off
        self.midi = midi
        self.vel = vel
        self.finger = finger        # 1..5, 0 = unknown/unused
        self.black = black
        self.measure = measure
        self.name = name            # e.g. "C#3"
        self.fx = fx                # {finger: world x} fan, or None
        self.key = None             # 3D key mesh name, filled on press
        self.fired = False


class _Player:
    def __init__(self, side, sign, pan, hand, rig):
        self.side = side            # "R" / "L"
        self.sign = sign            # +1 right, -1 left
        self.pan = pan              # stereo pan of the sound engine
        self.hand = hand
        self.rig = rig
        self.notes = []
        self.fire = 0               # next note to be pressed
        self.aim = 0                # note the hand is heading towards
        self.sched = 0              # next note to be handed to the synth
        self.held = []              # fired notes still sounding
        self.press = {f: 0.0 for f in FINGERS}
        self.active = {f: 0 for f in FINGERS}
        self.finger_x = {}


# --------------------------------------------------------------------------- #
# the keyboard
# --------------------------------------------------------------------------- #
class VirtualKeyboard:
    """3D keyboard with animated hands and synthesised piano sound."""

    def __init__(self, songname='', tempo=72.0, offscreen=False, shadows=False):
        self.KB = {}                # note name -> key mesh
        self.offscreen = bool(offscreen)
        # experimental: vtkShadowMapPass (hands cast shadows on the keys).
        # The shadow frustum is fitted to the scene, on wide screens some
        # surfaces fall outside it and turn black, so it is off by default.
        self.shadows = bool(shadows)
        self.vp = None
        self.rightHand = None
        self.leftHand = None
        self.RH = None              # HandRig
        self.LH = None
        self.vpRH = None            # legacy aliases
        self.vpLH = None
        self.playsounds = True
        self.verbose = True
        self.songname = songname
        self.tempo = float(tempo)   # beats per minute of the score
        self.speedfactor = 1.0      # playback rate (image + sound)
        self.volume = 0.8
        self.t0 = 0                 # legacy attributes
        self.dt = 0.1
        self.engagedfingersR = [False] * 6
        self.engagedfingersL = [False] * 6
        self.engagedkeysR = []
        self.engagedkeysL = []

        self.engine = None
        self._audio_epoch = -LEAD_IN   # score time of engine time 0
        self._players = {}
        self._hand_shadows = {}
        self._key_black = {}
        self._key_by_midi = {}
        self._key_amt = {}
        self._key_tgt = {}
        self._key_cnt = {}
        self._key_col = {}
        self._key_acc = {}
        self._key_mats = {}
        self._touched = set()
        self._playing = False
        self._ended = False
        self._cb_id = None
        self._total = 1.0
        self._prog = -1.0
        self._hud = None
        self._hud_t = -1e9
        self._err = False
        self._camera = None
        self._span = 0.0
        self._bar = None
        self._bar_m4 = None

        self.build_keyboard()

    # ------------------------------------------------------------------ screen
    @staticmethod
    def _screen_size():
        """Screen resolution in pixels (best effort)."""
        try:
            from vtkmodules.vtkRenderingCore import vtkRenderWindow
            s = vtkRenderWindow.GetScreenSize()
            if int(s[0]) > 0 and int(s[1]) > 0:
                return int(s[0]), int(s[1])
        except Exception:
            pass
        try:
            import tkinter as tk
            root = getattr(tk, "_default_root", None)
            own = False
            if root is None:
                root = tk.Tk()
                root.withdraw()
                own = True
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            if own:
                root.destroy()
            return int(sw), int(sh)
        except Exception:
            return 1920, 1080

    def _window_size(self):
        """A player window that always fits the screen."""
        sw, sh = self._screen_size()
        w = int(min(1560, max(900, sw - 24)))
        h = int(min(760, max(520, int(sh * 0.86))))
        return w, h

    # ------------------------------------------------------------------ scene
    def build_keyboard(self):
        """Create the plotter, the wooden frame and the 88 keys (non blocking)."""
        span = NR_OCTAVES * KEY_WB * 7.0
        self._span = span

        w, h = self._window_size()
        self.vp = Plotter(title='PianoPlayer ' + __version__, axes=0,
                          size=(w, h), bg='cornsilk', bg2='lb',
                          offscreen=self.offscreen)

        wood = _wood_texture()
        fallboard = Box(pos=(span / 2 + KEYBSIZE, 6, 1),
                        length=span + 1, height=3, width=5, c=WOOD)
        base = Box(pos=(span / 2 + KEYBSIZE, 0, -1),
                   length=span + 1, height=1, width=17, c=WOOD_DARK)
        if wood is not None:
            try:
                fallboard.texture(wood, repeat=True)
                base.texture(wood, repeat=True)
            except Exception:
                pass
        self.vp += fallboard
        self.vp += base

        # music desk with the song title ------------------------------------
        desk_c = (span / 1.6, 9.5, 11.0)
        desk = Box(pos=desk_c, length=span / 2.4,
                   height=15.0, width=1.4, c=DESK)
        desk.rotate(-20, axis=(1, 0, 0), point=desk_c)
        self.vp += desk
        if self.songname:
            # NB: vedo reads _.._ as subscript markup -> strip the underscores
            title = Text3D(os.path.basename(str(self.songname)).replace("_", " "),
                           s=2.4, c=(0.16, 0.13, 0.11), depth=0.25)
            b = title.bounds()
            title.rotate(70, axis=(1, 0, 0),
                         point=((b[0] + b[1]) / 2, (b[2] + b[3]) / 2,
                                (b[4] + b[5]) / 2))
            title.pos(span / 1.6, 8.6, 11.5)
            self.vp += title

        # keys ---------------------------------------------------------------
        nts = ("C", "D", "E", "F", "G", "A", "B")
        for ioct in range(NR_OCTAVES):
            octave = ioct + 1
            for ik, nt in enumerate(nts):
                x = ik * KEY_WB + (ioct + 1) * KEYBSIZE + KEY_WB / 2
                name = nt + str(octave)
                k = Box(pos=(x, -2, 0), length=KEY_WB - KEY_TOL,
                        height=1, width=12, c=KEY_WHITE)
                self.KB[name] = k
                self._key_black[name] = False
                self.vp += k
                self._register_midi(name, nt, octave)
                if nt not in ("E", "B"):
                    nb = Box(pos=(x + KEY_WB / 2, 0, 1), length=KEY_WB * 0.6,
                             height=1, width=8, c=KEY_BLACK)
                    nname = nt + '#' + str(octave)
                    self.KB[nname] = nb
                    self._key_black[nname] = True
                    self.vp += nb
                    self._register_midi(nname, nt + '#', octave)

        # progress bar, a strip running on top of the fallboard --------------
        if vtkMatrix4x4 is not None:
            self._bar = Box(pos=(0, 0, 0), length=1, height=0.10, width=3.2,
                            c=BAR_COLOR)
            m4 = vtkMatrix4x4()
            m4.Identity()
            self._bar.actor.SetUserMatrix(m4)
            self._bar_m4 = m4
            self.vp += self._bar

        # camera: fit the whole keyboard whatever the window aspect is.
        # a 37 deg down angle keeps world +z vertical and +x horizontal on
        # screen, so the keyboard never looks tilted.
        cx = span / 2 + KEYBSIZE
        aspect = float(w) / float(max(h, 1))
        dist = (span / 2.0 + 8.0) / (math.tan(math.radians(15.0)) * aspect)
        down = math.radians(37.0)
        focal = (cx, -2.0, 3.0)
        pos = (cx, focal[1] - dist * math.cos(down),
               focal[2] + dist * math.sin(down))
        self._camera = dict(pos=pos, focalPoint=focal,
                            viewup=(0.0, math.sin(down), math.cos(down)))
        self.vp.camera = self._camera
        self._add_lights()
        if self.shadows:
            self._enable_shadows()
        self.vp.render()
        return self.vp

    def _enable_shadows(self):
        """Let the hands and the desk cast shadows on the keys (best effort)."""
        try:
            from vtkmodules.vtkRenderingOpenGL2 import (
                vtkCameraPass,
                vtkRenderPassCollection,
                vtkSequencePass,
                vtkShadowMapBakerPass,
                vtkShadowMapPass,
            )
            shadows = vtkShadowMapPass()
            baker = shadows.GetShadowMapBakerPass()
            if baker is None:
                baker = vtkShadowMapBakerPass()
                shadows.SetShadowMapBakerPass(baker)
            passes = vtkRenderPassCollection()
            passes.AddItem(baker)
            passes.AddItem(shadows)
            seq = vtkSequencePass()
            seq.SetPasses(passes)
            cam = vtkCameraPass()
            cam.SetDelegatePass(seq)
            self.vp.renderer.SetPass(cam)
        except Exception as exc:
            self.shadows = False
            if self.verbose:
                print("shadows unavailable:", exc)

    def _register_midi(self, keyname, pitchname, octave):
        try:
            self._key_by_midi[int(midi_from_name(pitchname, octave))] = keyname
        except Exception:
            pass

    def _add_lights(self):
        if vtkLight is None or self.vp is None or self.vp.renderer is None:
            return
        cx = self._span / 2 + KEYBSIZE
        specs = (
            ((cx - 45, -70, 85), (cx, 0, 0), 0.90, (1.00, 0.97, 0.92)),
            ((cx + 55, -50, 60), (cx, 0, 0), 0.50, (0.86, 0.92, 1.00)),
            ((cx, 55, 70), (cx, 0, 6), 0.40, (1.00, 1.00, 1.00)),
        )
        # with the shadow map pass only one light can be used: every light
        # must own a shadow map, otherwise its contribution is dropped.
        if self.shadows:
            specs = specs[:1]
        try:
            for pos, foc, inten, col in specs:
                lt = vtkLight()
                lt.SetLightTypeToSceneLight()
                lt.SetPosition(*[float(v) for v in pos])
                lt.SetFocalPoint(*[float(v) for v in foc])
                lt.SetIntensity(float(inten))
                lt.SetColor(*[float(v) for v in col])
                # the shadow map is built inside the light cone: widen it to
                # cover the whole keyboard (default is a narrow 30 deg)
                if self.shadows:
                    lt.SetConeAngle(95.0)
                self.vp.renderer.AddLight(lt)
            self.vp.renderer.LightFollowCameraOff()
        except Exception:
            pass

    # ------------------------------------------------------------------ hands
    def _make_shadow(self):
        """Soft contact shadow that follows a hand.

        Real shadow maps (see the `shadows` option) do not cover the whole
        keyboard, so a cheap flat blob is used instead: it is drawn just
        above the key tops, where a hand shadow would fall.
        """
        d = Disc(pos=(0, -5.0, 0.56), r1=0.001, r2=1.0, res=20,
                 c=(0.02, 0.02, 0.04), alpha=0.13)
        d.scale(8.5, 5.0, 1.0)
        try:
            d.actor.GetProperty().LightingOff()
        except Exception:
            pass
        self.vp.add(d)
        return d

    def build_RH(self, hand):
        """Attach a `pianoplayer.hand.Hand` (side='right') to the scene."""
        self.rightHand = hand
        self.RH = HandRig(side="right", size=getattr(hand, "size", "M"))
        self.RH.add_to(self.vp)
        self.vpRH = self.RH
        self._hand_shadows["R"] = self._make_shadow()
        return self

    def build_LH(self, hand):
        """Attach a `pianoplayer.hand.Hand` (side='left') to the scene."""
        self.leftHand = hand
        self.LH = HandRig(side="left", size=getattr(hand, "size", "M"))
        self.LH.add_to(self.vp)
        self.vpLH = self.LH
        self._hand_shadows["L"] = self._make_shadow()
        return self

    # ------------------------------------------------------------------ score
    def _events_for(self, hand, sign, pan):
        """Turn a Hand noteseq into a time ordered list of _Ev."""
        sc = 60.0 / (max(float(self.tempo), 1e-3)
                     * max(float(self.speedfactor), 1e-6))
        nseq = list(getattr(hand, "noteseq", []) or [])
        fseq = list(getattr(hand, "fingerseq", []) or [])
        # map noteseq index -> fan row (generate() may have processed only a
        # subset of the notes, e.g. when a measure range was requested)
        fan_of = {}
        idxs = list(getattr(hand, "fingerseq_index", []) or [])
        if fseq and len(idxs) == len(fseq):
            fan_of = {i2: row for i2, row in zip(idxs, fseq)}
        elif len(fseq) == len(nseq) and fseq:
            fan_of = dict(enumerate(fseq))
        weights = getattr(hand, "weights", [None, 1.1, 1.0, 1.1, 0.9, 0.8])
        frest = getattr(hand, "frest", [None, -7.0, -2.8, 0.0, 2.8, 5.6])

        evs = []
        for i, n in enumerate(nseq):
            try:
                wx = sign * float(n.x)
            except Exception:
                continue
            try:
                fing = int(n.fingering)
            except Exception:
                fing = 0
            if fing < 1 or fing > 5:
                fing = 0

            name = nameof(n)
            midi = None
            try:
                midi = int(midi_of(n))
            except Exception:
                try:
                    midi = int(midi_from_name(n.name, n.octave))
                except Exception:
                    midi = None

            try:
                t_on = float(n.time) * sc
            except Exception:
                t_on = 0.0
            try:
                dur = float(n.duration) * sc
            except Exception:
                dur = 0.0
            if dur <= 1e-4:
                dur = 0.25 * sc

            w = 1.0
            if fing:
                try:
                    w = float(weights[fing])
                except Exception:
                    w = 1.0

            # musical dynamics: the melody is louder than the accompaniment,
            # down beats are accented, the thumb is stronger than the little
            # finger, plus a small deterministic humanisation.
            try:
                beatpos = float(n.time)
            except Exception:
                beatpos = 0.0
            onbeat = 1.0 - min(abs(beatpos - round(beatpos)) * 3.0, 1.0)
            base = 0.58 if sign > 0 else 0.50
            human = (((i * 2654435761) % 97) / 97.0) - 0.5
            vel = base + 0.18 * onbeat + 0.10 * (w - 1.0) + 0.06 * human
            vel = min(1.0, max(0.20, vel))

            fx = None
            row = fan_of.get(i)
            if row is not None:
                try:
                    fx = {j: sign * float(row[j]) for j in (1, 2, 3, 4, 5)}
                except Exception:
                    fx = None
            if fx is None and fing:
                fx = {j: wx + sign * (float(frest[j]) - float(frest[fing]))
                      for j in (1, 2, 3, 4, 5)}

            evs.append(_Ev(t_on, t_on + dur, midi, vel, fing,
                           key_is_black(wx), int(getattr(n, "measure", 0) or 0),
                           name, fx))

        evs.sort(key=lambda e: e.t_on)

        # never let one pitch overlap itself (one synth voice per midi)
        nxt = {}
        for e in reversed(evs):
            if e.midi is not None:
                coming = nxt.get(e.midi)
                if coming is not None and coming < e.t_off:
                    e.t_off = coming
                nxt[e.midi] = e.t_on
        for e in evs:
            e.t_off = max(e.t_off, e.t_on + 0.06)
        return evs

    # -------------------------------------------------------------------- play
    def play(self):
        """
        Animate the score with sound.

        Blocks until the rendering window is closed (or the piece is over and
        the window is closed by the user).
        """
        if self.vp is None:
            self.build_keyboard()
        self._prepare()

        if self._cb_id is not None:
            try:
                self.vp.remove_callback(self._cb_id)
            except Exception:
                pass
            self._cb_id = None
        try:
            self._cb_id = self.vp.add_callback("timer", self._on_timer)
            self.vp.timer_callback("create", dt=FPS_MS)
        except Exception as exc:
            print("Cannot start the animation timer:", exc)

        self._playing = True
        try:
            self.vp.show(interactive=1, resetcam=0)
        finally:
            self._playing = False
            self._teardown()
        return self

    def _prepare(self):
        self._players = {}
        for side, sign, pan, attr, rigname in (
                ("R", 1.0, +0.30, "rightHand", "RH"),
                ("L", -1.0, -0.30, "leftHand", "LH")):
            hand = getattr(self, attr, None)
            rig = getattr(self, rigname, None)
            if hand is None or rig is None:
                continue
            p = _Player(side, sign, pan, hand, rig)
            p.notes = self._events_for(hand, sign, pan)
            if not p.notes:
                continue
            self._players[side] = p
            if p.notes[0].fx:
                rig.settle(p.notes[0].fx, {})

        self._stop_audio()
        engine_wall = _time.time()
        if self.playsounds:
            try:
                self.engine = SoundEngine(volume=self.volume, enabled=True)
                self.engine.start()
                if self.verbose and self.engine.backend_name != "none":
                    print("Audio backend:", self.engine.backend_name)
            except Exception as exc:
                print("Audio unavailable:", exc)
                self.engine = None

        for name in self._key_tgt:
            self._key_tgt[name] = 0.0
            self._key_amt[name] = 0.0
        self._key_cnt = {}
        self._touched = set()

        tot = 1.0
        for p in self._players.values():
            tot = max(tot, max(e.t_off for e in p.notes))
        self._total = tot
        self._prog = -1.0
        self._hud_t = -1e9
        self._ended = False
        self._t0 = _time.time() + LEAD_IN
        # the synth counts seconds from the engine start while the score
        # counts them from _t0: keep the offset, so that a note placed at
        # score time T lands at engine time (T - _audio_epoch).
        self._audio_epoch = engine_wall - self._t0
        self._last = _time.time()

    def _teardown(self):
        self._stop_audio()
        if self.verbose and self._ended:
            print("End of note sequence reached.")

    def _stop_audio(self):
        if self.engine is not None:
            try:
                self.engine.all_notes_off()
            except Exception:
                pass
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None

    # ------------------------------------------------------------------ events
    def _key_for(self, ev):
        if ev.midi is not None:
            k = self._key_by_midi.get(ev.midi)
            if k is not None:
                return k
        return self.KB.get(ev.name)

    def _press(self, p, ev):
        if ev.finger:
            p.active[ev.finger] = p.active.get(ev.finger, 0) + 1
        key = self._key_for(ev)
        if key is None:
            return
        ev.key = key
        self._key_cnt[key] = self._key_cnt.get(key, 0) + 1
        self._key_tgt[key] = 1.0
        self._key_acc[key] = ACCENT_R if p.side == "R" else ACCENT_L
        self._touched.add(key)

    def _release(self, p, ev):
        if ev.finger and p.active.get(ev.finger):
            p.active[ev.finger] -= 1
        key = ev.key
        if key is None:
            return
        c = self._key_cnt.get(key, 0) - 1
        self._key_cnt[key] = c
        if c <= 0:
            self._key_tgt[key] = 0.0

    def _pump(self, p, now, engine):
        notes = p.notes

        # hand the coming notes to the synthesiser --------------------------
        if engine is not None:
            while p.sched < len(notes) and notes[p.sched].t_on - now <= AUDIO_LOOKAHEAD:
                e = notes[p.sched]
                if e.midi is not None:
                    t_on = max(0.0, e.t_on - self._audio_epoch)
                    t_off = max(t_on + 0.02, e.t_off - self._audio_epoch)
                    engine.note_on(e.midi, e.vel, when=t_on, pan=p.pan)
                    engine.note_off(e.midi, when=t_off)
                p.sched += 1
        else:
            p.sched = len(notes)

        # what the hand is aiming at ----------------------------------------
        while p.aim + 1 < len(notes) and notes[p.aim + 1].t_on - AIM_LEAD <= now:
            p.aim += 1
        target = notes[p.aim] if p.aim < len(notes) else None
        if target is not None and target.fx:
            p.finger_x = target.fx

        # presses due --------------------------------------------------------
        while p.fire < len(notes) and notes[p.fire].t_on <= now:
            e = notes[p.fire]
            p.fire += 1
            if not e.fired:
                e.fired = True
                p.held.append(e)
                self._press(p, e)

        # releases due -------------------------------------------------------
        i = 0
        while i < len(p.held):
            e = p.held[i]
            if e.t_off <= now:
                self._release(p, e)
                p.held.pop(i)
            else:
                i += 1

        # is a black key coming up soon? -------------------------------------
        advance = bool(target is not None and target.black
                       and (target.t_on - now) < BLACK_AIM)
        return advance

    # ------------------------------------------------------------------ frame
    def _on_timer(self, evt=None):
        if not self._playing or self.vp is None:
            return
        try:
            t = _time.time()
            dt = t - self._last
            self._last = t
            if dt <= 0.0 or dt > 0.25:
                dt = FPS_MS / 1000.0
            now = t - self._t0
            self._frame(now, dt)
            self.vp.render()
        except Exception:
            if not self._err:
                self._err = True
                import traceback
                traceback.print_exc()

    def _frame(self, now, dt):
        engine = self.engine      # the synth clock runs even without a device
        for p in self._players.values():
            advance = self._pump(p, now, engine)

            for f in FINGERS:                       # ease the fingers
                tgt = 1.0 if p.active.get(f) else 0.0
                cur = p.press[f]
                if cur != tgt:
                    tau = TAU_PRESS if tgt > cur else TAU_RELEASE
                    newv = tgt + (cur - tgt) * math.exp(-dt / max(tau, 1e-4))
                    if abs(newv - tgt) < 0.004:
                        newv = tgt
                    p.press[f] = newv

            p.rig.update(dt, p.finger_x, p.press, advance_black=advance)

            sh = self._hand_shadows.get(p.side)
            if sh is not None:
                root = p.rig.root
                sh.pos(float(root[0]),
                       -5.0 + (float(root[1]) - HandRig.ROOT_Y), 0.56)

        self._sync_keys(dt)
        self._hud_update(now)
        self._progress(now)

        if not self._ended and now >= self._total:
            self._ended = True
            if self.verbose:
                print("End of note sequence reached.")

    # -------------------------------------------------------------- key motion
    def _sync_keys(self, dt):
        for name in self._touched:
            tgt = self._key_tgt.get(name, 0.0)
            cur = self._key_amt.get(name, 0.0)
            if cur == tgt:
                continue
            tau = TAU_PRESS if tgt > cur else TAU_RELEASE
            a = tgt + (cur - tgt) * math.exp(-dt / max(tau, 1e-4))
            if abs(a - tgt) < 0.004:
                a = tgt
            self._key_amt[name] = a

            mesh = self.KB[name]
            black = self._key_black[name]
            m4 = self._key_mats.get(name)
            if m4 is None and vtkMatrix4x4 is not None:
                m4 = vtkMatrix4x4()
                m4.Identity()
                mesh.actor.SetUserMatrix(m4)
                self._key_mats[name] = m4
            if m4 is not None:
                _write_key_matrix(m4, a, black)
                mesh.actor.Modified()

            base = KEY_BLACK if black else KEY_WHITE
            acc = self._key_acc.get(name) or base
            col = _lerp3(base, acc, a)
            prev = self._key_col.get(name)
            if prev is None or abs(prev[0] - col[0]) + abs(prev[1] - col[1]) \
                    + abs(prev[2] - col[2]) > 0.012:
                mesh.color(col)
                self._key_col[name] = col

    # --------------------------------------------------------------------- HUD
    def _hud_update(self, now):
        if now < self._hud_t + HUD_PERIOD or self.vp is None:
            return
        self._hud_t = now

        measure = "-"
        for p in self._players.values():
            if p.aim < len(p.notes):
                measure = str(p.notes[p.aim].measure)
        txt = "  %s / %s      measure %s      %d bpm  (speed x%g)  " % (
            _fmt_time(now), _fmt_time(self._total), measure,
            int(round(self.tempo)), float(self.speedfactor))

        if self._hud is None:
            try:
                self._hud = Text2D(txt, pos='bottom-left', s=0.75,
                                   c=HUD_COLOR, bg='w')
                self.vp.add(self._hud)
            except Exception:
                self._hud = False
        elif self._hud:
            try:
                self._hud.text(txt)
            except Exception:
                pass

        if self.verbose and self.engagedfingersR is not None:
            rp = self._players.get("R")
            if rp is not None:
                for f in FINGERS:
                    self.engagedfingersR[f - 1] = rp.active.get(f, 0) > 0
                self.engagedkeysR = [n for n in self._key_tgt
                                     if self._key_tgt[n] > 0]

    def _progress(self, now):
        if self._bar is None or self._bar_m4 is None:
            return
        prog = 0.0 if self._total <= 0 else min(1.0, max(0.0, now / self._total))
        if abs(prog - self._prog) < 0.0005:
            return
        self._prog = prog
        m4 = self._bar_m4
        length = max(prog * self._span, 1e-3)
        for i in range(4):
            for j in range(4):
                m4.SetElement(i, j, 0.0)
        m4.SetElement(0, 0, length)
        m4.SetElement(0, 3, KEYBSIZE + length / 2.0)
        m4.SetElement(1, 1, 1.0)
        m4.SetElement(1, 3, 6.0)       # strip sits on the fallboard top
        m4.SetElement(2, 2, 1.0)
        m4.SetElement(2, 3, 2.55)
        m4.SetElement(3, 3, 1.0)
        m4.Modified()
        self._bar.actor.Modified()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    vk = VirtualKeyboard(songname="pianoplayer demo")
    vk.vp.show(interactive=1, camera=vk._camera, resetcam=0)
