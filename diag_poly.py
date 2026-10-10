"""
Peak polyphony and voice stealing for a real score, using exactly the event
list the 3D player schedules.

Voice stealing kills a ringing voice at full amplitude -> an audible click,
which sounds like a scratch on a record.

    python diag_poly.py [score.xml]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from music21 import converter

from pianoplayer.audio import SoundEngine, SynthCore
from pianoplayer.hand import Hand
from pianoplayer.scorereader import reader
from pianoplayer.utils import score_tempo
from pianoplayer.vkeyboard import VirtualKeyboard

S = {"steal": 0, "peak_voices": 0, "peak_events": 0}
_orig_spawn = SynthCore._spawn


def spawn(self, midi, vel, pan, frame):
    if len(self.voices) >= self.polyphony:
        S["steal"] += 1
    S["peak_voices"] = max(S["peak_voices"], len(self.voices) + 1)
    S["peak_events"] = max(S["peak_events"], len(self._events))
    return _orig_spawn(self, midi, vel, pan, frame)


SynthCore._spawn = spawn


def main(fn="scores/bach_invention4.xml"):
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
    for h, sign in hands:
        (vk.build_RH if sign > 0 else vk.build_LH)(h)

    evs, end = [], 0.0
    for (h, sign), pan in zip(hands, (0.30, -0.30)):
        for e in vk._events_for(h, sign, pan):
            if e.midi is None:
                continue
            evs.append((e.t_on, "on", e.midi, e.vel, pan))
            evs.append((e.t_off, "off", e.midi, 0.0, pan))
            end = max(end, e.t_off)
    evs.sort()

    eng = SoundEngine(volume=0.8, enabled=False, backend="none")
    t0 = __import__("time").perf_counter()
    eng.render(evs, end + 2.0)
    dt = __import__("time").perf_counter() - t0

    print("score %s  tempo %.1f  %d events, %.1f s" % (fn, tempo, len(evs), end))
    print("polyphony cap       : %d" % eng.synth.polyphony)
    print("peak voices at once : %d" % S["peak_voices"])
    print("peak queued events  : %d" % S["peak_events"])
    print("voice steals        : %d   <- a steal cuts a ringing note: click"
          % S["steal"])
    print("rendered in %.2f s (%.1fx realtime)" % (dt, end / max(dt, 1e-9)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scores/bach_invention4.xml")
