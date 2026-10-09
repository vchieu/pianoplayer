#-------------------------------------------------------------------------------
# Name:         PianoPlayer
# Purpose:      Real time piano sound engine (no external samples needed)
#-------------------------------------------------------------------------------
"""
Real-time piano sound engine for PianoPlayer.

The engine is a small *scheduler + mixer + synthesizer* that runs in a
background audio thread.  Notes are scheduled with `note_on()` / `note_off()`
on a common time line, so that:

 - the left and the right hand are heard **together** (no blocking call),
 - chords are truly simultaneous,
 - releasing a key actually damps the string (no endless sine beeps).

Each note is synthesized with an additive model that mimics the main
behaviour of a piano string:

 - inharmonic partials             f_n = f0 * n * sqrt(1 + B*n^2)
 - two slightly mistuned strings   on the low partials (natural beating)
 - strike position comb filter     (the hammer does not hit the string end)
 - partial dependent decay         (high partials die first)
 - hammer transient                (short filtered noise burst)
 - velocity dependent brightness, attack and level
 - damper on key release           (fast, partial dependent decay)

Backends, in order of preference: miniaudio, sounddevice, none.
"""
from __future__ import annotations

import heapq
import math
import queue
import threading
import time
import wave

import numpy as np

try:
    import miniaudio
    HAS_MINIAUDIO = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MINIAUDIO = False

try:
    import sounddevice as _sounddevice
    HAS_SOUNDDEVICE = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SOUNDDEVICE = False


SAMPLE_RATE = 44100

_N_PARTIALS = 12       # partials used by each string
_BEAT_PARTIALS = 4     # low partials played by 2 mistuned strings
_DETUNE = 0.00016      # ~0.3 cent, gives the natural "chorus" of a piano
_STRIKE_POS = 0.115    # hammer position along the string (0..1)
_TAU_REL = 0.075       # nominal damper time [s]
_LN1000 = math.log(1000.0)


############################################################ note helpers
def midi_of(n) -> int:
    """Return the MIDI number of an `INote` (or of a music21 note/pitch)."""
    obj = getattr(n, "note21", n)
    if hasattr(obj, "pitch"):          # music21.note.Note
        obj = obj.pitch
    if hasattr(obj, "midi"):           # music21.pitch.Pitch
        return int(obj.midi)
    if hasattr(n, "chord21"):          # last resort: first pitch of the chord
        return int(n.chord21.pitches[0].midi)
    raise ValueError(f"cannot read pitch of {n!r}")


def midi_from_name(name: str, octave: int) -> int:
    """('C',4) -> 60, ('C#',4) -> 61."""
    semis = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
             "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
             "A#": 10, "Bb": 10, "B": 11}
    if name not in semis:
        raise ValueError(f"unknown note name {name}")
    return 12 * (octave + 1) + semis[name]


def frequency(midi: int) -> float:
    """A4 = 440 Hz."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


##################################################### note parameter cache
_NOTE_CACHE: dict[int, dict] = {}


def _note_params(midi: int) -> dict:
    """Velocity independent parameters of one MIDI note (cached)."""
    cached = _NOTE_CACHE.get(midi)
    if cached is not None:
        return cached

    f0 = frequency(midi)
    n = np.arange(1, _N_PARTIALS + 1, dtype=float)

    # string inharmonicity: higher (stiffer) strings are more inharmonic
    pos = (midi - 21) / 87.0
    B = 0.00012 + 0.00110 * pos * pos
    f = f0 * n * np.sqrt(1.0 + B * n * n)

    # energy of each partial (the spectral tilt is applied per note)
    amp = np.abs(np.sin(np.pi * n * _STRIKE_POS)) / np.power(n, 1.6)

    # whole note decay and partial dependent decay
    t60 = float(np.clip(9.0 * math.sqrt(440.0 / max(f0, 20.0)), 0.9, 18.0))
    t60n = t60 / np.power(n, 0.55)
    rate = np.full(_N_PARTIALS, _LN1000 / t60)
    rate_n = _LN1000 / np.maximum(t60n, 0.05)

    # two mistuned strings for the lowest partials -> natural beating
    b = _BEAT_PARTIALS
    freqs = np.concatenate([f[:b] * (1 - _DETUNE), f[:b] * (1 + _DETUNE), f[b:]])
    amps = np.concatenate([amp[:b] * 0.5, amp[:b] * 0.5, amp[b:]])
    rates = np.concatenate([rate[:b], rate[:b], rate_n[b:]])
    partial_n = np.concatenate([n[:b], n[:b], n[b:]])
    # the damper stops high partials first
    rel_rate = _LN1000 / (_TAU_REL * np.power(partial_n, 0.30))
    n_all = partial_n

    peak = float(np.max(amps))
    if peak > 0:
        amps = amps / peak

    params = dict(freqs=freqs, amps=amps, rates=rates, rel_rate=rel_rate,
                  n_all=n_all, f0=f0, t60=t60)
    _NOTE_CACHE[midi] = params
    return params


##################################################### one sounding note
class _Voice:
    """A single sounding note (one key press)."""
    __slots__ = ("midi", "params", "amps", "start_frame", "rel_frame",
                 "pan_l", "pan_r", "noise", "gain")

    def __init__(self, midi, velocity, start_frame, pan, sr, gain=1.0):
        self.midi = midi
        self.params = p = _note_params(midi)
        v = float(min(max(velocity, 0.0), 1.0))

        # brighter when hit harder: tilt the spectrum with velocity
        amps = p["amps"] * np.power(p["n_all"], -0.85 * (1.0 - v))
        self.amps = amps / (float(np.max(amps)) or 1.0)

        self.start_frame = start_frame
        self.rel_frame = None

        self.gain = g = (0.25 + 0.75 * v) ** 1.4 * gain
        c = 0.35 * pan
        self.pan_l = math.sqrt(max(0.0, 0.5 * (1.0 - c)))
        self.pan_r = math.sqrt(max(0.0, 0.5 * (1.0 + c)))

        # short smoothed noise burst = hammer felt transient
        nn = max(64, int(0.045 * sr))
        rng = np.random.default_rng((midi * 977 + int(v * 500)) & 0xFFFFFFFF)
        nz = rng.standard_normal(nn)
        win = np.hanning(9)
        nz = np.convolve(nz, win / win.sum(), mode="same")
        self.noise = nz * (0.35 * (0.4 + 0.6 * v))

    # ----------------------------------------------------------
    def energy_at(self, frame: int) -> float:
        """Rough instantaneous amplitude, used for voice stealing and cleanup."""
        t = max(frame - self.start_frame, 0) / 44100.0
        e = math.exp(-float(np.min(self.params["rates"])) * t)
        if self.rel_frame is not None:
            tr = (frame - self.rel_frame) / 44100.0
            if tr > 0:
                e *= math.exp(-60.0 * tr)
        return e * float(np.max(self.amps)) * self.gain

    def alive_at(self, frame: int) -> bool:
        return self.energy_at(frame) > 2e-4


##################################################### synthesizer core
class SynthCore:
    """Sample accurate scheduler + additive piano synthesizer (numpy only)."""

    def __init__(self, sample_rate=SAMPLE_RATE, polyphony=32):
        self.sr = int(sample_rate)
        self.polyphony = int(polyphony)
        self.frame = 0                    # audio clock, in samples
        self.voices: list[_Voice] = []
        self._events: list[tuple] = []     # heap of (frame, seq, kind, ...)
        self._seq = 0
        self._pending: queue.SimpleQueue = queue.SimpleQueue()

    # ---------------------------------------------------------- scheduling
    def note_on(self, midi, velocity=0.6, when=None, pan=0.0):
        self._push(self._to_frame(when), 0, int(midi), float(velocity), float(pan))

    def note_off(self, midi, when=None):
        self._push(self._to_frame(when), 1, int(midi), 0.0, 0.0)

    def all_notes_off(self, when=None):
        frame = self._to_frame(when)
        for v in list(self.voices):
            # identify the voice by its start frame: unique and stable
            self._push(frame, 2, float(v.start_frame), 0.0, 0.0)

    def _to_frame(self, when):
        if when is None:
            return self.frame
        return max(int(round(when * self.sr)), self.frame)

    def _push(self, frame, kind, midi, vel, pan):
        self._seq += 1
        self._pending.put((frame, self._seq, kind, midi, vel, pan))

    def _drain(self):
        try:
            while True:
                heapq.heappush(self._events, self._pending.get_nowait())
        except queue.Empty:
            pass

    # ---------------------------------------------------------- rendering
    def render_block(self, nframes) -> np.ndarray:
        """Render `nframes` stereo samples, returns a float array of shape (n,2)."""
        nframes = int(nframes)
        out = np.zeros((nframes, 2), dtype=np.float64)
        self._drain()
        done = 0

        while done < nframes:
            # 1) apply every event that is due at the current clock position
            while self._events and self._events[0][0] <= self.frame:
                frame, _, kind, midi, vel, pan = heapq.heappop(self._events)
                if kind == 0:
                    self._spawn(midi, vel, pan, frame)
                elif kind == 1:
                    self._release(midi, frame)
                else:
                    self._release_by_start(int(midi), frame)

            # 2) render up to the next scheduled event (sample accurate)
            remaining = nframes - done
            nxt = self._events[0][0] if self._events else self.frame + remaining
            length = min(max(nxt - self.frame, 0), remaining)
            if length > 0:
                self._mix(out, done, length)
                self.frame += length
                done += length
            elif not self._events:
                break  # cannot happen, guards against a runaway loop

        return out

    def _spawn(self, midi, vel, pan, frame):
        if len(self.voices) >= self.polyphony:
            weakest = min(self.voices, key=lambda v: v.energy_at(self.frame))
            self.voices.remove(weakest)
        self.voices.append(_Voice(midi, vel, frame, pan, self.sr))

    def _release(self, midi, frame):
        target = None
        for v in self.voices:
            if v.midi == midi and v.rel_frame is None:
                if target is None or v.start_frame < target.start_frame:
                    target = v
        if target is not None:
            target.rel_frame = frame

    def _release_by_start(self, start_frame, frame):
        for v in self.voices:
            if v.start_frame == start_frame:
                if v.rel_frame is None or v.rel_frame > frame:
                    v.rel_frame = frame
                return

    def _mix(self, out, offset, length):
        if length <= 0:
            return
        if not self.voices:
            out[offset:offset + length] = 0.0
            return

        sr = float(self.sr)
        start = self.frame
        t_base = np.arange(length, dtype=float) / sr
        left = np.zeros(length)
        right = np.zeros(length)
        alive = []

        for v in self.voices:
            p = v.params
            t0 = max((start - v.start_frame) / sr, 0.0)
            t = t0 + t_base

            # --- envelope ------------------------------------------------
            env = np.minimum(t / 0.004, 1.0)                # smooth 4 ms attack
            env = env * env * (3.0 - 2.0 * env)             # smooth step
            env = env[None, :] * np.exp(-p["rates"][:, None] * t[None, :])

            if v.rel_frame is not None:
                rel = (start - v.rel_frame) / sr
                if rel >= 0.0:                              # damper is on
                    env = env * np.exp(-p["rel_rate"][:, None] * rel)

            sig = np.sin(p["freqs"][:, None] * (2.0 * np.pi) * t[None, :])
            sig = v.gain * (v.amps[:, None] * env * sig).sum(axis=0)

            # --- hammer transient ---------------------------------------
            i0 = int(t0 * sr)
            if i0 < len(v.noise):
                chunk = v.noise[i0:i0 + length]
                if len(chunk):
                    brust = chunk * np.exp(-np.arange(len(chunk)) / (0.012 * sr))
                    sig[:len(brust)] += brust

            left += sig * v.pan_l
            right += sig * v.pan_r

            if v.alive_at(start + length):
                alive.append(v)

        self.voices = alive
        out[offset:offset + length, 0] = left
        out[offset:offset + length, 1] = right

    # ---------------------------------------------------------- master
    @staticmethod
    def master(mix: np.ndarray, volume: float) -> np.ndarray:
        if volume != 1.0:
            mix = mix * float(volume)
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.985:                      # soft peak limit, never clip
            mix = mix * (0.985 / peak)
        return np.clip(mix, -1.0, 1.0)


##################################################### audio backends
class _MiniAudioBackend:
    name = "miniaudio"

    def __init__(self, synth: SynthCore, volume_getter):
        self.synth = synth
        self.volume_getter = volume_getter
        self.device = None

    def start(self):
        self.device = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=2,
            sample_rate=self.synth.sr,
            buffersize_msec=25,
        )
        gen = self._generator()
        next(gen)                      # prime the generator
        self.device.start(gen)

    def _generator(self):
        required = yield b""
        while True:
            n = int(required) if required else 512
            mix = SynthCore.master(self.synth.render_block(n), self.volume_getter())
            pcm = np.ascontiguousarray(mix * 32767.0, dtype=np.int16)
            required = yield pcm.tobytes()

    def stop(self):
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None


class _SounddeviceBackend:
    name = "sounddevice"

    def __init__(self, synth: SynthCore, volume_getter):
        self.synth = synth
        self.volume_getter = volume_getter
        self.stream = None

    def start(self):
        def cb(outdata, frames, time_info, status):
            mix = SynthCore.master(self.synth.render_block(frames), self.volume_getter())
            outdata[:] = (mix * 32767.0).astype(np.int16).reshape(-1, 2)

        self.stream = _sounddevice.OutputStream(
            samplerate=self.synth.sr, channels=2, dtype="int16",
            blocksize=512, callback=cb,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class _NullBackend:
    """No audio device: keeps the engine clock running, renders silence."""
    name = "none"

    def __init__(self, synth, volume_getter):
        self.synth = synth
        self._stop = threading.Event()

    def start(self):
        def idle():
            step = int(0.05 * self.synth.sr)
            while not self._stop.is_set():
                self.synth.render_block(step)
                time.sleep(0.05)

        threading.Thread(target=idle, daemon=True).start()

    def stop(self):
        self._stop.set()


##################################################### public engine
class SoundEngine:
    """
    High level, thread safe piano sound engine.

    Typical use::

        eng = SoundEngine(volume=0.8)
        eng.start()
        eng.note_on(60, velocity=0.7, when=0.5)   # half a second from now
        eng.note_off(60, when=1.2)
        ...
        eng.stop()
    """

    def __init__(self, volume=0.8, enabled=True, backend="auto",
                 sample_rate=SAMPLE_RATE, polyphony=32):
        self.volume = float(volume)
        self.enabled = bool(enabled)
        self.sample_rate = int(sample_rate)
        self.backend_name = "none"
        self.synth = SynthCore(self.sample_rate, polyphony)
        self._backend = _NullBackend(self.synth, lambda: self.volume)
        self._backend_pref = backend
        self._started = False

    # ----------------------------------------------------------- lifecycle
    def start(self):
        if self._started:
            return self
        self._started = True
        if self.enabled:
            backend = self._make_backend()
            try:
                backend.start()
                self._backend = backend
                self.backend_name = backend.name
            except Exception as exc:  # pragma: no cover - device problems
                print(f"Audio backend {backend.name!r} failed ({exc}); "
                      "continuing without sound.")
                self.backend_name = "none"
                self._backend = _NullBackend(self.synth, lambda: self.volume)
                self._backend.start()
        else:
            self._backend.start()
        return self

    def _make_backend(self):
        pref = self._backend_pref
        if pref in (None, "", "none"):
            return _NullBackend(self.synth, lambda: self.volume)
        if pref in ("auto", "miniaudio") and HAS_MINIAUDIO:
            return _MiniAudioBackend(self.synth, lambda: self.volume)
        if pref in ("auto", "sounddevice") and HAS_SOUNDDEVICE:
            return _SounddeviceBackend(self.synth, lambda: self.volume)
        if pref == "auto":
            print("No audio backend available (pip install miniaudio), "
                  "running silently.")
        return _NullBackend(self.synth, lambda: self.volume)

    def stop(self):
        if not self._started:
            return
        self.all_notes_off()
        try:
            self._backend.stop()
        except Exception:
            pass
        self._started = False

    # ----------------------------------------------------------- transport
    @property
    def now(self) -> float:
        """Current engine time in seconds (since start())."""
        return self.synth.frame / float(self.sample_rate)

    def note_on(self, midi, velocity=0.6, when=None, pan=0.0):
        if midi is None:
            return
        self.synth.note_on(midi, velocity, when, pan)

    def note_off(self, midi, when=None):
        if midi is None:
            return
        self.synth.note_off(midi, when)

    def all_notes_off(self):
        self.synth.all_notes_off()

    # ----------------------------------------------------------- offline
    def render(self, events, seconds) -> np.ndarray:
        """
        Render `events` offline.  `events` is an iterable of
        (time, 'on'|'off', midi, velocity, pan) tuples, returns (n, 2) floats.
        """
        sr = self.sample_rate
        total = max(int(seconds * sr), 1)
        local = SynthCore(sr, self.synth.polyphony)
        ordered = sorted(events, key=lambda e: (e[0], e[1] not in ("off", 1)))
        for (t, kind, midi, vel, pan) in ordered:
            when = int(round(t * sr)) / sr
            if kind in (0, "on", "onset"):
                local.note_on(midi, vel, when, pan)
            else:
                local.note_off(midi, when)
        chunks = []
        step = 4096
        while local.frame < total:
            chunks.append(local.render_block(min(step, total - local.frame)))
        mix = np.concatenate(chunks) if chunks else np.zeros((0, 2))
        return SynthCore.master(mix, self.volume)

    def export_wav(self, events, seconds, path) -> str:
        """Render `events` offline and write a 16 bit stereo WAV file."""
        mix = self.render(events, seconds)
        pcm = (mix * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return path
