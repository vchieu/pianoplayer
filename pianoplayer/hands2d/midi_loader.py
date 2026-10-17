"""
midi_loader.py (shim) - the note/song data model the vendored hands.py draws.

Kept field for field from D:\\Project\\pygame-piano-animator\\midi_loader.py
(Note, TrackInfo, MidiSong, the pedal constants and the queries hands.py
uses), minus everything that only exists to *load* a MIDI file: the file
reader, `pretty_midi` (not a pianoplayer dependency) and the fingering /
glissando text-meta-event decoding. pianoplayer builds the notes itself from
its own reader, so `MidiSong(notes)` is all that is ever constructed.
"""
from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from typing import List, Optional

LOWEST_PIANO_KEY = 21    # A0
HIGHEST_PIANO_KEY = 108  # C8
MIDDLE_C = 60

LEFT = "L"
RIGHT = "R"

SUSTAIN, SOSTENUTO, SOFT = 64, 66, 67
PEDALS = (SUSTAIN, SOSTENUTO, SOFT)

_BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def is_black_key(pitch: int) -> bool:
    """True for the sharps and flats (the black keys)."""
    return pitch % 12 in _BLACK_PITCH_CLASSES


def note_name(pitch: int) -> str:
    """60 -> 'C4'."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class Note:
    pitch: int        # MIDI note number, 21..108 on an 88-key piano
    start: float      # seconds
    end: float        # seconds
    velocity: int     # 1..127
    track: int        # index of the instrument/track it came from
    hand: str         # LEFT or RIGHT (pianoplayer: which hand plays it)
    finger: Optional[int] = None   # 1..5 when we know the fingering
    gliss: bool = False            # part of a glissando

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_black(self) -> bool:
        return is_black_key(self.pitch)

    @property
    def name(self) -> str:
        return note_name(self.pitch)


@dataclass(frozen=True)
class TrackInfo:
    index: int
    name: str
    program: int              # General MIDI program (0 = acoustic grand)
    note_count: int
    hand: Optional[str]       # LEFT/RIGHT if the whole track is one hand, else None


def pedal_switches(controls):
    """[(time, controller, 0 | 127)]: each pedal as an on/off switch (on at 64+), changes only."""
    out, state = [], {}
    for t, c, v in controls:
        on = 127 if v >= 64 else 0
        if state.get(c, 0) != on:
            state[c] = on
            out.append((t, c, on))
    return out


class MidiSong:
    """A song: sorted notes plus fast queries by time."""

    def __init__(self, notes: List[Note], tracks: List[TrackInfo] = (), duration: float = 0.0,
                 bar_times: List[float] = (), beat_times: List[float] = (), path: Optional[str] = None,
                 controls: Optional[list] = None):
        self.notes: List[Note] = sorted(notes, key=lambda n: (n.start, n.pitch))
        # pedals: [(time, controller, value)] played as switches (the MIDI
        # standard's 64 threshold); the file's raw values stay in raw_controls.
        self.raw_controls = sorted(controls or [], key=lambda c: (c[0], c[1]))
        self.controls = pedal_switches(self.raw_controls)
        self._control_times = [c[0] for c in self.controls]
        self.tracks = list(tracks)
        self.duration = duration
        self.bar_times = list(bar_times)
        self.beat_times = list(beat_times)
        self.path = path
        self.title = os.path.splitext(os.path.basename(path))[0] if path else "Untitled"
        self.cleanup: List[str] = []

        self._starts = [n.start for n in self.notes]
        # Longest note lets us bound how far back an overlapping note can start.
        self._max_duration = max((n.duration for n in self.notes), default=0.0)

    def __len__(self) -> int:
        return len(self.notes)

    def __bool__(self) -> bool:
        return True                  # a song without notes is still a song (`if song:` means "loaded")

    def notes_between(self, t0: float, t1: float) -> List[Note]:
        """Notes that overlap the time window [t0, t1), in start order."""
        lo = bisect.bisect_left(self._starts, t0 - self._max_duration)
        hi = bisect.bisect_left(self._starts, t1)
        return [n for n in self.notes[lo:hi] if n.end > t0]

    def active_notes(self, t: float) -> List[Note]:
        """Notes sounding at time t (start <= t < end)."""
        lo = bisect.bisect_left(self._starts, t - self._max_duration)
        hi = bisect.bisect_right(self._starts, t)
        return [n for n in self.notes[lo:hi] if n.end > t]

    def controls_between(self, t0: float, t1: float) -> list:
        """Pedal events with t0 < time <= t1."""
        lo = bisect.bisect_right(self._control_times, t0)
        hi = bisect.bisect_right(self._control_times, t1)
        return self.controls[lo:hi]

    def control_state(self, t: float) -> dict:
        """{controller: value} in force at time t (0 for pedals not yet pressed)."""
        state = {c: 0 for c in PEDALS}
        for _, c, v in self.controls[:bisect.bisect_right(self._control_times, t)]:
            state[c] = v
        return state

    def sustain_at(self, t: float) -> bool:
        return self.control_state(t).get(SUSTAIN, 0) >= 64

    def pitch_range(self) -> tuple:
        if not self.notes:
            return (LOWEST_PIANO_KEY, HIGHEST_PIANO_KEY)
        pitches = [n.pitch for n in self.notes]
        return (min(pitches), max(pitches))