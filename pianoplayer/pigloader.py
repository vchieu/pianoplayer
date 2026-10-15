"""Correct PIG (PianoFingeringDataset) loader (Phase 1, step 1.3).

Unlike scorereader.PIG2Stream (which re-emits same-onset chords once per
line and stores no chord fingerings -- F3), this builds scorereader.INote
lists directly: same-onset lines of one channel form ONE chord group, and
every note/chord member keeps its own expert finger.
"""
import music21.note
import music21.chord
import music21.pitch

from pianoplayer.scorereader import INote
from pianoplayer.utils import keypos

_ONSET_EPS = 1e-6


def _parse_finger(tok):
    """Expert finger 1..5, or None when unknown.

    `3` plain, `-3` left-hand (negative), `2_1` substitution (first
    number), `_` / non-numeric = unknown.
    """
    if tok is None:
        return None
    tok = str(tok).strip()
    head = tok.split("_")[0]
    if not head or head in ("+", "-"):
        return None
    try:
        f = abs(int(head))
    except (TypeError, ValueError):
        return None
    return f if 1 <= f <= 5 else None


def load_pig(fname):
    """Parse a PianoFingeringDataset text file.

    Returns {0: (inotes_rh, experts_rh), 1: (inotes_lh, experts_lh)}
    where each `inotes` is a list of scorereader.INote ready for
    Hand.generate() and `experts` is a parallel list of ints 1..5 or
    None (unknown).
    """
    channels = {0: [], 1: []}  # channel -> list of parsed line dicts
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            _noteid, onset, offset, name, _onv, _offv, channel, finger = parts[:8]
            try:
                onset = float(onset)
                offset = float(offset)
                channel = int(channel)
            except ValueError:
                continue
            if channel not in (0, 1):
                continue
            if offset - onset < 1e-4:  # zero-length: skip (as old code did)
                continue
            pitch = music21.pitch.Pitch(name.replace("b", "-"))
            channels[channel].append({
                "onset": onset,
                "offset": offset,
                "pitch": pitch,
                "expert": _parse_finger(finger),
            })

    out = {}
    for channel in (0, 1):
        lines = channels[channel]
        # group same-onset lines (equal within 1e-6 s), in file order
        groups = []
        for entry in lines:
            if groups and abs(entry["onset"] - groups[-1][0]) <= _ONSET_EPS:
                groups[-1][1].append(entry)
            else:
                groups.append((entry["onset"], [entry]))
        inotes, experts = [], []
        chordID = 0
        for _onset, members in groups:
            # chord members sorted by MIDI pitch ascending (as reader())
            members = sorted(members, key=lambda e: int(e["pitch"].midi))
            if len(members) >= 2:
                ch = music21.chord.Chord([e["pitch"] for e in members])
                for j, e in enumerate(members):
                    an = INote()
                    an.isChord = True
                    an.chordID = chordID
                    an.chord21 = ch
                    an.note21 = ch.pitches[j]
                    _fill(an, e)
                    an.chordnr = j
                    an.NinChord = len(members)
                    inotes.append(an)
                    experts.append(e["expert"])
                chordID += 1
            else:
                e = members[0]
                an = INote()
                an.isChord = False
                an.note21 = music21.note.Note(e["pitch"])
                _fill(an, e)
                an.chordnr = 0
                an.NinChord = 1
                inotes.append(an)
                experts.append(e["expert"])
        out[channel] = (inotes, experts)
    return out


def _fill(an, e):
    """Shared INote fields (quarters at 120 bpm; t_sec in seconds)."""
    pitch = e["pitch"]
    onset_sec = float(e["onset"])
    dur_sec = float(e["offset"]) - onset_sec
    an.name = pitch.name
    an.octave = pitch.octave
    an.x = keypos(pitch)
    an.isBlack = pitch.pitchClass in (1, 3, 6, 8, 10)
    an.midi = int(pitch.midi)
    an.time = an.time_real = onset_sec * 2.0
    an.duration = an.dur_real = max(dur_sec * 2.0, 0.05)
    an.t_sec = onset_sec
    an.measure = 0  # no measure filtering
    an.fingering = 0
