# AGENTS.md

PianoPlayer is a single-package Python app (music21 + vedo + numpy + miniaudio)
that finds an optimal piano fingering for a score and can play it back in 3D
with a built-in additive piano synth.

## Run / test

```bash
pip install -e .                      # editable install; the CLI script is bin/pianoplayer
python bin/pianoplayer scores/bach_invention4.xml -n10 -v   # CLI (no args = GUI)
python bin/pianoplayer -h             # all flags
```

Each `test_*.py` is a standalone script, not a pytest suite. Run from repo root:

```bash
python test_audio.py          # offline synth checks (FFT, envelope, timing)
python test_play.py           # headless 3D-frame smoke test -> ./test_play_out/*.png
python test_hand_render.py    # offscreen hand poses -> ./test_hand_render/*.png
python test_shadow.py         # vtkShadowMapPass pipeline sanity
python test_sound.py          # legacy simpleaudio chord playback
python test_regression.py     # finger/key/audio/fingering consistency, offscreen, no audio device
```

There is no linter, formatter, CI config, or pyproject; `setup.py` is the only
build manifest.

## Module map

| File | Role |
|------|------|
| `bin/pianoplayer` | CLI + Tkinter GUI entry point (installed as the `pianoplayer` script) |
| `pianoplayer/scorereader.py` | `reader()` -> music21 Stream to `INote` list (ties absorbed into the first note, simultaneous voices kept, `time_real`/`dur_real` carry the true onsets for playback); `PIG2Stream()` for PIG `.txt` |
| `pianoplayer/hand.py` | **Core algorithm** -- `Hand` finds the fingering |
| `pianoplayer/hand3d.py` | Procedural anatomical hand, analytic IK (`HandRig`) |
| `pianoplayer/audio.py` | Real-time additive piano synth (`SoundEngine`, `SynthCore`) |
| `pianoplayer/vkeyboard.py` | 3D scene, frame loop, A/V sync, HUD (`VirtualKeyboard`) |
| `pianoplayer/utils.py` | Keyboard geometry, note-name helpers, shared constants |
| `pianoplayer/wavegenerator.py` | Legacy simpleaudio chord player (deprecated) |

## The fingering algorithm (source of truth)

`Hand.generate()` (`hand.py`) walks the note sequence and, for every window of
up to 9 notes, calls `optimize_seq()` to brute-force every 5-finger assignment
and keep the one with the lowest average finger speed.

**Cost function** (`ave_velocity`): for each consecutive note pair,
`v = dx / (dt + 0.1)` where `dx` is the distance the finger travels on the
keyboard (cm) and `dt` is the inter-onset time (s). Black-key notes are
penalized by dividing `v` by `weights[finger] * bfactor[finger]` -- a
*lower* `bfactor` means a *stronger* black-key penalty, so the thumb
(0.3) is penalized most and the middle finger (1.1) least. The mean
over the window is the combination's cost; lower is better.

**Skip rules** (`optimize_seq.skip`): prune impossible/ugly transitions before
scoring them -- same finger on two different notes, finger crossings, thumb-to-black
crossings, and (inside chords) finger-order violations plus a set of normalized
stretch thresholds (`axba = |dx| * hf / 0.8` vs. finger-pair limit). These are
hard-coded heuristics; changing them changes the character of the output.

**Autodepth**: when `autodepth=True` (default, depth flag `0`), the window depth
is chosen per position as the smallest `i` in 4..9 whose note reaches > 3.5 s
ahead. Near the end of the score (< 10 notes left) it falls back to depth 9,
and the last full 9-note window is solved at `i = N-9` so the final note gets
a finger too. Scores with fewer than 9 notes are padded with dummy copies of
the last note and solved once.

**Key positions**: `utils.keypos()` computes a note's keyboard x from its
MIDI number, so enharmonic spellings (B#3 = C4, Cb4 = B3, E#, Fb, ...)
always land on the physical key that sounds.

**Left hand** is solved by mirroring: `anote.x = -anote.x`, i.e. the left hand
is treated as a right hand on a mirrored keyboard. All geometry and the
right-handed skip logic apply unchanged. The flip is flagged per note
(`anote._mirrored`), `generate()` restores its entry depth settings and
replaces (not appends) the `Fingering` articulation, so `generate()`
can run twice on the same `noteseq` and reproduces the same fingering.

Hand span (`size` `XXS`..`XXL`) scales `frest` (rest finger positions) by
`utils.handSizeFactor` (0.33..1.2) and prints the resulting thumb-pinkie span
(21 * hf cm).

## Audio engine (source of truth)

`audio.py` is a **sample-accurate scheduler + additive piano synth**:

- Each note spawns a `_Voice` whose partials are `f0 * n * sqrt(1 + B*n^2)`
  (inharmonic string), with the lowest `_BEAT_PARTIALS=2` doubled and detuned
  by `_DETUNE` (two-string beating), a comb filter for the hammer strike
  position, and partial-dependent decay/damper rates. Voice count is capped at
  `polyphony=32` with energy-based stealing.
- The device callback never renders: it copies pre-rendered chunks from a ring
  (`_CushionBackend`). The **animation thread** fills the ring via `pump()`
  because a background thread cannot win the GIL while VTK renders -- this is
  why the frame loop in `vkeyboard.py` calls `engine.pump()` every frame.
- A sample rendered at engine time `t` is *heard* at `t + output_latency`
  (measured from the priming gap, not assumed). The epoch offset
  `engine_wall + latency - _t0` is folded into `note_on(when=...)` so the
  sound is heard exactly when the finger lands; it is **re-measured every
  frame** (the latency only grows, each buffer underrun pads with silence),
  so notes scheduled after an underrun compensate for the delay.
  `AUDIO_LOOKAHEAD` (1.0 s) must
  cover device period + mixer + ring cushion + a stalled frame.
- Env: 4 ms smooth-step attack, exponential decay per partial, fast damper on
  release, hammer felt transient (filtered noise burst baked into the voice).
- Backends in order: `miniaudio`, `sounddevice`, `none` (null backend keeps the
  clock running silently). Force `PIANOPLAYER_LATENCY_MS` env var to recalibrate.

## 3D player (source of truth)

`vkeyboard.py` drives the scene with a vedo timer but runs its **own** frame
loop (`_run_frames`) instead of `interactor.Start()` -- the blocking message
wait costs ~30 ms/frame and starves the audio ring. The camera is a locked
orthographic top-down view (`_lock_camera` swaps the interactor style).

`hand3d.py` builds the hand from three draw calls (lofted arm+palm, five tapered
finger tubes, five nail ellipsoids) and re-poses them every frame with an
analytic 3-segment IK (`solve_chain`): MCP->PIP->DIP chain, PIP bends by `b1`,
DIP by `COUPLING * b1`, bisection on the chord length so the tip lands exactly
on the key. Faces are wound **inward** on purpose -- VTK would otherwise treat
outer faces as back-facing and shade them black; the left hand mirrors and
reverses face winding again for the same reason.

Key dip: keys hinge around `KEY_PIVOT_Y`, drop = `(KEY_PIVOT_Y - y) * sin(angle)`.

**Playback timing** (`_events_for` / `_beat_seconds`): note onsets and
durations are read from `time_real` / `dur_real` (true quarter lengths,
ties included) and converted to seconds through `tempo_map`
(`[(offset_quarters, bpm), ...]`, `utils.score_tempo_map()` honors
`MetronomeMark.getQuarterBPM()` so referents like ♪=120 or ♩.=60 are
correct, and tempo changes are integrated piecewise). With no map, the
constant `tempo` is used. While a finger holds a key, that finger stays
exactly on the pressed key's x (`_Ev.wx`) even though the hand aims at
the next note.

## Conventions

- Version is defined in **two** places that must stay in sync:
  `setup.py` (`version='2.2.0'`) and `pianoplayer/__init__.py`
  (`__version__ = "2.2.0"`).
- `output.xml` is the default annotated output (MusicXML with fingering
  articulations / lyrics).
- Beam numbering: right hand = 0, left hand = 1 (`-rbeam` / `-lbeam`).
- Fingering output: `articulations` (musescore-friendly) by default;
  `lyrics` with `-b/--below-beam`.

## Docs maintenance

**When source changes, update this file (`AGENTS.md`) and `README.md` together.**
The algorithm / audio / 3D notes above mirror `hand.py`, `audio.py`, `vkeyboard.py`
and `hand3d.py` -- if those change, the matching section here must change too.
