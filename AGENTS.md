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
python test_hand_render.py    # offscreen hand poses + a per-style loop (4 skins) -> ./test_hand_render/*.png
python test_shadow.py         # vtkShadowMapPass pipeline sanity
python test_sound.py          # legacy simpleaudio chord playback
python test_regression.py     # finger/key/audio/fingering consistency, offscreen, no audio device
python test_fingering.py      # fingering validity (fingers 1..5, held-finger, chord order) + 500-note solve time
python test_fingering_spans.py # hand span limits vs limit * hf (size matters)
python test_fingering_props.py # fingering property checks
python test_pig_loader.py     # PIG loading via load_pig (chords included)
python test_fingering_hints.py # INote.hint / PIN_COST hint handling
python test_fingering_valley.py # broken-chord valley outer-finger + no-reuse (50-nam-ve-sau LH bars)
python test_fingering_343.py # 3-4-3 weak-pair alternation (50-nam-ve-sau LH bars 31-32)
python test_pause_tempo.py # Space pause + tempo popup/--tempo (headless-safe)
python fingering_audit.py     # audit tool for fingering output
python tune_fingering.py      # tuning tool for fingering weights
python test_hand_reach.py     # fingertip reach/bend over all sizes/fingers/keys (70 cells)
python test_hand_anatomy.py   # anatomical proportions on the mesh (6 checks)
python test_review_fixes.py   # review-fix acceptance: cache persistence, t_sec, texture fallback, thumb lock
python eval_fingering.py      # fingering quality gates (gold + PIG via load_pig, chords included, note-weighted avg + F1-F3 + aim; gold-extra incl. figure-lite octave/arpeggio rows, informational; C arpeggio R is XFAIL)
python test_hand_look.py      # Tier A hand-look acceptance: tilt/gap/ratio/bend, offscreen
python test_sync.py           # A/V sync incl. mid-playback output-latency jump (FakeEngine)
python test_falling.py        # falling-note geometry (numpy only, no window)
python test_repeat_notes.py   # repeated same-pitch notes visibly release key/finger (no window)
python test_hands2d.py     # 2D hand backend: overlay paints, fingertips on the 3D keys, styles, view toggle
```

There is no linter, formatter, CI config, or pyproject; `setup.py` is the only
build manifest.

## Module map

| File | Role |
|------|------|
| `bin/pianoplayer` | CLI + Tkinter GUI entry point (installed as the `pianoplayer` script) |
| `pianoplayer/scorereader.py` | `reader()` -> music21 Stream to `INote` list (ties absorbed into the first note, simultaneous voices kept, `time_real`/`dur_real` carry the true onsets for playback; a chord reduced to one sounding pitch is not `isChord`; only numeric 1..5 lyrics become finger hints; single-note parts allowed); `PIG2Stream()` for PIG `.txt` (duplicate-chord behaviour unchanged; CLI/GUI import path only) |
| `pianoplayer/pigloader.py` | PIG dataset loading via `load_pig()` (chords included, note-weighted-average evaluation) |
| `pianoplayer/hand.py` | **Core algorithm** -- `Hand` finds the fingering |
| `pianoplayer/hand3d.py` | Procedural anatomical hand, analytic IK (`HandRig`), hand motion step 1 (anticipation / wrist fit / tendon fan), skin styles step 2 (`style=` + vertex-colour bands), anatomy/flexibility step 3 (per-finger joint limits + hover/idle knobs) |
| `pianoplayer/handspec.py` | Single source of truth for hand geometry (finger/base/lens/radii, arm profile, rest pose, SDF constants, `geom_hash()`), the 19-bone anatomy view (step 3, derived from those hashed constants), the motion knobs, the skin-style palettes/materials/bands and the per-finger flexibility knobs (none of those hashed); `hand3d` and `handmesh` both import from here |
| `pianoplayer/handmesh.py` | Unified skinned hand mesh (Tier B): SDF rest mesh, LBS weights, procedural skin texture (+ per-style texture variants) |
| `pianoplayer/audio.py` | Real-time additive piano synth (`SoundEngine`, `SynthCore`) |
| `pianoplayer/vkeyboard.py` | 3D scene, frame loop, A/V sync, HUD (`VirtualKeyboard`) |
| `pianoplayer/falling.py` | Synthesia-style falling notes: pure numpy bar/digit geometry + VTK `FallingNotes` layer |
| `pianoplayer/utils.py` | Keyboard geometry, note-name helpers, shared constants |
| `pianoplayer/wavegenerator.py` | Legacy simpleaudio chord player (deprecated) |
| `pianoplayer/hands2d/` | **Vendored 2D hand model** (`hands.py` pose solver + `skins.py` skins from pygame-piano-animator, verbatim except import lines), three shims (`midi_loader.py` without pretty_midi, `common.py` key geometry only, `paths.py` DATA_DIR) and the `Hands2D` pygame->VTK overlay adapter (`render.py`) |

## The fingering algorithm (source of truth)

`Hand.generate()` (`hand.py`) solves the whole in-range note slice at once
with `solve_viterbi()` -- a Viterbi search over finger-pair states (25
states, order-2), scored by `pair_cost()` and followed by a greedy
`_repair_held()` pass for held-finger violations deeper than order-2 memory.
500 notes solve in ~0.28 s (the old 9-note `5^9` window search took ~60 s);
on `bach_invention4.xml` the Viterbi cost beats the legacy cost.

**Cost function** (`pair_cost`, same formula as the old `ave_velocity`): for
each consecutive note pair, `v = dx / (dt + 0.1)` where `dx` is the distance
the finger travels on the keyboard (cm) and `dt` is the inter-onset time in
seconds (`INote.t_sec`, stamped by `reader()` through `utils.beat_to_seconds`
from the score tempo map; the CLI/GUI pass `score_tempo_map(sf)`). Same key
repeated: every finger starts from reach 0 and a same-finger repeat faster
than 0.25 s pays `REPEAT_PENALTY = 6` scaled linearly to zero gap (F2), so
fast repeats alternate fingers while slow ones may keep the finger.
Black-key notes are penalized by dividing `v` by
`weights[finger] * bfactor[finger]` -- a *lower* `bfactor` means a *stronger*
black-key penalty, so the thumb (0.3) is penalized most and the middle finger
(1.1) least. Fingers 4/5 jumping far (`> WEAK_FINGER_MIN_JUMP = 3.0` cm)
on a short gap (`< WEAK_FINGER_MAX_DT = 0.15` s, measured in seconds through
the score tempo map -- see F1) pay `WEAK_FINGER_PENALTY = +3`. Lower total is better.

**Skip rules** (`_skip_penalty`, hard-soft with `HARD = 200.0` added per
violation, applied to **every** consecutive pair including chord<->note):
same finger on two different notes/pitches, finger crossings, thumb-to-black
crossings, and (inside chords) finger-order violations plus smooth span
costs (`MAX_SPAN`/`COMFORT_SPAN` tables in `handspec.py`, incl. new thumb rows,
compared as raw cm vs `limit * hf` so hand size matters, quadratic ramp to
`HARD` beyond max, F4a). Same finger on different keys is HARD only while the first note is
still held or the gap is at most 0.25 s; slower reuse pays a soft
`SAME_FINGER_BASE + SAME_FINGER_DX * |dx|` (F3; base 6.0, dx 0.4/cm, so reuse
across a leap pays leap-proportionally more but distance matters less than
before -- learned direction). Non-thumb crossings pay a soft
`CROSS_BASE` cost scaling with interval and inverse time (F5); thumb-under
pays `UNDER[fa]` (after-3 cheapest, halved when departing a black key onto
white). Thumb-under steps arriving on white ride the hand shift
(`reach_err = 0.02`, < 4.0 cm). Static black/white preference per finger
(`BLACK_COST`, F6) is added, not velocity-multiplied. Soft (added cost, still losable) rather than pruning, so a legacy
finger 0 or an impossible spot never kills a whole solution.

**Chord post-pass** (`_apply_chord_candidates`, F4b): after Viterbi, every
chord group (≤5 notes) is pinned to one whole monotonic candidate
(pre-generated `C(5,k)`, top-6 by static cost, never empty) minimizing static
+ neighbour transition costs -- the order-2 Viterbi state cannot enforce
whole-chord consistency. Layered chord DP (Amendment A node cost) covers the
same whole-chord consistency; `legacy_chord_postpass` flag selects the legacy
pass. `_repair_held` keeps chord pitch order when
re-fingering members, then reports cost delta (F7). It runs in linear time.

**Hints**: `INote.midi` carries the MIDI pitch and `INote.hint` an optional
pinned finger; pinned notes pay `PIN_COST` unless the solver uses the hint.

**Valley outer-finger bias** (broken-chord lines): a strict local pitch extremum
qualifies via the small-leap path (>= `VALLEY_MIN_DEPTH` (2) semitones each side)
or the arrival-leap path (drop into valley / rise into peak >= `VALLEY_DROP_MIN`
(5) semitones, e.g. a repositioning octave drop with a 1-semitone step out).
Reward on the outer finger (LH valley -> 5, peak -> 1; mirrored RH):
`VALLEY_BIAS + VALLEY_LEAP_K * (arrival leap - depth)`, capped at
`VALLEY_BONUS_CAP` (2.0 + 2.0/leap-semitone, cap 15.0). Chord members never
qualify. `PATTERN_343_PENALTY` stays 0.0/OFF (penalizing the triple alone
dodges to a (3,2,3) crossing instead of the valley 5).

**Figure-lite soft suggestions** (`_figure_bias`, idea port from
3D-piano-animator figures.py, reimplemented on INotes): consecutive single
notes (chord members and > 0.6 s gaps break runs) get MIDI-only pattern
suggestions -- chromatic runs (white -> 1, black -> 3), one-direction scale
runs >= 6 notes (standard pattern from the run start), wide arpeggios
(<= 4 pitch classes, span over an octave: cyclic 1-2-3/1-2-3-4, peak -> 5),
broken octaves (low -> thumb-side finger, high -> 5, 4 on a black key).
Priced once per note in the layered-DP node cost (arrival column in the
legacy path), disagreement pays `FIGURE_W_*` (0.75/1.5/2.0/2.5, far below
HARD so physical limits always win). Weak-finger unary (4: +0.24, 5: +0.12)
breaks ties toward strong fingers; thumb-black static cost is 0.25
(pinky 0.35, learned direction); one finger may cover two neighbour keys in
a chord (`THUMB_DOUBLE` 7.0 / `PINKY_DOUBLE` 8.0 soft instead of HARD) and an
octave-plus grip on 1-4 with an inner note down pays `INNER_ROOM_14` (4.0).

**Phase-2 verdicts**: E1 prior validated/OFF, E2 (`V_CAP = 40` +
`SEC_SMOOTH = 0.2` candidates) unvalidated without PIG, E3/E4 OFF.

**Held-finger constraint** (`pair_cost` + `_repair_held`): when note `a` still
sounds while `b` starts (`a.time_real + a.dur_real > b.time_real + HELD_EPS`),
re-using the held finger on a different key costs `HARD`. The order-2 Viterbi
state cannot remember longer holds, so `_repair_held()` greedily re-fingers
any remaining violation (cheapest `pair_cost` replacement against both
neighbours, warning on stdout when `verbose`).

**Legacy mode**: `self.legacy_window = False` by default; when `True`,
`generate()` delegates to `_generate_legacy()` -- the old 9-note
sliding-window `optimize_seq()` brute force with autodepth (smallest depth
4..9 reaching > 3.5 s ahead) and dummy-note padding for scores under 9 notes.
`-d/--depth` only applies in legacy mode (noted in `--help`). F5 off-by-one
fixed: range end is `>= start_measure + nmeasures`. A single-note range gets
finger 3; no dummy padding in Viterbi mode.

**Key positions**: `utils.keypos()` computes a note's keyboard x from its
MIDI number, so enharmonic spellings (B#3 = C4, Cb4 = B3, E#, Fb, ...)
always land on the physical key that sounds.

**Left hand** is solved by mirroring: `anote.x = -anote.x`, i.e. the left hand
is treated as a right hand on a mirrored keyboard. All geometry and the
right-handed skip logic apply unchanged. The flip is flagged per note
(`anote._mirrored`), `generate()` restores its entry depth settings and
replaces (not appends) the `Fingering` articulation (a chord keeps one
number per pitch), so `generate()` can run twice on the same `noteseq`
and reproduces the same fingering.

Hand span (`size` `XXS`..`XXL`) scales `frest` (rest finger positions) by
`utils.handSizeFactor` (0.33..1.2) and prints the resulting thumb-pinkie span
(21 * hf cm).

## Audio engine (source of truth)

`audio.py` is a **sample-accurate scheduler + additive piano synth**:

- Each note spawns a `_Voice` whose partials are `f0 * n * sqrt(1 + B*n^2)`
  (inharmonic string), with the lowest `_BEAT_PARTIALS=2` doubled and detuned
  by `_DETUNE` (two-string beating), a comb filter for the hammer strike
  position, and partial-dependent decay/damper rates. Voice count is capped at
  `polyphony=32` with energy-based stealing: a stolen voice gets a smooth
  per-sample `fast_kill` fade (`_TAU_KILL = 0.005` s, silent after ~6 ms)
  instead of a hard cut, and up to `_MAX_DYING_VOICES = 8` dying voices may
  exceed the cap while they fade.
- Render mix is float32 throughout; each voice uses
  `nmax = clip(9000/f0, _MIN_PARTIALS=8, _MAX_PARTIALS=24)` partials, partials
  below `_PARTIAL_FLOOR = 1e-3` leave the mix, the attack smoothstep is
  skipped once past 4 ms, and voices before `full_until` (no release yet)
  skip the gate test. `pump()` budget adapts to the ring level
  (`PUMP_BUDGET_FULL = 0.008` above 50%, `PUMP_BUDGET_EMPTY = 0.060`
  below 25%). ~5x realtime on a 20-voice load; `diag_load.py` 60 s runs with
  `underruns = 0`.
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
  When the measured latency *grows* mid-playback, `SynthCore.shift_pending()`
  retroactively moves all scheduled-but-unrendered note on/off events by the
  delta (test doubles without it get `_retime_pending`, which rewrites
  pending `_Ev._sent_when` entries), so already-queued notes stay in sync.
  `SynthCore.drop_scheduled()` / `SoundEngine.drop_scheduled()` forget all
  scheduled (not yet rendered) note on/off events while keeping kind-2
  damping events -- used by pause (`all_notes_off()` first, then drop).
  `AUDIO_LOOKAHEAD` (1.0 s) must
  cover device period + mixer + ring cushion + a stalled frame.
- Manual A/V trim: `VirtualKeyboard.av_offset` (s; >0 = sound later than
  image), set via CLI `--av-offset-ms` (overrides the env var when nonzero)
  or `PIANOPLAYER_AV_MS` (ms). Raise it when the picture runs ahead of the
  sound. Notes scheduled in the past are clamped and counted in
  `_late_notes` (printed at teardown when verbose).
- Dynamics: velocity comes from the score (`note21.volume.velocity` / 127,
  else `velocityScalar`) when present, else the heuristic fallback (melody
  louder than accompaniment, down-beat accent, thumb stronger, deterministic
  humanisation). Simple sustain: a note strictly overlapped by the next onset
  rings `SUSTAIN_LEGATO = 0.15` s longer (chord mates and abutting notes keep
  their durations).
- Env: 4 ms smooth-step attack, exponential decay per partial, fast damper on
  release, hammer felt transient (filtered noise burst baked into the voice).
- Backends in order: `miniaudio`, `sounddevice`, `none` (null backend keeps the
  clock running silently). Force `PIANOPLAYER_LATENCY_MS` env var to recalibrate.

## 3D player (source of truth)

`vkeyboard.py` drives the scene with a vedo timer but runs its **own** frame
loop (`_run_frames`) instead of `interactor.Start()` -- the blocking message
wait costs ~30 ms/frame and starves the audio ring. The camera is a locked
orthographic top-down view (`_lock_camera` swaps the interactor style); the
`v` key flips to a ~55° perspective view and back, `Space` (`toggle_pause`)
freezes score time by skipping `_on_timer()` while still pumping events +
re-rendering the frozen frame, shifting `_t0`/`_audio_epoch` forward by the
paused span plus a `RESUME_LEAD = 0.9` s pre-roll on resume (the score replays
the last 0.9 s covering the ~0.4 s ring cushion + 0.87 s pipeline, so re-sent
notes never reach the synth past due; fired flags prevent double-fire). Pausing also damps
the synth (`all_notes_off()` then `SynthCore`/`SoundEngine.drop_scheduled()`,
which forgets every queued note on/off but keeps the kind-2 damping events)
and rewinds each player's `p.sched` to `p.fire`, so after resume `_pump()`
re-sends the pending notes with the shifted epoch (no early sound, no silent
key presses; the already-rendered ring cushion of up to ~0.4 s is left alone
and still heard). While paused each frame runs `_paused_tick()`, which keeps
`engine.pump()` feeding the ring, refreshes the HUD (so `PAUSED` is visible
live), and re-renders; key auto-repeat is ignored via release tracking
(`KeyReleaseEvent` drops the key from `_keys_down`, `last` slides on every
event so a hold never ages out of the `KEY_DEBOUNCE` (0.30 s) window, plus a
1.2 s safety accepting a still-"down" key whose release was lost)
(`GetRepeatCount()` reads
1 even for the first press on some VTK builds, so it cannot be used). HUD shows effective BPM (`tempo*speedfactor`,
`effective_bpm()`) and `PAUSED`. All clocks are monotonic
(`perf_counter` for `_t0`, `engine_wall`, pump deadlines -- no `time.time()`
anywhere in the `_t0` family).

`play()` maps the window first (`initialize_interactor()` + `Render()` + 3x
`render()`) and only then starts the synth clock in `_prepare()`, so the
cold-start stall cannot drain the audio ring before note one (`_run_frames`
initialises nothing).

`hand3d.py` builds each hand from a lofted arm+palm surface, five tapered
finger tubes re-posed every frame with an analytic 3-segment IK
(`solve_chain`, also returns bend `b1`): MCP->PIP->DIP chain, PIP bends by `b1`,
DIP by `COUPLING * b1`, bisection on the chord length so the tip lands exactly
on the key. `update()` records per-finger `tip_err`/`max_bend`. Faces are wound
**inward** on purpose -- VTK would otherwise treat
outer faces as back-facing and shade them black; the left hand mirrors and
reverses face winding again for the same reason. The forearm yaws outward
around the wrist (`0.4 * atan2(root_x - shoulder_x, 45)`, clamped
±0.10 rad, plus velocity yaw clamped ±0.05).

Root follows hand size: `ROOT_Y`/`ROOT_Z` scale with `hf` (reach fraction
0.69, `ROOT_Z = 2.0*(0.50+0.60*hf) + 1.0*max(0,hf-0.5)` arch dome for larger
hands only -- small-hand IK fails from a high wrist, `BLACK_SHIFT = 1.15`), the thumb MCP
moves forward with `hf` (`THUMB_BASE_Y0 = 3.55`, `THUMB_FWD = 4.4`, thumb
contact 0.65 cm nearer the player) and the pinky contact retreats 0.80 cm
(`_PINKY_CONTACT_DY`). All geometry lives in `handspec.py` (single source:
finger base/lens/radii, `ARM_PROFILE` with a `check_palm_monotonic()`
gate, rest angles, thumb CMC targets, SDF constants, `geom_hash()`).
Palm/arm loft uses the `ARM_PROFILE` stations with a small `THENAR = 0.28`
thumb-side bulge; finger sections are elliptical (flatter back, belly pad,
flatter distal segment) with web thickening near
the knuckles. Idle fingers glide toward the fan (`SMOOTH_TAU = 0.05`,
max ~0.4 cm/frame) then relax 75% toward their natural parallel slots
(`_IDLE_NATURAL_LERP`, idle thumb splays `0.5*hf` outward) and extend forward
(`_IDLE_FWD`) at per-finger `REST_LIFT` heights with
a small `0.08*sin(2π*0.35*t)` breathing oscillation (offsets scaled by
`hf/0.82`); idle fingers also wander (`WANDER_AMP` 0.05 cm at 0.23 Hz), dip
with pressing neighbours (`SYMPATHY_DIP` 0.10 cm), and ride higher over
black-key areas (`WHITE_RIDE` 0.15 cm); the wrist bounces on quick re-strikes
(`BOUNCE_LIFT` 0.25 cm, tau 0.18 s) plus a mean-press loudness lift (0.12 cm)
-- all visual-only, IK targets stay reachable so `tip_err` is unchanged; fingers in the `held`/`aim_fingers` sets (see `vkeyboard._pump`)
keep exact targets and the root spring chases the pre-lerp fan mean, so
held fingers never drift. A free (idle-transit) finger whose key-x target
lands within `_COINCIDE_SEP = 1.4` cm (× `hf/0.82`) of an anchored
(held/aiming/pressed) finger steps aside to that distance and rides
`_COINCIDE_LIFT = 0.8` cm higher, passing over instead of driving its chain
through the anchored tube (bug3 melt); pushed fingers skip the idle lerp
(which would drag them back) and the lift is reach-capped in z so the short
thumb never dangles. Anchored fingers are never moved. Joint rings get darker vertex colours; nails are a subtle vertex-colour patch
(no mesh, no white dots); pressed fingers blush only near the tip
(`smoothstep(0.6,1.0,snorm)`) with contact AO baked into the pad side;
each hand carries one opaque blob shadow of the *skinned* mesh flattened
onto `z'=1.6` (world-projection, hidden in the perspective view). Skin material is ambient 0.45 / diffuse
0.65 / specular 0.15 (shininess 24) with side-edge darkening, FXAA on (MSAA stays 0). A finger holding a key (`press > 0.5`) whose `tip_err >
0.1 cm` makes the root chase with fast `TAU = 0.03` for that frame, and a
stretched hold (`tip_err > 0.3`) additionally drags the root target toward
its own key (proportional pull, ≤0.4) so legato stretches stay attached.

**Hand motion, step 1** (ported from `pygame-piano-animator/hands.py` +
`pianist.py` BEHAVIORS; idea-port, not line-port -- every knob lives in the
"motion step 1" block of `handspec.py`, all visual-only, and all deliberately
**excluded from `geom_hash()`** so retuning motion never rebuilds the mesh
cache; 2D inches/model units became cm × `hf/0.82` like `_REST_REF_HF`):

- *anticipation* (1a): `_pump` stamps `p.strike_in` = seconds until each
  aiming finger's next unfired strike (`t_on - now`, past-due notes read 0.0)
  and passes it as `rig.update(..., strike_in=...)` -- a new **optional**
  param, so old `update(dt, finger_x, press, ...)` callers, `settle()`,
  `reset()` and every test behave exactly as before. An aiming finger whose
  lead is `>= STRIKE_T` (0.12 s) waits at its raised PREP height
  (`HOVER + PREP_ADD[f]*rs`, hands PREP-HOVER) and makes the final drop to the
  key inside `STRIKE_T` with a quadratic (`s*s`) ease, so the tip is exactly
  on the key at `t_on`; without `strike_in` the old binary hover still
  applies. A *rising* target eases up over `RISE_TAU` (0.045 s ~ the hands
  `RELEASE_T` 0.12 s lift), a *descending* one (the strike drop, the key going
  down) is taken exactly, so contact timing and `tip_err` never lag. A
  released finger lingers over its key for `LINGER_T` (0.06 s) and only then
  drifts back toward its rest slot over `RETURN_T` (0.35 s) -- the
  `_free_since` stamp gates the idle lerp, and never-pressed fingers get the
  full legacy lerp (`s_ret = 1`).
- *turn before slide* (1b): the anticipation yaw leans toward where the root
  still has to travel (`YAW_SPLAY_GAIN` 0.05 rad per cm, clamped to
  `WRIST_DEV_DEG` (-26°, +8°)) and is smoothed with `YAW_SPLAY_TAU` 0.04 s,
  faster than the root `TAU` 0.065 s, so the turn visibly leads the slide
  (a 12 cm shift reads ~3.5°, the wrist limit is reached past ~9 cm of
  travel). It is exactly 0 once the hand has arrived and `settle()` zeroes
  it, so reach cells and the static poses are bit-identical. The shoulder-arm
  yaw keeps the hands `YAW_PRIOR` role (preferring the forearm's line);
  `SPLAY_LIMIT_DEG` / `SPLAY_COMFORT_DEG` / `REACH_MIN` / `REACH_COMFORT` /
  `CROSS_TURN_DEG` are recorded in `handspec.py` as the splay/reach envelope
  this maps onto (the 3D rig has no per-finger splay clamp of its own).
- *tendon link, idle fan, gestures* (1c): a curving leader drags its
  `TENDON_FOLLOWERS` (3↔4 both ways, 5 follows both) `TENDON_LINK` 0.6 of the
  way down with it, never below `TENDON_FLOOR` (0.38 cm above the key tops),
  idle fingers only -- a follower reaching for a key of its own ignores it --
  smoothed with `TENDON_TAU` 0.04 s (this replaces the old R5.6
  `TENDON_LIFT_*` lift constants, which the source has no counterpart for).
  Idle fingers fan between their busy neighbours (hands `SHAPE_FALLOFF` 0.6
  per finger of distance, capped at `SHAPE_CAP_X` 2.1 cm × rs, ramped in over
  `SHAPE_RAMP`; a busy finger inside a comfortable grip pulls nothing and the
  thumb's pull fades out as it passes the index slot), so the hand's shape
  follows the notes. Inside a monotone scale run the root spring eases from
  `TAU` to `RUN_GLIDE_TAU` 0.12 s over `RUN_RAMP_T` 0.15 s (wrist glides,
  fingers cross); alternations faster than `TREM_GAP_T` 0.3 s rock the forearm
  `TREM_ROLL_MAX` 0.07 rad after `TREM_MIN_NOTES` 4 onsets, striking side down,
  and a caller may add its own `roll=` on `update()`.

**Hand styles, step 2** (ported from `pygame-piano-animator/skins.py`; idea-port,
not line-port -- the source *draws* 2D top-down polygons, here the same looks
are expressed as a texture variant + vertex-colour bands + material
coefficients; every knob lives in the "Skins" block of `handspec.py`, all
deliberately **excluded from `geom_hash()`** -- a style is painted, never
re-meshed, and only the texture filename carries it):

- `HandRig(..., style=)` is a new **optional** kwarg (last position). `None`
  reads `PIANOPLAYER_HAND_STYLE`, then falls back to `cartoon`
  (`hand3d.style_from_env`, same pattern as `PIANOPLAYER_FALLING`); the CLI
  `--hand-style` overrides the env (`VirtualKeyboard.hand_style`). An unknown
  name normalises to `cartoon` (`handspec.normalize_style`).
- `cartoon` (default) is the look that shipped, **pixel-identical**: it keeps
  `SKIN` / `SKIN_PRESS` / the nail patch / `_SKIN_PRESS_MUL` / the blob shadow
  colour and the `SKIN_*` material, its texture variant draws the same marks
  from the same `(rest_pts, bones)` inputs (verified byte-identical PNG), and
  it skips the new code paths entirely (`_apply_skin_shading`, scalar
  visibility stays off, plain cache filename -- prebuilt package textures still
  hit). `TEX_VERSION` is **not** bumped. `geom_hash()` untouched.
- `gloves`: white latex texture variant (stitched knuckles instead of skin
  creases) + vertex-colour bands (`hand3d._apply_style_bands`): dark thin arm
  and a pale puffy cuff across the wrist, selected by rest-frame y.
- `robot`: white shell plates with dark seams and a gloss line per plate +
  dark metal joint cylinders (a *band* along each rest bone near either end --
  the bones run down the middle of the mesh, so a centre-distance ring never
  reaches the surface) and dark fingertip caps (a sphere on the last joint).
  Specular 0.30 / shininess 40: 0.48/64 blew the white shell to pure white and
  swallowed the very rings and caps the style is made of.
- `skeleton`: bones only -- forced to `mesh="tubes"` (the closest 3D equivalent
  of the 2D bone drawing; tubes already existed) with a dark palette; no
  texture is built.
- Because the skinned mesh leaves scalar visibility off (a pre-existing VTK
  quirk: `MapScalars` on, but the visibility flag off), a style that needs
  vertex colours turns it on (`mesh.mapper.SetScalarVisibility(1)`) -- only
  for non-default styles.
- The tube path (`_paint`, the fallback and the skeleton style) reads the same
  `self._look` palette, so a style looks right in either mesh mode.
- `handmesh.texture_path(hf, side, size, style)` appends the style name to the
  cache filename for non-default styles only (`handtex_<side>_<hf>_<hash8>_v3
  <style><size>.png`); `_cache_hash_of` still parses the hash, so
  `--prune-cache` keeps current-style files.

**Hand anatomy + flexibility, step 3** (ported from
`pygame-piano-animator/pianist.py` `DEFAULT_ANATOMY` (19 bones) + its `wrist`
width slider + the `BEHAVIORS` technique sliders; idea-port, not line-port:
the 2D model *fits* a wrist to 19 adjustable bone lengths, the 3D rig solves
a fixed 18-bone skeleton, so **no bone was added** -- the step surfaces the
proportions and the flexibility as named knobs):

- *Anatomy block of `handspec.py`*, deliberately built from **derivations of
  the hashed geometry constants** (`FINGER_SPECS` lens/radii/base,
  `ARM_PROFILE`, `THUMB_CMC_*`, `thumb_base_y()`), never new constants:
  - `ANATOMY_BONES` / `BONE_KIND` -- the source's 19 bones (5 metacarpals +
    14 phalanges, the thumb having no middle phalanx) and their names. How
    they map onto the rig's 18 skinned bones (`handmesh.rest_bones`):
    `forearm` is the radius (no source bone), `palm` carries all five
    metacarpals, `f1_cmc` is the thumb's own thenar -> MCP span (mc1 given a
    bone of its own), `f{2..5}_0/1/2` are pp/mp/dp, `f1_0/1/2` are pp1/dp1
    plus the thumb pad past the IP joint.
  - `bone_lengths()` -- this rig's `DEFAULT_ANATOMY` view: a metacarpal is the
    wrist -> MCP distance, the thumb's mc1 the whole `THUMB_CMC_A` -> thumb-MCP
    span (the rig carries the CMC as its own bone). They read ~10 cm against
    the source's ~7 cm because `WRIST_Y` is the centre of the wrist loft,
    ~2.7 cm proximal to the joint the source measures from; the phalanges (the
    part the solver flexes) match within a few tenths of a cm.
  - `segment_ratio(f)` -- per-finger proximal/middle/distal proportions
    normalised to 1 (`SOURCE_ANATOMY` + `SOURCE_UNIT_CM` 1.09 hold the
    source's own numbers for comparison).
  - `knuckle_row()` -- the MCP row `{f: (x, y, z)}`; the z column is the
    knuckle dome (middle +0.35 cm over little). `hand3d._SPECS` now reads its
    bases from here (same `mcp_base()` values, anatomical name).
  - `wrist_width()` -- the wrist half-width row of `ARM_PROFILE` (the source's
    `wrist` slider scales it; hashed with the rest of the profile).
  - Everything above is hashed *by construction* (it *is* the hashed geometry),
    so no new hash input was needed and `geom_hash()` stayed `d055cc04` --
    which is also why `HANDMESH_VERSION` was **not** bumped. Editing a bone
    still rebuilds the mesh cache.
- *Flexibility / behaviour block of `handspec.py`* -- visual-only, **excluded
  from `geom_hash()`**, defaults = the values the rig already used (probe vs
  `1561c09`: pose + motion MD5s bit-identical):
  - `FLEX_BEND` / `FLEX_COUPLING` -- per-finger share (all 1.0) of the rig-wide
    `HandRig.BEND_MAX` (PIP flexion limit) and `HandRig.COUPLING`
    (DIP = coupling x PIP), applied where `update()` calls `solve_chain`.
    Overriding the class attrs still behaves exactly as before.
  - `HOVER_GAP` (was `HandRig.HOVER` 0.55, a world cm like `KEY_SINK`),
    `HOVER_ADD` (per-finger extra waiting height, 0 = shipped; the raised
    striking-ready counterpart is step 1's `PREP_ADD`) -- both feed the hover
    and the PREP height.
  - `IDLE_LIFT` / `IDLE_PULL_Y` / `IDLE_PULL_Z` -- the idle curl height and
    the retreat toward the wrist. `hand3d.update()` reads them **live** (the
    old module-level `_REST_LIFT` / `_REST_PULL_Y` / `_REST_PULL_Z` copies were
    dropped: a value snapshotted at import is not a knob). Do not confuse
    `IDLE_LIFT` with the hashed `REST_LIFT_Z` (the rest-pose target lift of
    the mesh).
  - `BEHAVIOR_PORT` -- where each source slider landed (and the few
    deliberately not ported: `max_speed`, `key_area_*`, `roll_speed`,
    `gesture_finger_action` need signals the rig does not get).

**2D hand render backend** (`pianoplayer/hands2d/`, default `--hands 2d`):
`hands.py` (pose solver / gestures / crossings), `skins.py`,
`fingering.py`, `figures.py`, `glissando.py`, `pianist.py` and `progress.py`
are vendored **verbatim** from `D:\Project\pygame-piano-animator` (only the
import lines were rewritten to package-relative form; a diff of the 7 files
shows nothing but import lines), next to three shims -- `midi_loader.py`
(`Note`/`TrackInfo`/`MidiSong`/`is_black_key`/`note_name`/pedal constants,
**no `pretty_midi`**), `common.py` (`KEY_LEN_WW`, `_BLACK_OFFSETS`, the
realistic `Keyboard.layout` only) and `paths.py` (`DATA_DIR`, a string under
`~/.cache/pianoplayer/hands2d`) -- and the adapter `render.py::Hands2D`.

The adapter is the only new logic: it turns the players' `_Ev` lists into
2D `Note`s (`pitch/start/end/velocity/hand/finger`, a note without a finger
gets `3`), passes `fingering={id(n): finger}` so the 2D planner, `figures` and
`pretty_midi` are never used, and creates one `HandAnimator` per hand with
`Pianist("Default", DEFAULT_ID, skin=skins.default_skin(style))`. The pixel
grid is the window's own: `ppc = win_w / (kb_w + 2*KB_MARGIN_X)`,
`px(x) = (x - x0)*ppc`, `py(y) = (y1 - y)*ppc` with
`x0/x1 = cx ∓ (kb_w/2 + KB_MARGIN_X)`, `y0 = KB_BOTTOM_Y`, `y1 = cy + pscale`.
The shim `Keyboard`'s rect is built with `rect_h = (4.0*ppc)/0.37` (the 4 cm
between the white- and black-key fronts, as in 3D) and `rect.bottom` pinned to
`py(-8.0)`, the 3D white keys' front; every `key_rects[midi].centerx` is then
overridden with `px(_key_x(midi))` (the 2D black-key offsets are discarded) and
`white_w = KEY_WB*ppc`, so a fingertip aimed at a key lands on that key's own
pixel.

The overlay surface is cropped to a horizontal **band** covering the world-y
range `KB_BOTTOM_Y .. BAND_TOP_Y` (`BAND_TOP_Y = 14.0 cm`, about 44% of the
window height on a 1785x1020 screen); the hands never paint above it, so
nothing above `y = +14 cm` is clipped (`test_hands2d.py` asserts the band's
top rows stay empty on every frame). Drawing goes to an offscreen
`pygame.Surface` (SRCALPHA, no display, no `pygame.init()`, no mixer) laid out
on the window's own pixel grid; the upload reverses the rows (pygame counts
down, a VTK image counts up) into a reused `(band_h, W, 4)` uint8 array and a
`vtkImageData` (`spacing 1/ppc`, origin `(x0, y0, 2.2)`, RGBA) shown by a
`vtkImageActor` added to the same renderer -- it alpha-composites over the 3D
keys and is hidden in the perspective (`v`) view.

`vkeyboard.VirtualKeyboard(hands=...)` (env `PIANOPLAYER_HANDS`, flag
`--hands`) picks the backend; anything but `2d` is `3d`. In `2d` mode
`build_RH`/`build_LH` skip the `HandRig` (`self.RH`/`self.LH` stay `None`),
`_prepare` still builds a rig-less `_Player` per side (keys and audio are
unchanged) and builds + warms the overlay *before* `_t0`, and `_frame` replaces
`p.rig.update(...)` with `Hands2D.update(now, dt)`. Any failure hides the
overlay and never stops playback; no pygame/VTK image means one warning and a
fallback to `3d` (with the rigs built then). Tests that read rig internals
(`test_play.py`, `test_regression.py`) are pinned to `hands="3d"`;
`test_hands2d.py` covers the 2D backend.

**Performance**: on a 1785x1020 window, `prepare_hands`+pose ~10-30 ms,
`draw_hands` ~14 ms, overlay upload ~16 ms, but VTK `render()` of the whole
scene (keys + HUD + falling bars) is ~80-150 ms, so the default 2D backend
runs ~5-10 fps vs ~63 fps for the 3D rig. The band crop saves texture/upload
cost (~4x fewer pixels than full window); `--hands 3d` / `PIANOPLAYER_HANDS=3d`
is the escape hatch for smooth playback.

Key dip: keys hinge around `KEY_PIVOT_Y`, drop = `(KEY_PIVOT_Y - y) * sin(angle)`.

**Playback timing** (`_events_for` / `_beat_seconds`): note onsets and
durations are read from `time_real` / `dur_real` (true quarter lengths,
ties included) and converted to seconds through `tempo_map`
(`[(offset_quarters, bpm), ...]`, `utils.score_tempo_map()` honors
`MetronomeMark.getQuarterBPM()` so referents like ♪=120 or ♩.=60 are
correct, and tempo changes are integrated piecewise). With no map, the
constant `tempo` is used. Playback rate only ever flows through
`speedfactor` (`set_tempo_bpm`/`effective_bpm()`): the tempo popup
(`prompt_tempo`, tkinter.simpledialog, score BPM as default, skipped when
offscreen, `--tempo` given, or `PIANOPLAYER_TEMPO_POPUP=0`) and the CLI
`--tempo BPM` flag (multiplies `--vedo-speed`) never touch A/V sync math. When the piece ends the score clock freezes at the total and the window stays open until the user closes it. While a finger holds a key, that finger stays
exactly on the pressed key's x (`_Ev.wx`) even though the hand aims at
the next note. Visual presses fire `PRESS_LEAD = 0.025` s early so the
key/finger travel lands with the heard attack (audio scheduling stays on
true score time); each hand carries a single opaque blob shadow.
Repeated same-pitch notes get a visual release `v_off` earlier than the audio
`t_off` (`_apply_visual_release`), so the key and the finger visibly rise
between re-strikes while the sound keeps its full length.

**Falling notes** (`pianoplayer/falling.py`, `FallingNotes` in `vkeyboard.py`):
Synthesia-style bars fall from the top of the window and touch the landing
line `HIT_Y` (top edge of the fall-line box) exactly at `t_on`; bar length
follows `v_off`, abutting bars stay `BAR_GAP` apart. Bars are visible
`FALL_SECONDS` (2.2 s) ahead, so `LEAD_IN = max(0.8, FALL_SECONDS + 0.2)`.
Digits 1-5 are flat quad geometry (not Text3D/texture) at the bottom of each
bar. Fixed-capacity meshes, numpy-view writes + `Modified()` per frame, any
error hides the layer (`hide()`) and disables it, never playback. Black-key
bars ride above white-key bars and left-hand bars above right-hand bars
(`BAR_Z`/`BAR_Z_BLACK`/`BAR_Z_LEFT`, digits at `DIGIT_Z`). Disable with
`PIANOPLAYER_FALLING=0`.

## Unified skinned hand mesh (Tier B, v2.6.0)

**Rest mesh** (`handmesh.build_rest_mesh`): One smooth-unioned mesh per hand
built from a signed-distance field (SDF) in wrist-local rest pose (F1: slight
curl + spread solved by `solve_chain` with `coupling = REST_DIP_RATIO`,
same as the rig). Primitives: forearm/palm ellipsoid chain from
`ARM_PROFILE` stations (2 farthest replaced by 5 unscaled stubs) + thenar
pair + hypothenar + 4 MCP mounds, per-finger 3 elliptical capsules +
MCP/PIP/DIP spheres + flattened pad-tip ellipsoid, thumb CMC capsule.
Combination uses `smin` (`-k*logaddexp`): intra-finger `k=0.05`,
finger/body joins `k=0.35`, thumb-CMC/body `k=0.40`, body chain `k=0.30`;
y-dependent webbing (`smin k=0.25` inside the web band, plain `min()`
outside so neighbours touch but never fuse, F2); thumb<->index web `k=0.35`.
Finger radii taper distally (`FING_R_TAPER`: DIP ×0.85, tip ×0.80; MCP/PIP
untouched so the graded proximal section stays put) and the thumb runs
slimmer proximally (1.06/0.96 → 0.98/0.88) to carve the thenar groove; the
web band tops at `WEB_Y_HI = 0.6` past the knuckles (was 1.2) so fusion
stays proximal like a real web. Distal gaps at M: ~0.3-1.0 cm (SDF
`CC@85%` = 4); proximal gaps stay webbed (`CC@50%` = 1, `gap50` ≤ 0.15).
Grid step 0.15 cm over adaptive bounds; `vtkFlyingEdges3D` iso=0 →
`vtkWindowedSincPolyDataFilter` (15 iter) → `vtkQuadricDecimation`
(≤8000 tris, `TRIS_MAX`) → normals. Cached to
`~/.cache/pianoplayer/handmesh_{side}_{hf:.2f}_{geomhash}.npz`
(`get_rest_data` stores the full set there: rest_pts, faces, rest_nrm plus
`idx`/`w` weights; `build_rest_mesh` alone stores the first three).
First build ~4.7-7.2 s at size M (~8-13 s at L/XL/XXL on this
machine), cached load ~40 ms for the full mesh + weight set. Left hand mirrors x +
reverses face winding (same as `_arm_surface`). `geom_hash()` covers
constants only -- geometry logic changes need a manual `HANDMESH_VERSION`
bump (v5: distal taper + slimmer thumb + proximal-only webbing; v4: thumb
MCP base matches the rig, rest coupling 0.65).

**Skinning weights** (`handmesh.compute_weights`): 18 bones (forearm, palm,
f1_cmc + 3 per finger). Distance-to-segment `d`; weight `w = exp(-((d-d_min)/0.45)^2)`;
top-4 renormalised. Verts with nearest bone in {forearm,palm,f1_cmc} and
runner-up >1.0 cm away get hard weight 1.0 (rigid, exact). Cached in same
.npz as `idx (N,4) int16` + `w (N,4) float32`.

**Per-frame LBS** (`HandRig._update_skin` in `hand3d.py`): Reuses already-solved
IK chains (`_skin_q[f]`). Per bone: rest frame `[X0 T0 Z0]` (from `rest_frames`),
current frame `[X T Z]` (from `_frames` on current bone dirs). `R_b = [X T Z]
@ [X0 T0 Z0].T`, `v' = P_b + R_b @ (v - P0_b)`. Vectorised over all vertices:
rigid verts (1.0 weight) get single transform; non-rigid use 4-bone blend via
`einsum` (no Python loops). Skinning ~2.7 ms/hand (M size, 3002 verts). Tip
accuracy vs IK `q[3]` ≤0.05 cm in press/rest poses; L/R mirror symmetry holds.

**Procedural skin texture** (`handmesh.build_skin_texture`): 512² default,
1024² via `PIANOPLAYER_TEXSIZE=1024` (own versioned cache file), PNG written
to cache dir (`handtex_{side}_{hf:.2f}_{geomhash}_{TEX_VERSION}{style}{size}.png`,
the `{style}` marker empty for the default so existing caches and the prebuilt
package textures keep hitting), loaded via path; `texture_path` ctor param
overrides with user photo (same planar UV). Base tone `(0.95,0.80,0.70)` ±3%
two-octave value noise (PIL upscaled). Joint-crease bands (~12% dark + light
ridge) at each finger's MCP/PIP/DIP v-rows across its u-span. 2–3 palm-crease
arcs. Per-fingertip nail ellipse (`(0.97,0.82,0.74)` → whiter tip + cuticle
arc). Fingertip/palm-edge blush + pore speckle. Planar UV from rest bounds
(`u=(x-xmin)/W`, `v=(y-ymin)/H`), static TCoords at build. `Mesh.texture(path)`
applied; the actor colour goes white only when the texture loads --
a missing/garbage texture keeps the SKIN colour (fallback). A `style=` variant
swaps the palette and which marks are drawn from the same
`STYLE_TEX` table (`handmesh.STYLE_TEX`); the `cartoon` entry reproduces the
above byte-for-byte (checked by regenerating and diffing the PNG).

**API**: `HandRig(mesh="tubes"|"skinned", texture_path=None, style=None)`
default flips to `"skinned"` after acceptance; tubes stay as automatic fallback
(try/except + one-line warning). `style` is the hand look (see "Hand styles,
step 2"), default `cartoon`; CLI/GUI add `--hand-style` / `PIANOPLAYER_HAND_STYLE`.

## Conventions

- Version is defined in **two** places that must stay in sync:
  `setup.py` (`version='2.9.3'`) and `pianoplayer/__init__.py`
  (`__version__ = "2.9.3"`).
- `output.xml` is the default annotated output (MusicXML with fingering
  articulations / lyrics).
- Beam numbering: right hand = 0, left hand = 1 (`-rbeam` / `-lbeam`).
- Fingering output: `articulations` (musescore-friendly) by default;
  `lyrics` with `-b/--below-beam`.
- R0 IoU gate (`diag_ref_iou.py`, never asserts): needs local-only
  `docs/ref_top.<jpg|png>` (user photo, `docs/ref_*` is gitignored) +
  `docs/ref_mask.png` via `tools/make_ref_mask.py`. Baseline 2026-10-13:
  **0.772** @rot=180 (chromatic-cluster pose) vs ≥ 0.85 target; gap is
  personal slender proportion (photo bbox h/w 1.79 vs 1.64), not a mesh
  defect -- R3 asset replacement deferred, see plan Deviations.

## Docs maintenance

**When source changes, update this file (`AGENTS.md`) and `README.md` together.**
The algorithm / audio / 3D notes above mirror `hand.py`, `audio.py`, `vkeyboard.py`
and `hand3d.py` -- if those change, the matching section here must change too.
