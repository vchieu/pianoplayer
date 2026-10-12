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
python test_fingering.py      # fingering validity (fingers 1..5, held-finger, chord order) + 500-note solve time
python test_hand_reach.py     # fingertip reach/bend over all sizes/fingers/keys (70 cells)
python test_hand_look.py      # Tier A hand-look acceptance: tilt/gap/ratio/bend, offscreen
python test_sync.py           # A/V sync incl. mid-playback output-latency jump (FakeEngine)
```

There is no linter, formatter, CI config, or pyproject; `setup.py` is the only
build manifest.

## Module map

| File | Role |
|------|------|
| `bin/pianoplayer` | CLI + Tkinter GUI entry point (installed as the `pianoplayer` script) |
| `pianoplayer/scorereader.py` | `reader()` -> music21 Stream to `INote` list (ties absorbed into the first note, simultaneous voices kept, `time_real`/`dur_real` carry the true onsets for playback; a chord reduced to one sounding pitch is not `isChord`; only numeric 1..5 lyrics become finger hints; single-note parts allowed); `PIG2Stream()` for PIG `.txt` |
| `pianoplayer/hand.py` | **Core algorithm** -- `Hand` finds the fingering |
| `pianoplayer/hand3d.py` | Procedural anatomical hand, analytic IK (`HandRig`) |
| `pianoplayer/handmesh.py` | Unified skinned hand mesh (Tier B): SDF rest mesh, LBS weights, procedural skin texture |
| `pianoplayer/audio.py` | Real-time additive piano synth (`SoundEngine`, `SynthCore`) |
| `pianoplayer/vkeyboard.py` | 3D scene, frame loop, A/V sync, HUD (`VirtualKeyboard`) |
| `pianoplayer/utils.py` | Keyboard geometry, note-name helpers, shared constants |
| `pianoplayer/wavegenerator.py` | Legacy simpleaudio chord player (deprecated) |

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
crossings, and (inside chords) finger-order violations plus a set of
normalized stretch thresholds (`axba = |dx| * hf / 0.8` vs. finger-pair
limit). Same finger on different keys is HARD only while the first note is
still held or the gap is at most 0.25 s; slower reuse pays a soft
`SAME_FINGER_BASE + 0.15 * |dx|` (F3). Soft (added cost, still losable) rather than pruning, so a legacy
finger 0 or an impossible spot never kills a whole solution.

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
`v` key flips to a ~55° perspective view and back. All clocks are monotonic
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
0.69, `ROOT_Z = 2.0*(0.50+0.60*hf)`, `BLACK_SHIFT = 1.15`), the thumb MCP
moves forward with `hf` (`THUMB_BASE_Y0 = 3.4`, `THUMB_FWD = 4.4`, thumb
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
`hf/0.82`); fingers in the `held`/`aim_fingers` sets (see `vkeyboard._pump`)
keep exact targets and the root spring chases the pre-lerp fan mean, so
held fingers never drift. Joint rings get darker vertex colours; nails are a subtle vertex-colour patch
(no mesh, no white dots); pressed fingers blush only near the tip
(`smoothstep(0.6,1.0,snorm)`) with contact AO baked into the pad side;
each hand carries one opaque blob shadow of the *skinned* mesh flattened
onto `z'=1.6` (world-projection, hidden in the perspective view). Skin material is ambient 0.40 / diffuse
0.70 with side-edge darkening, FXAA on (MSAA stays 0). A finger holding a key (`press > 0.5`) whose `tip_err >
0.1 cm` makes the root chase with fast `TAU = 0.03` for that frame, and a
stretched hold (`tip_err > 0.3`) additionally drags the root target toward
its own key (proportional pull, ≤0.4) so legato stretches stay attached.

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

## Unified skinned hand mesh (Tier B, v2.5.0)

**Rest mesh** (`handmesh.build_rest_mesh`): One smooth-unioned mesh per hand
built from a signed-distance field (SDF) in wrist-local rest pose (F1: slight
curl + spread solved by `solve_chain`). Primitives: forearm/palm ellipsoid
chain from `_ARM` stations (proximal stations replaced by 5 unscaled stubs),
per-finger 2 tapered capsules + PIP joint sphere, thumb CMC capsule. Combination
uses `smin` (-k*logaddexp) for intra-finger (k=0.05), finger↔palm/thenar
(k=0.47), thumb-CMC↔palm (k=0.40), body chain (k=0.30); plain `min()` for
finger↔finger so neighbours touch but never fuse (F2). Grid step 0.15 cm over
adaptive bounds; `vtkFlyingEdges3D` iso=0 → `vtkWindowedSincPolyDataFilter`
(15 iter) → `vtkQuadricDecimation` (≤6000 tris) → normals. Cached to
`~/.cache/pianoplayer/handmesh_{side}_{hf:.2f}_v1.npz` (rest_pts, faces,
rest_nrm). First build ~4.3 s, cached load ~0.04 s. Left hand mirrors x +
reverses face winding (same as `_arm_surface`).

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

**Procedural skin texture** (`handmesh.build_skin_texture`): 512² PNG written
to cache dir (`handtex_v2.png`), loaded via path; `texture_path` ctor param
overrides with user photo (same planar UV). Base tone `(0.95,0.80,0.70)` ±3%
two-octave value noise (PIL upscaled). Joint-crease bands (~12% dark + light
ridge) at each finger's MCP/PIP/DIP v-rows across its u-span. 2–3 palm-crease
arcs. Per-fingertip nail ellipse (`(0.97,0.82,0.74)` → whiter tip + cuticle
arc). Fingertip/palm-edge blush + pore speckle. Planar UV from rest bounds
(`u=(x-xmin)/W`, `v=(y-ymin)/H`), static TCoords at build. `Mesh.texture(path)`
applied.

**API**: `HandRig(mesh="tubes"|"skinned", texture_path=None)` default flips to
`"skinned"` after acceptance; tubes stay as automatic fallback (try/except +
one-line warning). CLI/GUI flags unchanged.

## Conventions

- Version is defined in **two** places that must stay in sync:
  `setup.py` (`version='2.5.0'`) and `pianoplayer/__init__.py`
  (`__version__ = "2.5.0"`).
- `output.xml` is the default annotated output (MusicXML with fingering
  articulations / lyrics).
- Beam numbering: right hand = 0, left hand = 1 (`-rbeam` / `-lbeam`).
- Fingering output: `articulations` (musescore-friendly) by default;
  `lyrics` with `-b/--below-beam`.

## Docs maintenance

**When source changes, update this file (`AGENTS.md`) and `README.md` together.**
The algorithm / audio / 3D notes above mirror `hand.py`, `audio.py`, `vkeyboard.py`
and `hand3d.py` -- if those change, the matching section here must change too.
