# PianoPlayer

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/fb513fa9c1a34f9988ec5dc443cbe633)](https://app.codacy.com/app/marcomusy/pianoplayer?utm_source=github.com&utm_medium=referral&utm_content=marcomusy/pianoplayer&utm_campaign=Badge_Grade_Settings)
[![Downloads](https://pepy.tech/badge/pianoplayer)](https://pepy.tech/project/pianoplayer)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![DOI](https://zenodo.org/badge/107160052.svg)](https://zenodo.org/badge/latestdoi/107160052)


Automatic piano fingering generator. <br />
Find the optimal fingering combination to play a piano score.
Optionally visualize it in 3D with [vedo](https://github.com/marcomusy/vedo).<br />

## Download and Install:
```bash
pip install pianoplayer
```
The 3D player comes with a built-in piano synthesiser: notes are rendered
on the fly by additive synthesis, so there are **no sound samples to download**
and chords are played in tempo. If no audio device is available the
animation simply runs silent.

Requirements: `music21`, `vedo` (+`vtk`), `numpy` and `miniaudio`.

#### Installing without 3D rendering
To only install the core functionality and skip everything else, use the following:
```bash
pip install music21
pip install --no-deps pianoplayer
```

To visualize the output annotated score install the latest
[musescore](https://musescore.org/en/download), or any other renderer
of [MusicXML](https://en.wikipedia.org/wiki/MusicXML)
files.

**Windows 10** users can place this file:
[pianoplayer.bat](https://github.com/marcomusy/pianoplayer/blob/master/pianoplayer.bat)
on their desktop and double click it: it finds Python and PianoPlayer by
itself, there is nothing to edit.
Check out [this link](https://github.com/marcomusy/pianoplayer/issues/27) in case of
installation problems.


## CLI Usage:
Example command line from terminal:<br />
`pianoplayer scores/bach_invention4.xml --verbose -n10 -rvzm`<br />
will find the right hand fingering for the first 10 measures,
pop up a 3D rendering window and invoke *musescore*.

Add `-v` to watch & listen to the result: the 3D player animates both hands
on the keyboard and plays the score with the built-in synthesiser, at the
tempo written in the score. Use `--vedo-speed 1.5` to play it faster and
`-z` to mute it. The window stays open until you close it.

The output is saved as a [MusicXML](https://en.wikipedia.org/wiki/MusicXML)
file with name `output.xml`.<br />

```bash
pianoplayer         # if no argument is given a GUI will pop up (on windows try `python pianoplayer.py`)
# Or
pianoplayer [-h] [-o] [-n] [-s] [-d] [-k] [-rbeam] [-lbeam] [-q] [-m] [-v] [--vedo-speed]
            [-z] [-l] [-r] [-XXS] [-XS] [-S] [-M] [-L] [-XL] [-XXL]
            filename
# Valid file formats: MusicXML, musescore, midi (.xml, .mscz, .mscx, .mid)
#
# Optional arguments:
#   -h, --help            show this help message and exit
#   -o , --outputfile     Annotated output xml file name
#   -n , --n-measures     [100] Number of score measures to scan
#   -s , --start-measure  Start from measure number [1]
#   -d , --depth          [auto] Depth of combinatorial search, [2-9] (legacy window mode only)
#   -rbeam                [0] Specify Right Hand beam number
#   -lbeam                [1] Specify Left Hand beam number
#   --quiet               Switch off verbosity
#   -m, --musescore       Open output in musescore after processing
#   -b, --below-beam      Show fingering numbers below beam line
#   -v, --with-vedo       Play the 3D scene with sound after processing
#   --vedo-speed          [1.0] Playback speed factor (animation and sound)
#   --tempo               [off] Playback tempo in BPM (multiplies --vedo-speed; skips the tempo popup)
#   --av-offset-ms        [0] Sound-minus-image latency in ms (overrides PIANOPLAYER_AV_MS)
#   -z, --sound-off       Disable sound
#   -l, --left-only       Fingering for left hand only
#   -r, --right-only      Fingering for right hand only
#   -XXS, --hand-size-XXS Set hand size to XXS
#   -XS, --hand-size-XS   Set hand size to XS
#   -S, --hand-size-S     Set hand size to S
#   -M, --hand-size-M     Set hand size to M
#   -L, --hand-size-L     Set hand size to L
#   -XL, --hand-size-XL   Set hand size to XL
#   -XXL, --hand-size-XXL Set hand size to XXL
```

### GUI Usage:<br />
Just type `pianoplayer` in a terminal
(or double click the
[pianoplayer.bat](https://github.com/marcomusy/pianoplayer/blob/master/pianoplayer.bat) file),
then:

![newgui](https://user-images.githubusercontent.com/32848391/63605343-09365000-c5ce-11e9-97b8-a5642e71ca24.png)

- press **Import Score** (valid formats: *musescore, MusicXML, MIDI, [PIG](http://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/)*)
- press **GENERATE** (`output.xml` is written)
- press **Musescore** to visualize the annotated score
- press **3D Player** to show the animation (Press `Esc` to quit the application)


#### Example output, as displayed in *musescore*:

(If fingering numbers are not visible enough try `-b` option.)


![bachinv4](https://user-images.githubusercontent.com/32848391/63605352-10f5f480-c5ce-11e9-8b00-34f1adc2e79b.png)


![pianoplayer3d](https://user-images.githubusercontent.com/32848391/63605322-0176ab80-c5ce-11e9-8213-b572d0303523.gif)


## The 3D Player:
Pressing **3D Player** (GUI) or adding `-v` (CLI) opens the animated scene:

- **flat top-down view**: the full 88-key keyboard runs along the bottom of the
  window and the whole area above it is left free for the notes to fall into,
  the way Synthesia lays it out;
- **fixed camera**: the view is a locked, orthographic look straight down, so
  dragging the mouse never moves it; press `v` to flip to a ~55° perspective
  view and back; `Esc`/`q` or the close button quits;
- **procedural hands**: palms, joints, finger pads and nails are built from
  primitives and re-posed every frame by a small analytic IK solver, so the
  fingers arch and flex like real ones and the thumb reaches the black keys.
  Idle fingers stay parallel and extended with splayed thumbs; a pressed
  finger only blushes slightly at the tip, nails are a subtle patch of
  vertex colour, and each hand carries a soft projector shadow on the keys
  (hidden when you flip to the perspective view); edges are smoothed with
  FXAA;
- **unified skinned mesh**: the five detached finger tubes are replaced by
  one smooth-unioned mesh per hand (elliptical finger sections, pad tips,
  webbing, SDF rest pose, linear-blend skinning from the same IK chains),
  with per-size/side procedural skin texture (tone noise, joint creases,
  nails, fingertip blush; 512 px default, 1024 via `PIANOPLAYER_TEXSIZE=1024`,
  versioned cache name so look upgrades rebuild once) and a press blush baked into vertex colours
  (multiplied with the texture). Tubes remain as automatic fallback.
  The rest mesh is cached under a geometry-hash name (first build
  ~4.7-7.2 s at size M, ~8-13 s at L/XL/XXL on this machine; the full
  mesh + skinning-weight set is cached, so a warm start loads in ~40 ms
  instead of rebuilding); skinning runs in ~2.7 ms/frame per hand.
  Skin material is ambient 0.45 / diffuse 0.65 / specular 0.15 (shininess 24)
  with a softened blob shadow; idle fingers wander/breathe, dip with
  pressing neighbours, ride higher over black-key areas, and the wrist
  bounces on quick re-strikes plus a small loudness lift (all visual-only,
  IK fingertip targets stay exact);
- **hand motion, step 1** (ported from the 2D pygame-piano-animator pose
  solver; idea-port, all visual-only):
  - *anticipation*: the player knows how long until each finger's next strike,
    so a finger about to play lifts to a raised "ready" height early, hovers
    there and then drops onto its key in the last 0.12 s with an accelerating
    ease -- the tip is exactly on the key at the moment it sounds;
  - *linger and return*: a released finger stays over its key for a moment
    before drifting back to its resting slot, instead of snapping away;
  - *turn before slide*: a hand that has to travel a long way turns at the
    wrist first (within the wrist's own range) and slides second;
  - *linked tendons + idle fan*: a finger curving down to a black key takes
    its neighbours partly with it, and the idle fingers fan out between the
    busy ones, so the hand's shape keeps following the notes;
  - *gestures*: the wrist glides through scale runs, rocks the forearm in fast
    repeated-note (tremolo) figures, and bounces on repeated chords;
- **hand styles** (ported from the same 2D project's skin styles; idea-port --
  the source draws 2D top-down polygons, here the looks are expressed as
  texture variants + vertex-colour bands + material coefficients, so nothing
  about the geometry or the IK changes): `cartoon` (default -- skin tone,
  creases, nails; exactly the look that shipped), `gloves` (white latex with
  stitching, dark thin arm and a pale puffy cuff), `robot` (white shell
  plates with dark metal joint cylinders, dark fingertip caps and a gloss
  highlight) and `skeleton` (bones only: dark tubes). Pick one with
  `--hand-style` (CLI) or `PIANOPLAYER_HAND_STYLE`; each non-default style
  gets its own texture cache file, and `test_hand_render.py` renders one
  offscreen frame per style;
- **hand anatomy + flexibility, step 3** (ported from the same 2D project's
  per-bone anatomy table and technique sliders; idea-port -- the 2D model
  fits a wrist to 19 adjustable bone lengths, the 3D rig solves a fixed
  18-bone skeleton, so nothing gained a new bone):
  - the 19 bones are exposed under anatomical names in `handspec.py`
    (`bone_lengths()` for the source's per-bone view, `segment_ratio()` for
    each finger's proximal/middle/distal proportions, `knuckle_row()` for the
    MCP row and its knuckle-dome heights, `wrist_width()` for the wrist
    slider). They are *derived* from the same geometry constants, so
    retuning a bone still rebuilds the cached mesh (`geom_hash()`);
  - how flexibly each finger folds and how high it waits above the keys are
    now named knobs: `FLEX_BEND` / `FLEX_COUPLING` (per-finger share of the
    PIP flexion limit and of the DIP-follows-PIP ratio), `HOVER_GAP` /
    `HOVER_ADD` / `PREP_ADD` (waiting and strike-ready heights) and
    `IDLE_LIFT` / `IDLE_PULL_Y` / `IDLE_PULL_Z` (the idle curl). Defaults are
    exactly the values that shipped, so the rig is bit-identical until you
    change one -- and none of them are inside `geom_hash()`, because how a
    finger folds is not a shape;
  - `BEHAVIOR_PORT` in `handspec.py` records where every one of the 2D
    model's technique sliders landed here (and which few were deliberately not
    ported, with the reason);
- **keys dip** on their hinge when a finger presses them, and flash with an
  orange (right hand) or violet (left hand) accent; repeated same-pitch
  notes get a visual release `v_off` earlier than the audio `t_off`, so the
  key and the finger visibly rise between re-strikes while the sound keeps
  its full length;
- **falling notes** (`pianoplayer/falling.py`): Synthesia-style bars fall
  from the top of the window and touch the landing line `HIT_Y` exactly at
  `t_on`; bars are visible `FALL_SECONDS` (2.2 s) ahead, so
  `LEAD_IN = max(0.8, FALL_SECONDS + 0.2)`; every bar carries its finger
  number 1-5 as flat quad geometry (not Text3D/texture); black-key bars ride
  above white-key bars and left-hand bars above right-hand bars, and any
  layer error hides it instead of freezing bars. Disable with
  `PIANOPLAYER_FALLING=0`. Geometry is covered by `test_falling.py` and
  the repeat release by `test_repeat_notes.py`;
- **tempo comes from the score**: every note - including chords - sounds at
  its exact onset time, tempo marks with any referent (eighth = 120,
  dotted quarter = 60, ...) are interpreted correctly and tempo changes
  inside the score are honored; `--vedo-speed` scales both animation
  and sound. At playback start a popup offers the score's BPM as the
  default (OK keeps it, typing a number rescales via `speedfactor`);
  `--tempo BPM` sets it non-interactively (skips the popup) and
  `PIANOPLAYER_TEMPO_POPUP=0` disables the popup; when the piece ends the
  score clock freezes at the total and the window stays open;
- **HUD** (top left) shows elapsed/total time, current measure and the
  effective tempo in BPM, plus a progress strip along the fall line;
- **Space pauses** the player: the frame freezes while `q`/Esc still quit;
  pausing damps the synth (`all_notes_off` + `drop_scheduled`) and rewinds
  each hand's schedule so the pending notes are re-sent on resume (no early
  sound, no silent key presses); the audio ring keeps being pumped while
  paused, the HUD shows `PAUSED` live, and holding Space does not
  auto-repeat (release-tracked debounce). On resume the score replays the
  last 0.9 s (`RESUME_LEAD` pre-roll covering the ring cushion + pipeline,
  so no note sounds late) while ringing notes decay naturally;
- **sound stays glued to the picture**: the synth clock is monotonic, the
  output latency is re-measured every frame, and already-scheduled notes are
  shifted when the latency grows mid-playback. If your machine still shows a
  constant offset, increase `--av-offset-ms` (or set the `PIANOPLAYER_AV_MS`
  env var to the same number of ms) when the image leads the sound;
- the built-in synthesiser renders a full polyphonic piano with no samples,
  panning each hand slightly to its side of the stage.


## How the algorithm works:
The algorithm minimizes the fingers speed needed to play a sequence of notes or chords by searching
through feasible combinations of fingerings. The whole passage is solved at
once with a Viterbi search over finger-pair states (25 states), scored with
the finger-speed cost below plus penalties for rule violations (same finger
on different keys, crossings, thumb-to-black, overstretch, re-using a still
held finger), followed by a greedy repair pass for long held notes
(linear-time `_repair_held`).

Span limits (`MAX_SPAN`/`COMFORT_SPAN`, incl. new thumb rows) are compared as
raw cm vs `limit * hf`, so hand size matters. Notes carry `INote.midi` plus an
optional `INote.hint` (pinned finger, `PIN_COST` unless honoured). Chord
groups use a layered chord DP (Amendment A node cost) for whole-chord
consistency (`legacy_chord_postpass` flag keeps the legacy candidate pass).
Broken-chord valleys/peaks reward the outer finger (leap-scaled, capped;
arrival drops >= 5 semitones also qualify); slow same-finger reuse on different
keys costs base + distance rate (base 6.0 + 0.4/cm). `PATTERN_343_PENALTY` stays OFF.
Figure-lite soft suggestions (no new solver: priced once per single note in
the DP node/arrival costs, far below HARD so physical limits always win):
chromatic runs (white -> 1, black -> 3), one-direction scale runs >= 6 notes
(standard pattern from the run start), wide arpeggios (cyclic 1-2-3/1-2-3-4,
peak -> 5), broken octaves (low -> thumb side, high -> 5, 4 on black).
Weak-finger unary (4: +0.24, 5: +0.12) breaks ties toward strong fingers;
thumb-black static cost lowered to 0.25 (pinky 0.35); one finger may cover two
neighbour keys in a chord (thumb 7.0 / pinky 8.0 soft instead of HARD) and an
octave-plus grip on 1-4 with an inner note down pays 4.0 (preferring 1-5).
Phase-2 verdicts: E1 prior validated/OFF, E2 (`V_CAP = 40` + `SEC_SMOOTH = 0.2`
candidates) unvalidated without PIG, E3/E4 OFF.

PIG evaluation loads ground truth via `load_pig` (`pianoplayer/pigloader.py`,
chords included, note-weighted average); `PIG2Stream` duplicate-chord
behaviour is unchanged (CLI/GUI import path only).

One possible advantage of this algorithm over similar ones is that it is completely *dynamic*,
which means that it
takes into account the physical position and speed of fingers while moving on the keyboard
and the duration of each played note.
It is *not* based on a static look-up table of likely or unlikely combinations of fingerings.

Fingering a piano score can vary a lot from individual to individual, therefore there is not such
a thing as a "best" choice for fingering.
This algorithm is meant to suggest a fingering combination which is "optimal" in the sense that it
minimizes the effort of the hand avoiding unnecessary movements.

## Parameters you can change:
- Your hand size (from 'XXS' to 'XXL') which sets the relaxed distance between thumb and pinkie.
- The beam number associated to the right hand is by default nr.0 (nr.1 for left hand).
You can change it with `-rbeam` and `-lbeam` command line options.
- Depth of combinatorial search, from 3 up to 9 notes ahead of the currently playing note. By
default the algorithm selects this number automatically based on the duration of the notes to be played.
This option only applies to the legacy sliding-window search mode; the default solver looks at the whole passage at once.
- The look and the motion of the animated hands (anticipation height, strike
  and linger timings, wrist turn, tendon link, idle fan, scale-run glide,
  tremolo rock) live in the "motion step 1" block of `pianoplayer/handspec.py`
  and can be retuned there; they never affect the fingering, and they are
  excluded from the mesh cache hash, so tuning them does not rebuild anything.
- The hand skin: `--hand-style cartoon|gloves|robot|skeleton` (CLI, 3D
  player only) or `PIANOPLAYER_HAND_STYLE`; default `cartoon`. The palettes,
  material coefficients and vertex-colour bands live in the "Skins" block of
  `pianoplayer/handspec.py`. A style only changes how the hand is painted --
  never its geometry, IK or reach -- and it is excluded from the mesh cache
  hash (only the texture filename carries it).
- The hand's anatomy and flexibility live in `pianoplayer/handspec.py` too:
  the bone lengths, per-finger segment ratios, knuckle row and wrist width in
  the "Anatomy" block (geometry -- inside `geom_hash()`, so a change
  rebuilds the cached mesh), and the per-finger joint limits and
  hover/idle heights in the "Flexibility / behaviour" block (`FLEX_BEND`,
  `FLEX_COUPLING`, `HOVER_GAP`, `HOVER_ADD`, `IDLE_LIFT`, `IDLE_PULL_Y`,
  `IDLE_PULL_Z`; visual-only and excluded from the hash). Defaults are the
  shipped values; `BEHAVIOR_PORT` documents which 2D-project slider each knob
  came from.

## Limitations
- Awkward crossings (e.g. the 3rd finger over the 4th) are allowed but pay a
  soft cost scaling with interval and inverse time; chords are always pinned
  to monotonic fingerings.
- Hands are always assumed independent from each other.
- Small notes/ornaments are ignored.


