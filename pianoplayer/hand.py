#-------------------------------------------------------------------------------
# Name:         PianoPlayer
# Purpose:      Find optimal fingering for piano scores
# Author:       Marco Musy
#-------------------------------------------------------------------------------
import copy
import itertools
import warnings

import numpy as np
from music21.articulations import Fingering
import pianoplayer.utils as utils
import pianoplayer.handspec as _HS


# Phase B (Viterbi fingering) constants. All at the top so they are easy to tune.
HARD = 200.0            # cost added per violated skip/held rule (hard-soft: a
                        # violated rule never eliminates the path, fixing F4)
WEAK_FINGER_PENALTY = 3.0   # B3: light penalty, see pair_cost()
WEAK_FINGER_MAX_DT = 0.15   # B3/F1: seconds; fires when the inter-onset gap
                            # in seconds is below this.
WEAK_FINGER_MIN_JUMP = 3.0  # B3: ... and the finger itself travels more than
                            # this (cm).
HELD_EPS = 1e-3             # B2: overlap slack in quarter notes.
SEC_SMOOTH = 0.1            # F1: smoothing term added to dt, in seconds.
REPEAT_PENALTY = 6.0        # F2: same finger + same key + dt < REPEAT_MAX_DT
REPEAT_MAX_DT = 0.25        #   adds REPEAT_PENALTY*(MAX_DT-dt)/MAX_DT (s).
SAME_FINGER_BASE = 4.0      # F3: soft cost of reusing one finger on different
SAME_FINGER_DX = 0.75       #   keys: BASE + DX*|dx_cm| when dt >= MAX_DT.
                            #   Round 2: reuse across a leap pays
                            #   leap-proportionally more (was 0.15; 0.5 left
                            #   a slow Bb3-G3 1-1 reuse in Tim_Em b.4).
SAME_FINGER_MAX_DT = 0.25   # F3: faster reuse than this (s) is still HARD.
FALLBACK_BPM = 120.0        # F1: quarters -> seconds when t_sec is missing
                            # (hand-built INotes that never went through
                            # reader()).
PIN_COST = 1000.0           # Phase 4 (§6): added per hinted note (hint in
                            # 1..5) whose finger differs from the hint
                            # (= 5 HARD: a pin loses only to 5+ stacked
                            # violations, never to a mere preference).
VALLEY_BIAS = 2.0           # Valley outer-finger bias: reward (negative
                            # cost) for the outer finger on a strict local
                            # pitch extremum of a broken-chord line (LH:
                            # minimum -> 5, maximum -> 1; mirrored for RH).
                            # 0.0 = off.
VALLEY_MIN_DEPTH = 2        # semitones each side required for the bias.
VALLEY_LEAP_K = 2.0         # Round 2: extra reward per semitone of arrival
                            # leap (drop into a valley / rise into a peak)
                            # beyond VALLEY_MIN_DEPTH -- a bigger leap means
                            # the hand repositions, so the outer finger
                            # anchors with less velocity-model resistance.
VALLEY_BONUS_CAP = 15.0     # Round 2: max total valley reward (magnitude).

# Phase 2 experiments (each OFF until validated per PIANOPLAYER_FINGERING_PLAN
# §4.0; defaults reproduce current behaviour exactly).
START_PREF_RIGHT = {1: -10.0, 5: -1.0}   # E1: first-note prior, right hand
START_PREF_LEFT = {5: -5.0, 1: -1.0}     # E1: first-note prior, left hand
V_CAP = float("inf")        # E2: cap on the velocity term reach_err/dt
                            # (cm/s) before the weight division; inf = off.
UNDER_REACH_CM = 1e9        # E3: hf=1 distance a thumb can tuck under
                            # without extra effort (1e9 = off).
UNDER_DIST_K = 0.0          # E3: extra cost factor beyond that distance
                            # (0.0 = off).
UNDER_LEAP_FROM5 = 1.0      # E3: multiplier of UNDER[5] when the thumb
                            # lands >= 4 cm away (1.0 = off).
PATTERN_343_PENALTY = 0.0   # E4: extra cost for 3-4-3 / 4-3-4 triples in
                            # solve_viterbi (0.0 = off).


def _dt_sec(na, nb):
    """Inter-onset gap in seconds (F1), without the smoothing term."""
    ta = getattr(na, "t_sec", None)
    tb = getattr(nb, "t_sec", None)
    if ta is None or tb is None:
        return abs(nb.time - na.time) * 60.0 / FALLBACK_BPM
    return abs(float(tb) - float(ta))


def _hint_of(note):
    """Pinned finger for a note (Phase 4, §6): 1..5, else 0 (= open).

    `getattr` keeps hand-built notes without a `hint` attribute (and
    foreign mocks) on the hint-free path, which is bit-identical."""
    try:
        h = int(getattr(note, "hint", 0) or 0)
    except (ValueError, TypeError):
        return 0
    return h if 1 <= h <= 5 else 0


def _valley_bonus(nprev, n, nnext, finger, side):
    """Outer-finger reward for a broken-chord valley/peak.

    Note `n` (with both neighbours present) that is a strict local pitch
    minimum gets a reward on the outer finger (LH 5 / RH 1); a strict
    local maximum gets it on the other outer finger (LH 1 / RH 5).
    Depth = min(|midi - neighbour|) each side must reach VALLEY_MIN_DEPTH
    semitones, else 0.0. Round 2: the reward grows with the arrival leap
    (drop into a valley, rise into a peak -- same constants, symmetric):
    VALLEY_BIAS + VALLEY_LEAP_K * (leap - VALLEY_MIN_DEPTH), capped at
    VALLEY_BONUS_CAP magnitude. Chord members never qualify (simultaneous
    notes have no melodic direction); missing midi (hand-built INotes)
    also gives 0.0. `side` is "left"/"right" (mirroring-safe: midi-based).

    Shared by the layered-DP node costs and the legacy Viterbi path so
    both solvers agree; single notes (no both neighbours) get 0.0.
    """
    if nprev is None or nnext is None:
        return 0.0
    if not VALLEY_BIAS:
        return 0.0
    if finger not in (1, 2, 3, 4, 5):
        return 0.0
    if getattr(nprev, "isChord", False) or getattr(n, "isChord", False) \
            or getattr(nnext, "isChord", False):
        return 0.0
    mp = getattr(nprev, "midi", None)
    m = getattr(n, "midi", None)
    mn = getattr(nnext, "midi", None)
    if mp is None or m is None or mn is None:
        return 0.0
    try:
        mp, m, mn = int(mp), int(m), int(mn)
    except (TypeError, ValueError):
        return 0.0
    if m < mp and m < mn:
        if min(mp - m, mn - m) < VALLEY_MIN_DEPTH:
            return 0.0
        outer = 5 if side == "left" else 1
        if finger != outer:
            return 0.0
        leap = mp - m  # drop into the valley from the previous note
    elif m > mp and m > mn:
        if min(m - mp, m - mn) < VALLEY_MIN_DEPTH:
            return 0.0
        outer = 1 if side == "left" else 5
        if finger != outer:
            return 0.0
        leap = m - mp  # rise into the peak from the previous note
    else:
        return 0.0
    bonus = VALLEY_BIAS + VALLEY_LEAP_K * max(0, leap - VALLEY_MIN_DEPTH)
    return -min(bonus, VALLEY_BONUS_CAP)


def _set_fingering(n21, fng, replace=True):
    """Attach a Fingering articulation.

    `replace=False` keeps the fingering already there: a chord gets
    one number per pitch, and only its first note (`chordnr == 0`)
    purges the previous run's numbers -- so re-running generate()
    neither drops nor doubles the annotation."""
    if replace:
        for a in [a for a in n21.articulations if isinstance(a, Fingering)]:
            n21.articulations.remove(a)
    n21.articulations.append(fng)


#####################################################
class Hand:
    def __init__(self, side="right", size='M'):

        self.LR        = side
        # fingers pos at rest first is dummy, (cm), asymmetry helps with scales
        self.frest     = [None,  -7.0,-2.8, 0.0, 2.8, 5.6]
        self.weights   = [None,   1.1, 1.0, 1.1, 0.9, 0.8] # finger relative strength
        self.bfactor   = [None,   0.3, 1.0, 1.1, 0.8, 0.7] # hit of black key bias
        self.noteseq   = []
        self.fingerseq = []
        self.fingerseq_index = None  # list (maybe empty) once generate() ran
        self.depth     = 9
        self.autodepth = True
        self.verbose   = True
        self.lyrics    = False  # show fingering numbers as lyrics in musescore
        self.size      = size
        # Phase B: False = Viterbi solve of the whole range (default);
        # True = legacy 9-note sliding-window search (optimize_seq).
        # The -d/--depth flag only has an effect in legacy mode.
        self.legacy_window = False
        # Phase 3 (§5): False = layered chord-as-note DP (default);
        # True = legacy note-by-note Viterbi + _apply_chord_candidates
        # post-pass (kept for one release for comparison).
        self.legacy_chord_postpass = False

        self.hf = utils.handSizeFactor(size)
        for i in (1,2,3,4,5):
            if self.frest[i]: self.frest[i] *= self.hf
        self.cfps = list(self.frest) # hold current finger positions


    #####################################################
    def set_fingers_positions(self, fings, notes, i):

        fi = fings[i]
        ni = notes[i]
        ifx = self.frest[fi]
        if ifx is not None:
            for j in (1,2,3,4,5):
                jfx = self.frest[j]
                self.cfps[j] = (jfx-ifx) + ni.x


    #####################################################
    def ave_velocity(self, fingering, notes):
        ###calculate v for playing for notes in a given fingering combination

        self.set_fingers_positions(fingering, notes, 0)  #init fingers position

        vmean = 0.
        for i in range(1, self.depth):
            na = notes[i-1]
            nb = notes[i]
            fb = fingering[i]
            dx = abs(nb.x - self.cfps[fb])   # space travelled by finger fb
            dt = abs(nb.time - na.time) +0.1 # available time +smoothing term 0.1s
            v  = dx/dt                       # velocity
            if nb.isBlack:                   # penalty (by increasing speed)
                v /= self.weights[fb] * self.bfactor[fb]
            else:
                v /= self.weights[fb]
            vmean += v
            self.set_fingers_positions(fingering, notes, i) #update all fingers

        return vmean / (self.depth-1)


    #####################################################
    def _span_cost(self, fa, fb, span_cm):
        """F4a: quadratic cost up to COMFORT_SPAN, then steeper up to MAX_SPAN,
        HARD beyond.  span_cm is the RAW keyboard distance in cm (never
        multiplied by hf); the limits are scaled by hf."""
        if fa == fb:
            return HARD
        # Order fingers so fa < fb for lookup (MAX_SPAN is symmetric)
        a, b = (fa, fb) if fa < fb else (fb, fa)
        max_span = _HS.MAX_SPAN[a][b] * self.hf
        comfort = _HS.COMFORT_SPAN[a][b] * self.hf
        if span_cm <= comfort:
            return 0.0
        if span_cm >= max_span:
            return HARD
        # Quadratic ramp from comfort to max_span
        t = (span_cm - comfort) / (max_span - comfort)
        return HARD * 0.5 * t * t  # up to 0.5*HARD at max_span

    def _static_chord_cost(self, fingers, notes):
        """Static cost for a chord fingering (no velocity, just spans + preferences)."""
        cost = 0.0
        for i, (fa, na) in enumerate(zip(fingers, notes)):
            for j, (fb, nb) in enumerate(zip(fingers, notes)):
                if i >= j:
                    continue
                span_cm = abs(nb.x - na.x)
                cost += self._span_cost(fa, fb, span_cm)
            # F6: static black/white preference
            if na.isBlack:
                cost += _HS.BLACK_COST.get(fa, 0.0)
            else:
                cost += _HS.WHITE_COST.get(fa, 0.0)
        return cost

    def _chord_finger_candidates(self, chord_notes, limit=6):
        """F4b: Generate monotonic finger sets for a chord (≤5 notes).

        Returns top-`limit` candidates by static cost (`limit=None`
        returns all, used by the Phase 3 layered DP). Never returns
        empty: if the span filter rejects every monotonic combo, retry
        unfiltered (monotonicity is guaranteed by construction either
        way).
        """
        k = len(chord_notes)
        if k > 5:
            warnings.warn(f"Chord with {k} notes > 5: splitting", stacklevel=2)
            return None  # Signal to split
        fingers = (1, 2, 3, 4, 5)
        # Generate all monotonic combinations C(5, k)
        combos = []
        for combo in itertools.combinations(fingers, k):
            # Monotonic order: right hand ascending (low pitch -> low finger),
            # left hand descending (low pitch -> high finger)
            if self.LR == "right":
                combos.append(list(combo))
            else:
                combos.append(list(reversed(combo)))
        # Pass 1: filter by span (adjacent pairs must fit MAX_SPAN)
        cands = [c for c in combos if self._chord_span_ok(c, chord_notes)]
        if not cands:
            cands = list(combos)  # pass 2: span-relaxed, still monotonic
        scored = [(self._static_chord_cost(c, chord_notes), c) for c in cands]
        scored.sort(key=lambda x: x[0])
        ranked = [f for _, f in scored]
        if limit is None:
            return ranked
        return ranked[:limit]

    def _chord_span_ok(self, fings, chord_notes):
        """True when every adjacent pair of the chord fits MAX_SPAN."""
        for i in range(len(fings) - 1):
            span_cm = abs(chord_notes[i + 1].x - chord_notes[i].x)
            if self._span_cost(fings[i], fings[i + 1], span_cm) >= HARD:
                return False
        return True

    def _apply_chord_candidates(self, notes, fingers):
        """F4b post-pass: pin every chord group to one whole candidate.

        Phase 3 (§5): retired in the default layered-DP path -- the DP
        already assigns every chord layer one whole monotonic candidate,
        so there is nothing left to pin and this returns the input
        unchanged (hence the audit `postpass_changed` is 0 there).
        With `legacy_chord_postpass = True` the old behaviour below is
        kept verbatim for one release:

        The order-2 Viterbi state cannot enforce whole-chord consistency
        (HARD is a soft cost, so paths can switch candidates mid-chord);
        this pass picks, per group, the candidate minimizing static cost
        plus transition costs to the fixed neighbours. Monotonicity and
        distinct fingers hold by construction.
        """
        if not self.legacy_chord_postpass:
            return list(fingers)
        f = list(fingers)
        n = len(notes)
        chord_cands = getattr(self, "_chord_candidates", {})
        for start, end, chord_notes in self._group_chords(notes):
            cands = chord_cands.get(start)
            if not cands:
                continue
            k = end - start
            # Phase 4 (§6): keep only hint-consistent candidates
            # (fallback: all). Guarded by any-hints, so hint-free
            # groups behave exactly as before.
            hints = [_hint_of(notes[start + i]) for i in range(k)]
            if any(hints):
                filt = [c for c in cands
                        if len(c) == k and all(
                            h == 0 or c[i] == h
                            for i, h in enumerate(hints))]
                if filt:
                    cands = filt
            best, bestc = None, None
            for c in cands:
                if len(c) != k:
                    continue
                tot = self._static_chord_cost(c, chord_notes)
                if start > 0:
                    tot += self.pair_cost(notes[start - 1], f[start - 1],
                                          notes[start], c[0])
                for i in range(k - 1):
                    tot += self.pair_cost(notes[start + i], c[i],
                                          notes[start + i + 1], c[i + 1])
                if end < n:
                    tot += self.pair_cost(notes[end - 1], c[-1],
                                          notes[end], f[end])
                if best is None or tot < bestc:
                    best, bestc = c, tot
            if best is not None:
                for i, fb in enumerate(best):
                    f[start + i] = fb
        return f

    def _group_chords(self, notes):
        """Group consecutive notes with same chordID into chord groups.
        Returns list of (start_idx, end_idx, chord_notes) for each chord."""
        groups = []
        i = 0
        n = len(notes)
        while i < n:
            if notes[i].isChord:
                cid = notes[i].chordID
                start = i
                chord_notes = []
                while i < n and notes[i].isChord and notes[i].chordID == cid:
                    chord_notes.append(notes[i])
                    i += 1
                groups.append((start, i, chord_notes))
            else:
                i += 1
        return groups

    def _precompute_chord_candidates(self, notes):
        """Precompute finger candidates for each chord group."""
        chord_groups = self._group_chords(notes)
        self._chord_candidates = {}  # start_idx -> list of finger lists
        self._chord_spans = {}       # start_idx -> (start, end)
        for start, end, chord_notes in chord_groups:
            cands = self._chord_finger_candidates(chord_notes)
            if cands is None:
                # >5 notes: split into overlapping sub-chords (simplified: warn and use first 5)
                # For now, just use first 5 notes
                cands = self._chord_finger_candidates(chord_notes[:5])
            self._chord_candidates[start] = cands
            self._chord_spans[start] = (start, end)

    def _crossing_cost(self, fa, fb, xba, dt_gap, na, nb):
        """F5: crossing cost replacing unconditional HARD.
        fa->fb is a crossing when (fb-fa)*xba < 0 (fingers cross over x).
        For left hand, x is mirrored (negated), so we use -xba for crossing check.
        Returns added cost (0 if not a crossing)."""
        if fa == fb:
            return 0.0
        # For left hand, x is mirrored (higher pitch = more negative x)
        # So crossing condition is inverted: use -xba for left hand
        eff_xba = -xba if self.LR == "left" else xba
        if (fb - fa) * eff_xba >= 0:
            return 0.0  # not a crossing
        
        # Thumb-under: fb == 1 (thumb passes under fa)
        if fb == 1 and fa > 1:
            base = _HS.UNDER.get(fa, 1.0)
            # E3 (OFF defaults): distance-aware thumb-under. Longer tucks
            # cost extra beyond UNDER_REACH_CM, and a 5 -> 1 leap over a
            # third or more is the standard arpeggio hand rotation.
            base *= 1.0 + UNDER_DIST_K * max(
                0.0, abs(xba) / (UNDER_REACH_CM * self.hf) - 1.0)
            if fa == 5 and abs(xba) >= 4.0:
                base *= UNDER_LEAP_FROM5
            # Departing from a black key onto white (the classic Bb->C tuck):
            # the hand is already raised, so the tuck rides the hand position
            # (halved). White->white tucks pay full UNDER so the after-3 vs
            # after-4 margin survives downstream strong-finger benefits
            # (G-major gold); tucking ONTO black pays full base plus the
            # bfactor velocity penalty (legacy no-thumb-on-black rule).
            if na.isBlack and not nb.isBlack:
                base *= 0.5
        # Thumb-over: fa == 1 (fa is thumb, fb passes over) - rare
        elif fa == 1 and fb > 1:
            base = _HS.UNDER.get(fb, 2.0)
            if not na.isBlack:
                base *= 0.2
        else:
            # Non-thumb crossing: higher finger over lower
            a, b = (fa, fb) if fa < fb else (fb, fa)
            base = _HS.CROSS_BASE.get((a, b), 1.5)
        # Scale by interval (semitones) and inverse time
        if na.midi is not None and nb.midi is not None:
            semitones = abs(int(nb.midi) - int(na.midi))
        else:
            semitones = int(round(abs(xba) / 1.0))      # hand-built INotes without midi
        interval_mult = max(1.0, semitones / 2.2)       # was /3.0 on a cm approximation
        time_mult = max(1.0, 0.2 / max(dt_gap, 1e-3))
        return base * interval_mult * time_mult

    def _skip_penalty(self, fa, fb, na, nb):
        """Hard-soft version of the old optimize_seq.skip() rules.

        Returns HARD when the (fa on na -> fb on nb) transition breaks a
        rule, 0.0 otherwise. Unlike skip(), a violation never eliminates
        the combination, so a path always exists (fixes F4).

        The rules run on EVERY pair (fixes F2): two single notes keep the
        old rules; two notes of the same chord keep the old chord rules;
        any other mix (chord<->note, chord<->chord) gets the "same finger,
        different pitch" rule plus the single-note crossing rules on x.
        """
        xba = nb.x - na.x  # physical distance btw the second to first note, in cm
        dt_gap = _dt_sec(na, nb)

        # Same finger, different key: F3 soft cost (or HARD if held/fast)
        if fa == fb and abs(xba) > 1e-9:
            if na.time_real + na.dur_real > nb.time_real + HELD_EPS:
                return HARD  # finger still holding na: cannot replay (B2)
            # F3 boundary is inclusive (eighth notes at 120 bpm sit exactly
            # at 0.25 s); the +1e-6 epsilon keeps the boundary stable against
            # float rounding in beat_to_seconds (a computed 0.2500000001 s
            # must stay HARD, a computed 0.2499999999 s likewise).
            if dt_gap <= SAME_FINGER_MAX_DT + 1e-6:
                return HARD  # too fast to lift and re-place the finger
            return SAME_FINGER_BASE + SAME_FINGER_DX * abs(xba)

        if na.isChord and nb.isChord and na.chordID == nb.chordID:
            # na and nb are notes in the same chord: check finger order + span
            if self.LR == 'left' and fa < fb:
                return HARD
            if self.LR == 'right' and fa > fb:
                return HARD
            # F4a: span cost replaces the old axba threshold chain
            span_cm = abs(xba)
            return self._span_cost(fa, fb, span_cm)

        # Single notes and mixed pairs: crossing rules on x (F5)
        cross = self._crossing_cost(fa, fb, xba, dt_gap, na, nb)
        if cross >= HARD:
            return HARD
        if cross > 0:
            return cross

        # Thumb-on-black crossing rule (legacy, keep as soft)
        # For left hand, xba sign is inverted
        eff_xba = -xba if self.LR == "left" else xba
        if fa > 1 and fb == 1 and nb.isBlack and eff_xba > 0:
            return 0.5  # soft penalty instead of HARD
        if fa == 1 and fb > 1 and na.isBlack and eff_xba < 0 and na.duration < 2:
            return 0.5

        return 0.0


    #####################################################
    def pair_cost(self, na, fa, nb, fb):
        """Cost of playing nb with finger fb right after na with finger fa.

        The velocity term reproduces ave_velocity() exactly: reach_err is
        the distance finger fb itself travels, dt the inter-onset gap plus
        the 0.1 smoothing term, weights/bfactor the old finger strength and
        black-key penalties. On top come the hard-soft skip rules (B1), the
        held-finger constraint (B2) and the weak-finger penalty (B3).
        F6: static cost per (finger, black/white) added, not multiplied by velocity.
        
        Special handling for thumb-under on white keys (scale pattern):
        the hand translates, so the thumb doesn't travel the full rest-position distance.
        """
        if fa not in (1,2,3,4,5) or fb not in (1,2,3,4,5):
            return HARD  # e.g. a legacy finger 0: always losable, never fatal
        same_key = abs(nb.x - na.x) < 1e-9
        xba = nb.x - na.x  # physical distance btw the second to first note, in cm
        
        # F5/F6: Special case - thumb-under on white keys (scale pattern)
        # The hand translates forward, so reach_err is just the key distance
        # For right hand: thumb-under is fa in {3,4}, fb=1, ascending white-to-white
        # For left hand (mirrored): thumb-under is fa=2 (index), fb=1, ascending white-to-white
        # Also: left hand after thumb (1) goes to middle (3) for the next white step
        eff_xba = -xba if self.LR == "left" else xba
        # Hand-shift credit applies when arriving on white (departure may be
        # black: the classic Bb->C tuck; arrival on black never gets credit).
        ascending_white = (eff_xba > 0 and not nb.isBlack)
        is_thumb_under_right = (self.LR == "right" and fa in (3, 4) and fb == 1 and ascending_white)
        is_thumb_under_left = (self.LR == "left" and fa == 2 and fb == 1 and ascending_white)
        # Left hand: after thumb (1), middle (3) takes the next white key step (hand shift)
        is_left_thumb_to_middle = (self.LR == "left" and fa == 1 and fb == 3 and ascending_white)
        # Whole-step tucks (Bb->C = 3.54 cm) are standard: cap at 4.0 cm so
        # fourths and wider still pay full translation, semitones/whole
        # steps ride the hand shift.
        is_step = (is_thumb_under_right or is_thumb_under_left or is_left_thumb_to_middle) and abs(nb.x - na.x) < 4.0
        
        if same_key:
            reach_err = 0.0
        elif is_step:
            # Thumb-under on white keys: hand shifts so thumb is already in position
            # Use near-zero reach_err to strongly prefer this pattern over adjacent fingers
            reach_err = 0.02  # cm - hand translation handles it
        else:
            reach_err = abs(nb.x - (na.x + self.frest[fb] - self.frest[fa]))
            
        dt_gap = _dt_sec(na, nb)
        dt = dt_gap + SEC_SMOOTH  # available time (s) + smoothing term
        v  = min(reach_err/dt, V_CAP)  # E2: V_CAP bounds the velocity term
                                       # (inf = unbounded, as before)
        if nb.isBlack:                   # penalty (by increasing speed)
            v /= self.weights[fb] * self.bfactor[fb]
        else:
            v /= self.weights[fb]
        cost = v + self._skip_penalty(fa, fb, na, nb)
        # F6: static black/white preference per finger (not multiplied by velocity)
        if nb.isBlack:
            cost += _HS.BLACK_COST.get(fb, 0.0)
        else:
            cost += _HS.WHITE_COST.get(fb, 0.0)
        # B2: a finger still holding na when nb starts must not replay a
        # different key at the same time.
        if na.time_real + na.dur_real > nb.time_real + HELD_EPS:
            if fa==fb and abs(nb.x - na.x)>1e-9:
                cost += HARD
        # F2: hammering one key with one finger gets tiring fast: penalise
        # same-finger repeats below REPEAT_MAX_DT, linearly up to
        # REPEAT_PENALTY for a zero gap. Slow repeats (dt >= 0.25 s) are
        # free, so the solver may keep the finger there. Strict `<` (no
        # epsilon): a repeat exactly at the boundary is free by design,
        # unlike the F3 reuse ban above which is inclusive by design.
        if fa==fb and same_key and dt_gap < REPEAT_MAX_DT:
            cost += REPEAT_PENALTY * (REPEAT_MAX_DT - dt_gap) / REPEAT_MAX_DT
        # B3: weak fingers (4, 5) must not jump far on short notice.
        # (No extra thumb-on-black penalty: bfactor already covers it.)
        if fb in (4,5) and dt_gap < WEAK_FINGER_MAX_DT \
                and reach_err > WEAK_FINGER_MIN_JUMP:
            cost += WEAK_FINGER_PENALTY
        return cost


    #####################################################
    def total_cost(self, notes, fingers):
        """Sum of pair_cost() over the consecutive pairs of a solution.

        Used to compare the Viterbi output against the legacy-window
        output with one and the same cost function (B5).
        """
        if len(notes) < 2 or len(fingers) < 2:
            return 0.0
        return float(sum(self.pair_cost(na, fa, nb, fb) for na, fa, nb, fb
                         in zip(notes, fingers, notes[1:], fingers[1:])))


    #####################################################
    def solve_viterbi(self, notes):
        """Optimal finger sequence (1..5) for the whole note list.

        Default (Phase 3, §5) is the layered chord-as-note DP
        (`solve_layered`); with `legacy_chord_postpass = True` the old
        note-by-note Viterbi below runs instead (caller must still apply
        `_apply_chord_candidates`, as `generate()` does).
        """
        if self.legacy_chord_postpass:
            return self._solve_viterbi_legacy(notes)
        return self.solve_layered(notes)

    def _build_layers(self, notes):
        """Split notes into layers (§5): one single note, or one whole
        chord group (consecutive notes sharing a chordID). Returns a
        list of (start, end) index ranges."""
        layers = []
        i, n = 0, len(notes)
        while i < n:
            if notes[i].isChord:
                cid = notes[i].chordID
                j = i
                while j < n and notes[j].isChord and notes[j].chordID == cid:
                    j += 1
                layers.append((i, j))
                i = j
            else:
                layers.append((i, i + 1))
                i += 1
        return layers

    def _layer_candidates(self, layer_notes):
        """Candidate finger tuples for one layer (§5).

        Single note -> (1,)..(5,). Chord (<=5) -> ALL monotonic sets
        from `_chord_finger_candidates` WITHOUT the top-6 pruning (at
        most C(5,k) <= 10 per layer), span-filtered with unfiltered
        fallback, sorted by static cost. Chords with >5 notes assign
        the lowest 5 here; each extra note takes the finger of the
        nearest assigned pitch at decode (documented compromise).
        """
        if len(layer_notes) == 1 and not layer_notes[0].isChord:
            return [[f] for f in (1, 2, 3, 4, 5)]
        members = layer_notes
        if len(members) > 5:
            warnings.warn(
                f"Chord with {len(members)} notes > 5: lowest 5 assigned,"
                " extras take the nearest assigned finger", stacklevel=2)
            members = members[:5]
        return self._chord_finger_candidates(members, limit=None)

    def _hint_consistent_candidates(self, layer_notes, cands):
        """Phase 4 (§6): keep only candidates matching every hinted
        chord member; fall back to all candidates when none matches.
        Hint-free layers return `cands` unchanged (same object: the
        hint-free path is bit-identical). Members past the first 5
        (>5-note chords) ride the nearest assigned finger at decode
        (see `_expand_layer_fingers`), so their hints constrain the
        assigned finger they ride on (best-effort)."""
        hints = [_hint_of(m) for m in layer_notes]
        if not any(hints):
            return cands
        fit = layer_notes[:5]
        extra_ride = [min(range(5),
                          key=lambda i: (abs(m.x - fit[i].x), i))
                      for m in layer_notes[5:]]
        keep = [c for c in cands
                if all(h == 0 or c[j if j < 5 else extra_ride[j - 5]] == h
                       for j, h in enumerate(hints))]
        return keep or cands

    def _layer_held_extra(self, Anotes, candA, Bnotes, candB):
        """HARD per held-finger violation over ALL note pairs across two
        adjacent layers, EXCEPT (last of A, first of B), which
        `pair_cost` already covers. Same condition as the B2 rule in
        `pair_cost` (same finger, different key, A-note still sounding
        at B-note onset)."""
        extra = 0.0
        la = len(Anotes) - 1
        for p, (na, fa) in enumerate(zip(Anotes, candA)):
            if fa not in (1, 2, 3, 4, 5):
                continue
            for q, (nb, fb) in enumerate(zip(Bnotes, candB)):
                if p == la and q == 0:
                    continue  # inside pair_cost already
                if fb not in (1, 2, 3, 4, 5):
                    continue
                if fa == fb and abs(nb.x - na.x) > 1e-9 \
                        and na.time_real + na.dur_real > nb.time_real + HELD_EPS:
                    extra += HARD
        return extra

    def _expand_layer_fingers(self, layer_notes, cand):
        """Per-note fingers for one solved layer, expanding >5-note
        chords (lowest 5 assigned; each extra note takes the finger of
        the nearest assigned pitch by keyboard distance, lowest index
        on ties)."""
        if len(layer_notes) > 5:
            fit = layer_notes[:5]
            out = list(cand)
            for m in layer_notes[5:]:
                bi = min(range(5),
                         key=lambda i: (abs(m.x - fit[i].x), i))
                out.append(cand[bi])
            return out
        return list(cand)

    def solve_layered(self, notes):
        """Layered chord-as-note DP (§5, Phase 3).

        Each layer is one note or one whole chord. Layer-to-layer cost
        is `pair_cost()` between the LAST note of layer i and the FIRST
        note of layer i+1 (position/velocity/crossing, incl. all Phase 2
        OFF-default constants via the shared code path) plus the
        held-finger rule over ALL other note pairs across the two
        layers, plus the chord-layer interior objective inside each
        chord layer (Amendment A: `_static_chord_cost` plus internal
        consecutive `pair_cost`, the validated post-pass interior).
        The order-2 state mirrors `solve_viterbi` (so
        `PATTERN_343_PENALTY` applies to three consecutive single-note
        layers exactly as there); a held note spanning two layers stays
        in the `_repair_held` post-pass, not in the DP state.
        Deterministic: ties keep the first minimum.
        """
        n = len(notes)
        if n == 0:
            return []
        if n == 1:
            h0 = _hint_of(notes[0])  # Phase 4 (§6): a pinned lone note
            return [h0 or 3]         # takes its hint, else finger 3
        layers = self._build_layers(notes)
        L = len(layers)
        layer_notes = [notes[s:e] for s, e in layers]
        cands = [self._layer_candidates(ln) for ln in layer_notes]
        is_single = [len(ln) == 1 and not ln[0].isChord
                     for ln in layer_notes]
        # Phase 4 (§6): chord layers keep only hint-consistent
        # candidates (fallback: all). Single-note layers price the
        # mismatch in their node cost below (once per note, first
        # layer included via the dp init).
        cands = [cc if single else self._hint_consistent_candidates(ln, cc)
                 for ln, cc, single in zip(layer_notes, cands, is_single)]
        # Node costs: chord-layer interior objective (Amendment A) =
        # _static_chord_cost + internal consecutive pair_cost between
        # adjacent chord members -- exactly the validated
        # _apply_chord_candidates interior (static + internal pair
        # costs). Neighbour transitions stay inter-layer pair_cost
        # (last -> first), so nothing is double-counted.
        node = []
        for j, (ln, cc, single) in enumerate(zip(layer_notes, cands,
                                                 is_single)):
            if single:
                h0 = _hint_of(ln[0])  # Phase 4 (§6): PIN_COST on a
                # mismatch, once per note, plus the valley outer-finger
                # bias (needs both neighbours; first/last layer get 0).
                nprev = layer_notes[j - 1][-1] if j > 0 else None
                nnext = layer_notes[j + 1][0] if j < L - 1 else None
                node.append([(0.0 if h0 == 0 or c[0] == h0 else PIN_COST)
                             + _valley_bonus(nprev, ln[0], nnext, c[0],
                                             self.LR)
                             for c in cc])
            else:
                fit = ln[:5] if len(ln) > 5 else ln
                costs = []
                for c in cc:
                    tot = self._static_chord_cost(c, fit)
                    for j in range(len(fit) - 1):
                        tot += self.pair_cost(fit[j], c[j],
                                              fit[j + 1], c[j + 1])
                    costs.append(tot)
                node.append(costs)
        if L == 1:
            # A single chord layer: cheapest interior objective (+ start prior).
            start_pref = (START_PREF_RIGHT if self.LR == "right"
                          else START_PREF_LEFT)
            best = None
            for c, cc in enumerate(cands[0]):
                tot = node[0][c] + start_pref.get(cc[0], 0.0)
                if best is None or tot < best[1]:
                    best = (c, tot)
            return self._expand_layer_fingers(layer_notes[0],
                                              cands[0][best[0]])
        # Boundary pair_cost tables (float32, exactly like solve_viterbi).
        bounds = []
        for j in range(L - 1):
            na, nb = layer_notes[j][-1], layer_notes[j + 1][0]
            table = np.zeros((6, 6), dtype=np.float32)
            for fa in range(1, 6):
                for fb in range(1, 6):
                    table[fa, fb] = self.pair_cost(na, fa, nb, fb)
            bounds.append(table)
        start_pref = (START_PREF_RIGHT if self.LR == "right"
                      else START_PREF_LEFT)
        # dp[(a, b)]: cheapest cost ending with cands[j-1][a] on layer
        # j-1 and cands[j][b] on layer j.
        dp = {}
        for a, ca in enumerate(cands[0]):
            for b, cb in enumerate(cands[1]):
                dp[(a, b)] = (start_pref.get(ca[0], 0.0) + node[0][a]
                               + float(bounds[0][ca[-1], cb[0]])
                               + self._layer_held_extra(layer_notes[0], ca,
                                                        layer_notes[1], cb)
                               + node[1][b])
        back = {1: {(a, b): a for (a, b) in dp}}
        for j in range(2, L):
            ndp, nback = {}, {}
            for b, cb in enumerate(cands[j - 1]):
                for c, cc in enumerate(cands[j]):
                    pc = (float(bounds[j - 1][cb[-1], cc[0]])
                          + self._layer_held_extra(layer_notes[j - 1], cb,
                                                   layer_notes[j], cc)
                          + node[j][c])
                    best, best_a = None, None
                    for a, ca in enumerate(cands[j - 2]):
                        tot = dp[(a, b)] + pc
                        # E4 (OFF default): same 3-4-3 / 4-3-4 rule as
                        # solve_viterbi, for three single-note layers.
                        if PATTERN_343_PENALTY and is_single[j - 2] \
                                and is_single[j - 1] and is_single[j]:
                            triple = (ca[0], cb[0], cc[0])
                            if triple == (3, 4, 3) \
                                    or triple == (4, 3, 4):
                                tot += PATTERN_343_PENALTY
                        if best is None or tot < best:
                            best, best_a = tot, a
                    ndp[(b, c)] = best
                    nback[(b, c)] = best_a
            dp, back[j] = ndp, nback

        bb, bc = min(dp, key=lambda k: dp[k])
        pick = [0] * L
        pick[L - 1], pick[L - 2] = bc, bb
        for j in range(L - 1, 1, -1):
            pick[j - 2] = back[j][(pick[j - 1], pick[j])]
        out = []
        for ln, cc in zip(layer_notes,
                           [cands[j][pick[j]] for j in range(L)]):
            out.extend(self._expand_layer_fingers(ln, cc))
        return out

    def _solve_viterbi_legacy(self, notes):
        """Note-by-note Viterbi (pre-Phase-3 path).

        Kept verbatim behind `legacy_chord_postpass = True`; the caller
        must still apply `_apply_chord_candidates` afterwards.
        State is the finger pair (f_{i-1}, f_i) -- 25 states, 125
        transitions per step -- so that, besides the consecutive-pair
        cost, a finger held by note i-2 can also be honoured when note i
        starts (B2). Back-pointers recover the cheapest path; ties keep
        the first minimum, so repeated runs agree exactly.

        F4b: chord groups are pinned to whole candidates by
        `_apply_chord_candidates` after the solve (not here: an order-2
        state cannot enforce whole-chord consistency).
        F8: Vectorized pair_cost table per note pair.
        """
        n = len(notes)
        if n == 0:
            return []
        if n == 1:
            h0 = _hint_of(notes[0])  # Phase 4 (§6): as in solve_layered
            return [h0 or 3]
        
        # F4b: Precompute chord candidates
        self._precompute_chord_candidates(notes)
        
        # F8: Precompute pair_cost tables for all consecutive note pairs
        # cost_table[i][fa][fb] = cost of (notes[i]=fa -> notes[i+1]=fb)
        cost_tables = []
        for i in range(n - 1):
            na, nb = notes[i], notes[i + 1]
            table = np.zeros((6, 6), dtype=np.float32)  # 1-indexed
            for fa in range(1, 6):
                for fb in range(1, 6):
                    table[fa, fb] = self.pair_cost(na, fa, nb, fb)
            cost_tables.append(table)

        fingers = (1, 2, 3, 4, 5)
        # Phase 4 (§6): PIN_COST for any transition whose finger for a
        # hinted note differs from the hint. notes[0] is priced via the
        # dp init below; notes[i+1] via cost_tables[i] (each hinted
        # note exactly once, so hint-free tables are untouched).
        for i in range(n - 1):
            hb = _hint_of(notes[i + 1])
            if hb:
                for fb in fingers:
                    if fb != hb:
                        cost_tables[i][:, fb] += PIN_COST
            # Valley bias: each interior note's outer-finger reward is
            # priced exactly once, as the arrival of cost_tables[i]
            # (notes[0]/notes[-1] have no both neighbours, so get none)
            # -- the same shared helper as the layered-DP node costs,
            # so both solvers agree.
            if i + 2 < n:
                nprev, ncur, nnext = notes[i], notes[i + 1], notes[i + 2]
                for fb in fingers:
                    cost_tables[i][:, fb] += _valley_bonus(
                        nprev, ncur, nnext, fb, self.LR)

        # dp[(fa, fb)]: cheapest cost of a path ending with fa on
        # notes[i-1] and fb on notes[i].
        dp = {}
        # F0/F6: Start preference - right hand prefers thumb (1), left prefers pinky (5)
        # Stronger preference to overcome the pairwise cost advantage of starting on 2/3
        # Left hand needs even stronger preference for pinky to avoid thumb-start traps
        start_pref = START_PREF_RIGHT if self.LR == "right" else START_PREF_LEFT
        h0 = _hint_of(notes[0])
        for fa in fingers:
            for fb in fingers:
                dp[(fa, fb)] = (cost_tables[0][fa, fb]
                                + start_pref.get(fa, 0.0)
                                + (0.0 if h0 == 0 or fa == h0 else PIN_COST))

        back = {1: {k: k[0] for k in dp}}
        for i in range(2, n):
            na, nb, nc = notes[i-1], notes[i], notes[i-2]
            nc_held = nc.time_real + nc.dur_real > nb.time_real + HELD_EPS
            ndp, nback = {}, {}

            for fa in fingers:
                for fb in fingers:
                    pc = cost_tables[i-1][fa, fb]
                    best, bestfp = None, None
                    for fp in fingers:
                        c = dp[(fp, fa)] + pc
                        # E4 (OFF default): penalise the awkward 3-4-3 /
                        # 4-3-4 weak-pair alternation (order-2 state).
                        if PATTERN_343_PENALTY and (
                                (fp, fa, fb) == (3, 4, 3)
                                or (fp, fa, fb) == (4, 3, 4)):
                            c += PATTERN_343_PENALTY
                        if nc_held and fp == fb and abs(nb.x - nc.x) > 1e-9:
                            c += HARD
                        if best is None or c < best:
                            best, bestfp = c, fp
                    ndp[(fa, fb)] = best
                    nback[(fa, fb)] = bestfp
            dp, back[i] = ndp, nback
        
        bfa, bfb = min(dp, key=lambda k: dp[k])
        out = [0] * n
        out[n-1], out[n-2] = bfb, bfa
        for i in range(n-1, 1, -1):
            out[i-2] = back[i][(out[i-1], out[i])]
        return out


    #####################################################
    def _repair_held_reference(self, notes, fingers):
        """Greedy fix for same-finger overlaps deeper than the Viterbi
        order-2 memory (a note held across 3+ later notes, typical of
        multi-voice beams, B2/F3).

        Walks the notes in order; a note sharing its finger with any
        still-held earlier note on a different key takes the cheapest
        free finger (measured with pair_cost against both neighbours).
        At most a handful of notes are ever held at once, so a free
        finger always exists in practice. Returns the corrected list.
        
        F7: Recompute total cost after repair; warn if increased; log if >1% notes repaired.
        """
        f = list(fingers)
        n = len(notes)
        fixed = 0
        for i in range(n):
            if f[i] not in (1,2,3,4,5):
                continue
            bi = notes[i]
            if _hint_of(bi):
                continue  # Phase 4 (§6): pinned notes keep their finger
            bi_end = bi.time_real + max(bi.dur_real, 0.0)
            blocked = set()
            for j in range(i):
                if f[j] not in (1,2,3,4,5):
                    continue
                aj = notes[j]
                if abs(aj.x - bi.x) < 1e-9:
                    continue  # same pitch: legal
                aj_end = aj.time_real + max(aj.dur_real, 0.0)
                if aj_end > bi.time_real + HELD_EPS \
                        and bi_end > aj.time_real + HELD_EPS:
                    blocked.add(f[j])
            if f[i] not in blocked:
                continue
            best, bestc = None, None
            for cand in (1,2,3,4,5):
                if cand in blocked:
                    continue
                # F4b: repairing a chord member must not break the chord's
                # pitch order. In (possibly mirrored) x, both hands want
                # fingers strictly increasing with x (left x is negated in
                # generate(), so low pitch = large x = large finger).
                if bi.isChord:
                    keep = True
                    for j in range(n):
                        if j == i or f[j] not in (1,2,3,4,5):
                            continue
                        aj = notes[j]
                        if not aj.isChord or aj.chordID != bi.chordID:
                            continue
                        if abs(aj.x - bi.x) < 1e-9:
                            if cand != f[j]:
                                keep = False  # same pitch: same finger
                                break
                            continue
                        if (aj.x < bi.x and not f[j] < cand) or \
                           (aj.x > bi.x and not cand < f[j]):
                            keep = False
                            break
                    if not keep:
                        continue
                c = 0.0
                if i > 0:
                    c += self.pair_cost(notes[i-1], f[i-1], bi, cand)
                if i+1 < n:
                    c += self.pair_cost(bi, cand, notes[i+1], f[i+1])
                if best is None or c < bestc:
                    best, bestc = cand, c
            if best is not None and best != f[i]:
                if self.verbose:
                    print('  held-finger repair: note %d (%s%s) finger %d -> %d'
                          % (i, bi.name, bi.octave, f[i], best))
                f[i] = best
                fixed += 1
            elif self.verbose:
                print('  held-finger repair: note %d (%s%s) keeps conflicting'
                      ' finger %d (no free finger)' % (i, bi.name, bi.octave, f[i]))
        
        # F7: Recompute total cost after repair
        if fixed:
            cost_before = self.total_cost(notes, fingers)
            cost_after = self.total_cost(notes, f)
            if cost_after > cost_before:
                if self.verbose:
                    print('  WARNING: held-finger repair increased total cost: %.2f -> %.2f'
                          % (cost_before, cost_after))
            if fixed > 0.01 * n:
                if self.verbose:
                    print('  held-finger repair: fixed %d / %d notes (%.1f%%)'
                          % (fixed, n, 100.0 * fixed / n))
            elif self.verbose:
                print('  held-finger repair: fixed %d note(s)' % fixed)
        return f


    #####################################################
    def _repair_held(self, notes, fingers):
        """Linear-time version of _repair_held_reference (F7).

        Same signature and same results. Notes must be sorted by
        time_real (unsorted input delegates to the reference). Keeps an
        `active` list of earlier indices whose note may still sound: per
        note, expired entries (end <= onset + HELD_EPS) can never matter
        again because onsets never decrease, so `blocked` is computed
        from `active` only, with the same conditions as the reference.
        The chord-order check iterates a precomputed chord_members map
        instead of range(n). Verbose printing and the F7 cost report are
        unchanged.
        """
        n = len(notes)
        if not all(notes[k].time_real <= notes[k + 1].time_real + 1e-9
                   for k in range(n - 1)):
            return self._repair_held_reference(notes, fingers)
        chord_members = {}
        for k, nk in enumerate(notes):
            if nk.isChord:
                chord_members.setdefault(nk.chordID, []).append(k)
        f = list(fingers)
        fixed = 0
        active = []
        for i in range(n):
            bi = notes[i]
            bi_end = bi.time_real + max(bi.dur_real, 0.0)
            active = [j for j in active
                      if notes[j].time_real + max(notes[j].dur_real, 0.0)
                      > bi.time_real + HELD_EPS]
            if f[i] not in (1,2,3,4,5):
                active.append(i)
                continue
            if _hint_of(bi):
                # Phase 4 (§6): pinned notes keep their finger, but
                # they still sound, so they still block the others.
                active.append(i)
                continue
            blocked = set()
            for j in active:
                if f[j] not in (1,2,3,4,5):
                    continue
                aj = notes[j]
                if abs(aj.x - bi.x) < 1e-9:
                    continue  # same pitch: legal
                aj_end = aj.time_real + max(aj.dur_real, 0.0)
                if aj_end > bi.time_real + HELD_EPS \
                        and bi_end > aj.time_real + HELD_EPS:
                    blocked.add(f[j])
            if f[i] not in blocked:
                active.append(i)
                continue
            best, bestc = None, None
            for cand in (1,2,3,4,5):
                if cand in blocked:
                    continue
                # F4b: repairing a chord member must not break the chord's
                # pitch order. In (possibly mirrored) x, both hands want
                # fingers strictly increasing with x (left x is negated in
                # generate(), so low pitch = large x = large finger).
                if bi.isChord:
                    keep = True
                    for j in chord_members.get(bi.chordID, ()):
                        if j == i or f[j] not in (1,2,3,4,5):
                            continue
                        aj = notes[j]
                        if abs(aj.x - bi.x) < 1e-9:
                            if cand != f[j]:
                                keep = False  # same pitch: same finger
                                break
                            continue
                        if (aj.x < bi.x and not f[j] < cand) or \
                           (aj.x > bi.x and not cand < f[j]):
                            keep = False
                            break
                    if not keep:
                        continue
                c = 0.0
                if i > 0:
                    c += self.pair_cost(notes[i-1], f[i-1], bi, cand)
                if i+1 < n:
                    c += self.pair_cost(bi, cand, notes[i+1], f[i+1])
                if best is None or c < bestc:
                    best, bestc = cand, c
            if best is not None and best != f[i]:
                if self.verbose:
                    print('  held-finger repair: note %d (%s%s) finger %d -> %d'
                          % (i, bi.name, bi.octave, f[i], best))
                f[i] = best
                fixed += 1
            elif self.verbose:
                print('  held-finger repair: note %d (%s%s) keeps conflicting'
                      ' finger %d (no free finger)' % (i, bi.name, bi.octave, f[i]))
            active.append(i)

        # F7: Recompute total cost after repair
        if fixed:
            cost_before = self.total_cost(notes, fingers)
            cost_after = self.total_cost(notes, f)
            if cost_after > cost_before:
                if self.verbose:
                    print('  WARNING: held-finger repair increased total cost: %.2f -> %.2f'
                          % (cost_before, cost_after))
            if fixed > 0.01 * n:
                if self.verbose:
                    print('  held-finger repair: fixed %d / %d notes (%.1f%%)'
                          % (fixed, n, 100.0 * fixed / n))
            elif self.verbose:
                print('  held-finger repair: fixed %d note(s)' % fixed)
        return f


    #####################################################
    def optimize_seq(self, nseq, istart):
        '''Generate meaningful fingering for a note sequence of size depth'''

        #---------------------------------------------------------
        def skip(fa,fb, na,nb, level):
            ### two-consecutive-notes movement, skipping rules ###
            # fa is fingering for note na, level is passed only for debugging

            xba = nb.x - na.x  # physical distance btw the second to first note, in cm

            if not na.isChord and not nb.isChord: # neither of the 2 notes live in a chord
                if fa==fb and xba and na.duration<4:
                    return True # play different notes w/ same finger, skip
                if fa>1 : # if a is not thumb
                    if fb>1 and (fb-fa)*xba<0: return True # non-thumb fingers are crossings, skip
                    if fb==1 and nb.isBlack and xba>0: return True # crossing thumb goes to black, skip
                else: # a is played by thumb:
                    # skip if  a is black  and  b is behind a  and  fb not thumb  and na.duration<2:
                    if na.isBlack and xba<0 and fb>1 and na.duration<2: return True

            elif na.isChord and nb.isChord and na.chordID == nb.chordID:
                # na and nb are notes in the same chord
                if fa==fb: return True   # play different chord notes w/ same finger, skip
                if fa<fb and self.LR=='left' : return True
                if fa>fb and self.LR=='right': return True
                # NOTE: legacy axba scaling (raw cm * hf / 0.8) is inverted
                # vs _span_cost (raw cm vs limit * hf); left as-is, legacy only.
                axba = abs(xba)*self.hf /0.8
                # max normalized distance in cm btw 2 consecutive fingers
                if axba> 5 and (fa==3 and fb==4 or fa==4 and fb==3): return True
                if axba> 5 and (fa==4 and fb==5 or fa==5 and fb==4): return True
                if axba> 6 and (fa==2 and fb==3 or fa==3 and fb==2): return True
                if axba> 7 and (fa==2 and fb==4 or fa==4 and fb==2): return True
                if axba> 8 and (fa==3 and fb==5 or fa==5 and fb==3): return True
                if axba>11 and (fa==2 and fb==5 or fa==5 and fb==2): return True
                if axba>12 and (fa==1 and fb==2 or fa==2 and fb==1): return True
                if axba>14 and (fa==1 and fb==3 or fa==3 and fb==1): return True
                if axba>16 and (fa==1 and fb==4 or fa==4 and fb==1): return True

            return False
        #---------------------------------------------------------------------------

        if self.autodepth:
            #choose depth based on time span of 3.5 seconds
            if nseq[0].isChord:
                self.depth = max(3, nseq[0].NinChord - nseq[0].chordnr + 1)
            else:
                tn0 = nseq[0].time
                for i in (4,5,6,7,8,9):
                    self.depth = i
                    if nseq[i-1].time - tn0 > 3.5:
                        break
        depth = self.depth

        fingers = (1,2,3,4,5)
        n1, n2, n3, n4, n5, n6, n7, n8, n9 = nseq
        u_start = [istart]
        if istart == 0:
            u_start = fingers

        #####################################
        out = ([0 for i in range(depth)], -1)
        minvel = 1.e+10
        for f1 in u_start:
            for f2 in fingers:
                if skip(f1,f2, n1,n2, 2): continue
                for f3 in fingers:
                    if f3 and skip(f2,f3, n2,n3, 3): continue
                    if depth<4: u=[False]
                    else:       u=fingers
                    for f4 in u:
                        if f4 and skip(f3,f4, n3,n4, 4): continue
                        if depth<5: u=[False]
                        else:       u=fingers
                        for f5 in u:
                            if f5 and skip(f4,f5, n4,n5, 5): continue
                            if depth<6: u=[False]
                            else:       u=fingers
                            for f6 in u:
                                if f6 and skip(f5,f6, n5,n6, 6): continue
                                if depth<7: u=[False]
                                else:       u=fingers
                                for f7 in u:
                                    if f7 and skip(f6,f7, n6,n7, 7): continue
                                    if depth<8: u=[False]
                                    else:       u=fingers
                                    for f8 in u:
                                        if f8 and skip(f7,f8, n7,n8, 8): continue
                                        if depth<9: u=[False]
                                        else:       u=fingers
                                        for f9 in u:
                                            if f9 and skip(f8,f9, n8,n9, 9): continue
                                            c = [f1,f2,f3,f4,f5,f6,f7,f8,f9]
                                            v = self.ave_velocity(c, nseq)
                                            if v < minvel:
                                                out = (c, v)
                                                minvel  = v
        # if out[1]==-1: exit() #no combination found
        return out


    ###########################################################################################
    def generate(self, start_measure=0, nmeasures=1000):
        """Find a finger (1..5) for every note in the [start_measure,
        start_measure + nmeasures) range (measures are 1-based on the
        command line; start_measure == 1 is mapped to 0 as before).

        Unless self.legacy_window is True, the whole in-range slice is
        solved at once with solve_viterbi() (B1): no dummy-note padding
        is needed for short scores, and a single note gets finger 3.
        solve_viterbi() runs the Phase 3 layered chord-as-note DP by
        default (whole-chord candidates, no post-pass); with
        self.legacy_chord_postpass = True it runs the legacy
        note-by-note Viterbi plus the chord post-pass instead.
        The legacy 9-note sliding-window search lives in
        _generate_legacy(), where -d/--depth still applies.
        """

        if start_measure == 1:
            start_measure=0 # avoid confusion with python numbering

        if self.verbose:
            print('Your hand span set to size-'+self.size,
                  'which is', 21*self.hf, 'cm')
            print('(max relaxed distance between thumb and pinkie)')

        if self.LR == "left":
            for anote in self.noteseq:
                if not getattr(anote, "_mirrored", False):
                    anote.x = -anote.x  # play left as a right on a mirrored keyboard
                    anote._mirrored = True   # ...only once: generate() may run again

        if self.legacy_window:
            return self._generate_legacy(start_measure, nmeasures)

        self.fingerseq = []        # fingertip fan of every processed note ..
        self.fingerseq_index = []  # .. and the noteseq index it belongs to

        kept = []
        for i, an in enumerate(self.noteseq):
            if an.measure:
                if an.measure < start_measure : continue
                if an.measure >= start_measure + nmeasures : break
            kept.append(i)
        if not kept:
            return

        if len(kept) == 1:
            h0 = _hint_of(self.noteseq[kept[0]])  # Phase 4 (§6): pinned
            fingers = [h0 or 3]                  # lone note takes hint
        else:
            sub = [self.noteseq[i] for i in kept]
            fingers = self.solve_viterbi(sub)
            fingers = self._apply_chord_candidates(sub, fingers)
            fingers = self._repair_held(sub, fingers)

        for k, i in enumerate(kept):##############

            an = self.noteseq[i]
            best_finger = fingers[k]

            an.fingering = best_finger
            for j in (1,2,3,4,5):
                self.cfps[j] = (self.frest[j]-self.frest[best_finger]) + an.x
            self.fingerseq.append(list(self.cfps))
            self.fingerseq_index.append(i)

            if best_finger>0:
                fng = Fingering(best_finger)
                if an.isChord:
                    if len(an.chord21.pitches) < 3:
                        # dont show fingering in the lyrics line for >3 note-chords
                        if self.lyrics:
                            nl = len(an.chord21.pitches) - an.chordnr
                            an.chord21.addLyric(best_finger, nl)
                        else:
                            # a chord appends one fingering per pitch;
                            # only its first note purges the old ones
                            _set_fingering(an.chord21, fng,
                                           replace=an.chordnr == 0)
                else:
                    if self.lyrics:
                        an.note21.addLyric(best_finger)
                    else:
                        _set_fingering(an.note21, fng)

            #---------------------------------------------------------------------------- print
            if self.verbose:
                if an.measure: print(f"meas.{an.measure: <3}", end=' ')
                print(f"finger_{best_finger}  plays  {an.name: >2}{an.octave}")
            else:
                if k and not k%100 and an.measure:
                    print('scanned', k, '/', len(kept),
                          'notes, measure', an.measure+1, ' for the', self.LR ,'hand...')


    ###########################################################################################
    def _generate_legacy(self, start_measure=0, nmeasures=1000):
        """Legacy 9-note sliding-window search (optimize_seq).

        Kept for comparison (B5) behind self.legacy_window; this is the
        only mode where self.depth / self.autodepth (-d/--depth) apply.
        start_measure is already converted (1 -> 0) by generate()."""

        start_finger, out, vel = 0, [0 for i in range(9)], 0
        entry_autodepth, entry_depth = self.autodepth, self.depth
        N = len(self.noteseq)
        if self.depth < 3: self.depth = 3
        if self.depth > 9: self.depth = 9

        # fewer than 9 notes: optimize_seq needs exactly 9, so pad the window
        # with dummy notes (same pitch, later in time) and solve it once.
        if 2 <= N < 9:
            pad = list(self.noteseq)
            last = pad[-1]
            for k in range(9 - N):
                q = copy.copy(last)
                q.isChord = False
                q.time = last.time + k + 1
                pad.append(q)
            self.autodepth = False
            self.depth = 9
            res, vel = self.optimize_seq(pad, 0)
            out = [0] + list(res)      # slot 0 is "consumed" by pop(1) below

        self.fingerseq = []        # fingertip fan of every processed note ..
        self.fingerseq_index = []  # .. and the noteseq index it belongs to

        for i in range(N):##############

            an = self.noteseq[i]
            if an.measure:
                if an.measure < start_measure : continue
                if an.measure >= start_measure + nmeasures : break

            if i > N-11:
                self.autodepth = False
                self.depth = 9

            best_finger = 0
            ninenotes = self.noteseq[i : i+9]
            if i > N-9:                 # window of 9 is still complete at i=N-9
                if len(out)>1: best_finger = out.pop(1)
            else:
                out, vel  = self.optimize_seq(ninenotes, start_finger)
                best_finger  = out[0]
                start_finger = out[1]

            an.fingering = best_finger
            outfan = list(out)
            if outfan: outfan[0] = best_finger
            self.set_fingers_positions(outfan, ninenotes, 0)
            self.fingerseq.append(list(self.cfps))
            self.fingerseq_index.append(i)

            if best_finger>0:
                fng = Fingering(best_finger)
                if an.isChord:
                    if len(an.chord21.pitches) < 3:
                        # dont show fingering in the lyrics line for >3 note-chords
                        if self.lyrics:
                            nl = len(an.chord21.pitches) - an.chordnr
                            an.chord21.addLyric(best_finger, nl)
                        else:
                            # a chord appends one fingering per pitch;
                            # only its first note purges the old ones
                            _set_fingering(an.chord21, fng,
                                           replace=an.chordnr == 0)
                else:
                    if self.lyrics:
                        an.note21.addLyric(best_finger)
                    else:
                        _set_fingering(an.note21, fng)

            #---------------------------------------------------------------------------- print
            if self.verbose:
                if not best_finger: best_finger = '?'
                if an.measure: print(f"meas.{an.measure: <3}", end=' ')
                print(f"finger_{best_finger}  plays  {an.name: >2}{an.octave}", end=' ')
                if i < N-10:
                    print(f"  v={round(vel,1)}", end='')
                    if self.autodepth:
                        print("\t "+str(out[0:self.depth]) + " d:" + str(self.depth))
                    else:
                        print("\t"+("   "*(i%self.depth))+str(out[0:self.depth]))
                else:
                    print()
            else:
                if i and not i%100 and an.measure:
                    print('scanned', i, '/', N,
                          'notes, measure', an.measure+1, ' for the', self.LR ,'hand...')

        # Restore the depth settings the caller had on entry: the scan
        # forces depth 9 on its last stretch, and leaving that behind
        # would make a second generate() solve other depths and find a
        # different fingering for the same notes.
        self.autodepth = entry_autodepth
        self.depth = entry_depth




