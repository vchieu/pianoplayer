"""Parameter tuning harness (Phase 2, step 2.1 of PIANOPLAYER_FINGERING_PLAN.md).

Coordinate descent over existing cost constants. Never edits source files:
it patches module attributes in memory, evaluates, restores everything in
a `finally` block, then prints a report plus a unified-diff-style suggestion
a human applies by hand.

    python tune_fingering.py [pig_path_or_dir]

Objective: first reject any candidate that breaks the hard gold rows
(eval_fingering GOLD at default tempo + C scales at 60/200 bpm, KNOWN_FAIL
handling unchanged); then maximize the training PIG note-weighted average
(hold-out = sorted pieces with index % 5 == 0, at most 60 pieces). With no
dataset: maximize GOLD_EXTRA exact-match % and print the "no PIG" hint line.

Acceptance for this step: runs without a dataset, and its baseline
objective equals the one printed by eval_fingering.py (same solve/GOLD_EXTRA
imported from there, so equality holds by construction).
"""
import copy
import glob
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_fingering import GOLD, GOLD_EXTRA, KNOWN_FAIL, solve  # noqa: E402
from pianoplayer.hand import Hand  # noqa: E402
from pianoplayer.pigloader import load_pig  # noqa: E402

MAX_PIECES = 60
MIN_IMPROVEMENT = 0.1  # percentage points
MAX_SWEEPS = 2

# Entries: (module_name, attr, candidates) for scalars, or
# (module_name, attr, key, candidates) for dict attributes.
PARAMS = [
    ("pianoplayer.hand", "REPEAT_PENALTY", [3.0, 6.0, 9.0]),
    ("pianoplayer.hand", "WEAK_FINGER_PENALTY", [1.5, 3.0, 5.0]),
    ("pianoplayer.hand", "SAME_FINGER_BASE", [1.0, 2.0, 4.0]),
    ("pianoplayer.hand", "SEC_SMOOTH", [0.05, 0.10, 0.20]),
    ("pianoplayer.handspec", "UNDER", 2, [1.5, 3.0, 6.0]),
    ("pianoplayer.handspec", "UNDER", 3, [0.5, 1.0, 2.0]),
    ("pianoplayer.handspec", "UNDER", 4, [4.0, 8.0, 16.0]),
    ("pianoplayer.handspec", "UNDER", 5, [4.5, 9.0, 18.0]),
    ("pianoplayer.handspec", "BLACK_COST", 1, [0.0, 0.5, 1.0]),
    ("pianoplayer.handspec", "BLACK_COST", 5, [0.0, 0.3, 0.6]),
]


def _mod(name):
    return importlib.import_module(name)


def entry_label(entry):
    if len(entry) == 3:
        return "%s.%s" % (entry[0].split(".")[-1], entry[1])
    return "%s.%s[%s]" % (entry[0].split(".")[-1], entry[1], entry[2])


def get_current(entry):
    mod = _mod(entry[0])
    if len(entry) == 3:
        return getattr(mod, entry[1])
    return dict(getattr(mod, entry[1]))[entry[2]]


def apply_value(entry, value):
    mod = _mod(entry[0])
    if len(entry) == 3:
        setattr(mod, entry[1], value)
    else:
        _attr, key = entry[1], entry[2]
        newd = dict(getattr(mod, _attr))
        newd[key] = value
        setattr(mod, _attr, newd)


def snapshot_all():
    """Saved originals for every patched (module, attr): scalar or dict copy."""
    saved = {}
    for entry in PARAMS:
        mod = _mod(entry[0])
        key = (entry[0], entry[1])
        if key not in saved:
            val = getattr(mod, entry[1])
            saved[key] = dict(val) if isinstance(val, dict) else val
    return saved


def restore_all(saved):
    for (modname, attr), val in saved.items():
        setattr(_mod(modname), attr, copy.deepcopy(val))


# ------------------------------------------------------------------ metrics --
def gold_ok():
    """Hard-gold gate: mirrors eval_fingering test_gold (default tempo) +
    test_tempo_dependence (C scales at 60 and 200 bpm). XPASS = failure."""
    for label, names, side, want in GOLD:
        got, _h = solve(names, side)
        if label in KNOWN_FAIL:
            if list(got) == list(want):
                return False  # XPASS: known-fail row newly passes
        elif list(got) != list(want):
            return False
    for bpm in (60, 200):
        gotR, _h = solve(
            ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
            "right", bpm=bpm)
        if list(gotR) != [1, 2, 3, 1, 2, 3, 4, 5]:
            return False
        gotL, _h = solve(
            ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
            "left", bpm=bpm)
        if list(gotL) != [5, 4, 3, 2, 1, 3, 2, 1]:
            return False
    return True


def gold_extra_pct():
    """Objective in no-dataset mode: GOLD_EXTRA exact-match %, same helper
    as eval_fingering.test_gold_extra (baseline equality by construction)."""
    match = 0
    for label, names, side, want in GOLD_EXTRA:
        got, _h = solve(names, side)
        match += list(got) == list(want)
    return 100.0 * match / len(GOLD_EXTRA)


def score_pig_files(files):
    """Note-weighted match % over a set of PIG files (matches eval's
    pig_match_rate internals, aggregated over files). Returns None when no
    comparable notes."""
    match_total, known_total = 0, 0
    for fn in files:
        try:
            data = load_pig(fn)
        except Exception:
            continue
        for channel, side in ((0, "right"), (1, "left")):
            inotes, experts = data[channel]
            if not any(e is not None for e in experts):
                continue
            h = Hand(side, "M")
            h.verbose = False
            try:
                h.noteseq = inotes
                h.generate()
            except Exception:
                continue
            for n, e in zip(h.noteseq, experts):
                if e is None:
                    continue
                known_total += 1
                match_total += n.fingering == e
    if not known_total:
        return None
    return 100.0 * match_total / known_total


def collect_pig(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.txt")))
    else:
        files = [path]
    return files[:MAX_PIECES]


# ------------------------------------------------------------------- search --

def main():
    pig_path = sys.argv[1] if len(sys.argv) > 1 else None
    pig_files = None
    if pig_path and os.path.exists(pig_path):
        pig_files = collect_pig(pig_path)
    use_pig = bool(pig_files)

    if use_pig:
        train = [f for i, f in enumerate(pig_files) if i % 5 != 0]
        holdout = [f for i, f in enumerate(pig_files) if i % 5 == 0]
        print("tune: %d PIG pieces (train %d, hold-out %d)"
              % (len(pig_files), len(train), len(holdout)))

        def objective():
            return score_pig_files(train)
    else:
        print("tune: no PIG dataset given -- maximizing GOLD_EXTRA %; "
              "no PIG: result is only a hint")

        def objective():
            return gold_extra_pct()

    saved = snapshot_all()
    old_values = [get_current(e) for e in PARAMS]
    try:
        if not gold_ok():
            print("tune: BASELINE BREAKS GOLD -- refusing to tune")
            return 1
        base_obj = objective()
        if base_obj is None:
            print("tune: no comparable PIG notes -- nothing to maximize")
            return 1
        if use_pig:
            base_hold = score_pig_files(holdout)
            print("tune: baseline train %.2f%% hold-out %s"
                  % (base_obj, ("%.2f%%" % base_hold)
                     if base_hold is not None else "n/a"))
        else:
            print("tune: baseline GOLD_EXTRA %.1f%%" % base_obj)

        best = {i: old_values[i] for i in range(len(PARAMS))}
        best_obj = base_obj
        # current applied state tracks best-so-far across params
        for i, entry in enumerate(PARAMS):
            apply_value(entry, best[i])
        rows = []  # (label, old, new, train_delta, hold_delta)
        for sweep in range(1, MAX_SWEEPS + 1):
            changed = False
            for i, entry in enumerate(PARAMS):
                label = entry_label(entry)
                cands = entry[-1]
                local_best_val, local_best_obj = best[i], best_obj
                for cand in cands:
                    apply_value(entry, cand)
                    if not gold_ok():
                        continue
                    obj = objective()
                    if obj is None:
                        continue
                    if obj > local_best_obj:
                        local_best_obj, local_best_val = obj, cand
                apply_value(entry, best[i])  # revert to pre-trial state
                if local_best_obj > best_obj + MIN_IMPROVEMENT:
                    rows.append((label, best[i], local_best_val,
                                 local_best_obj - best_obj))
                    best[i] = local_best_val
                    best_obj = local_best_obj
                    changed = True
                apply_value(entry, best[i])
            if not changed:
                print("tune: converged after sweep %d" % sweep)
                break
        else:
            print("tune: sweep budget (%d) exhausted" % MAX_SWEEPS)

        print("tune: final objective %.2f%% (baseline %.2f%%)" %
              (best_obj, base_obj))
        if use_pig:
            final_hold = score_pig_files(holdout)
            print("tune: hold-out before %s after %s"
                  % (("%.2f%%" % base_hold)
                     if base_hold is not None else "n/a",
                     ("%.2f%%" % final_hold)
                     if final_hold is not None else "n/a"))
        print("")
        print("%-22s %10s %10s %12s %12s" %
              ("param", "old", "new", "train delta", "hold-out delta"))
        if not rows:
            print("(no parameter improved the objective by > %.1f point; "
                  "no change suggested)" % MIN_IMPROVEMENT)
        for label, old, new, delta in rows:
            if use_pig:
                print("%-22s %10s %10s %+11.2fpp %12s" %
                      (label, old, new, delta, "see above"))
            else:
                print("%-22s %10s %10s %+11.2fpp %12s" %
                      (label, old, new, delta, "n/a (no PIG)"))
        print("")
        print("Suggested patch (apply by hand, then re-run the suites):")
        if not rows:
            print("(none)")
        else:
            print("--- a/cost-constants")
            print("+++ b/cost-constants")
            for label, old, new, _delta in rows:
                print("-%s = %r" % (label, old))
                print("+%s = %r" % (label, new))
        return 0
    finally:
        restore_all(saved)


if __name__ == "__main__":
    sys.exit(main())
