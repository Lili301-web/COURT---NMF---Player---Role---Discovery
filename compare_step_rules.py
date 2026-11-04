"""Compare the solver's step rules on the actual COURT-NMF objective.

The paper claims a purpose-built solver rather than a generic NMF routine.  That
claim is worth evidence, and the obvious challenge to it is the multiplicative
update (MU) that is native to standard NMF: it needs no step size, keeps
non-negativity by construction, and has a monotonicity proof.  This script runs
every available rule from the same seeded initialisation on the same objective
and records what each one reaches.

Rules
    backtracking  alternating projected gradient with an Armijo search (default)
    mu            Lee-Seung multiplicative updates extended to all four terms of
                  Eq. (1), with the row simplex restored by renormalisation
    eg            exponentiated gradient (entropic mirror descent) on W, which
                  stays on the simplex without any projection, MU on H
    fixed         projected gradient with one small constant step
    harmonic      the original 1/t Robbins-Monro schedule

Writes ``outputs/solver_comparison.csv`` and ``.json``.

Usage
-----
    python compare_step_rules.py                       # all rules
    python compare_step_rules.py --rules mu backtracking
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"

_spec = importlib.util.spec_from_file_location("cne", Path(__file__).with_name("court_nmf_experiment.py"))
cne = importlib.util.module_from_spec(_spec)
sys.modules["cne"] = cne
_spec.loader.exec_module(cne)

# MU needs a far larger cap than the gradient rules, so each rule gets a cap
# generous enough that failing to converge is a property of the rule and not of
# the budget it was given.
CAPS = {"backtracking": 4000, "fixed": 20000, "harmonic": 20000, "eg": 20000, "mu": 20000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="+",
                    default=["backtracking", "mu", "eg", "fixed", "harmonic"])
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=cne.SEED)
    a = ap.parse_args()

    df, X, _ = cne.preprocess()
    edges = cne.career_edges(df)
    rows = []
    for rule in a.rules:
        cap = CAPS.get(rule, 20000)
        t0 = time.time()
        W, H, B, out = cne.fit_court(df, X, iterations=cap, seed=a.seed, edges=edges,
                                     step_rule=rule, tol=a.tol, patience=a.patience)
        elapsed = time.time() - t0
        d = out["diagnostics"]
        hist = pd.DataFrame(out["history"])
        hist.assign(step_rule=rule).to_csv(OUT / f"solver_history_{rule}.csv", index=False)
        rows.append({
            "step_rule": rule,
            "converged": bool(d["converged"]),
            "iterations_to_converge": int(d["stopped_at"]),
            "iteration_cap": cap,
            "wall_seconds": round(elapsed, 1),
            "final_objective": float(hist["objective"].iloc[-1]),
            "weighted_mse": float(d["weighted_mse"]),
            "trajectory_mse": float(d["trajectory_mse"]),
            "mean_max_membership": float(d["mean_max_membership"]),
            "mean_entropy": float(d["mean_entropy"]),
            "monotone_violations": int(d.get("monotone_violations", 0)),
            "needs_step_size": rule not in ("mu",),
            "simplex_exact": rule not in ("mu",),
        })
        print(f"{rule:<14} conv={rows[-1]['converged']!s:<5} at {rows[-1]['iterations_to_converge']:>6} "
              f"obj {rows[-1]['final_objective']:>10.2f} in {elapsed:>6.0f}s", flush=True)

    d = pd.DataFrame(rows).sort_values("final_objective").reset_index(drop=True)
    best = d["final_objective"].min()
    d["pct_above_best"] = (d["final_objective"] / best - 1) * 100
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "solver_comparison.csv", index=False)
    (OUT / "solver_comparison.json").write_text(
        json.dumps({"seed": a.seed, "tol": a.tol, "patience": a.patience,
                    "rows": d.to_dict("records")}, indent=2), encoding="utf-8")
    print()
    print(d[["step_rule", "converged", "iterations_to_converge", "wall_seconds",
             "final_objective", "pct_above_best", "monotone_violations"]].to_string(index=False))


if __name__ == "__main__":
    main()
