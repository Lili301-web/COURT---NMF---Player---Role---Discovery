"""Direct external validation of COURT-NMF role memberships.

Joins the independently labelled 195-player evaluation set to the COURT-NMF
player-season memberships using pid, year, and team. It reports a chi-square
test for dominant roles and Kruskal--Wallis tests for each soft role membership.

The dominant-role crosstab can be degenerate: if a role is nobody's argmax among
the labelled sample, padding the table to all five roles leaves an all-zero row,
whose expected frequencies are zero and which makes ``chi2_contingency`` raise.
The full padded table is still written out (an empty role is a finding, not an
error), but the test itself runs on the non-empty sub-table, and falls back to a
permutation p-value whenever an expected cell drops below 5 -- which is when the
asymptotic chi-square stops being trustworthy anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal


BASE = Path(__file__).resolve().parents[1]
LABELS = BASE / "outputs" / "expanded_validation" / "expanded_validation_labeled_sample.csv"
MEMBERSHIPS = BASE / "outputs" / "court_nmf" / "COURT-NMF_memberships.csv"
OUT = BASE / "outputs" / "court_nmf_validation"
POSITIONS = ["guard", "forward", "center"]


def cramers_v(chi2: float, n: int, rows: int, cols: int) -> float:
    denominator = n * min(rows - 1, cols - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denominator))


def permutation_chi_square(table: pd.DataFrame, n_perm: int = 9999,
                           seed: int = 42) -> float:
    """Monte-Carlo p-value for the chi-square statistic of a contingency table.

    Used instead of the asymptotic p-value when any expected count is below 5.
    Rebuilds the two label vectors the table came from, shuffles one of them, and
    counts how often a shuffled table reaches the observed statistic.
    """
    rows, cols = [], []
    for i, r in enumerate(table.index):
        for j, c in enumerate(table.columns):
            rows.extend([i] * int(table.iat[i, j]))
            cols.extend([j] * int(table.iat[i, j]))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    n_rows, n_cols = table.shape

    def statistic(a: np.ndarray, b: np.ndarray) -> float:
        observed = np.zeros((n_rows, n_cols))
        np.add.at(observed, (a, b), 1)
        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        total = observed.sum()
        expected = row_sums @ col_sums / total
        mask = expected > 0
        return float((((observed - expected) ** 2)[mask] / expected[mask]).sum())

    observed_statistic = statistic(rows, cols)
    rng = np.random.default_rng(seed)
    hits = sum(statistic(rows, rng.permutation(cols)) >= observed_statistic
               for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def safe_chi_square(padded: pd.DataFrame) -> dict:
    """Chi-square on the non-empty sub-table, with the degenerate cases handled."""
    table = padded.loc[padded.sum(axis=1) > 0, padded.sum(axis=0) > 0]
    dropped_rows = [r for r in padded.index if r not in table.index]
    dropped_cols = [c for c in padded.columns if c not in table.columns]

    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "chi_square": float("nan"), "degrees_of_freedom": 0,
            "p_value": float("nan"), "p_value_method": "not testable",
            "cramers_v": float("nan"), "all_expected_at_least_five": False,
            "min_expected": float("nan"),
            "tested_shape": list(table.shape),
            "empty_rows_dropped": dropped_rows, "empty_columns_dropped": dropped_cols,
            "note": ("Fewer than two non-empty rows or columns remain, so no test of "
                     "association is defined. The dominant-role assignment has "
                     "collapsed onto a single role."),
            "expected": None,
        }

    chi2, p_asymptotic, dof, expected = chi2_contingency(table)
    all_five = bool((expected >= 5).all())
    if all_five:
        p_value, method = float(p_asymptotic), "asymptotic"
    else:
        p_value, method = permutation_chi_square(table), "permutation (9,999 shuffles)"

    return {
        "chi_square": float(chi2), "degrees_of_freedom": int(dof),
        "p_value": float(p_value), "p_value_method": method,
        "cramers_v": cramers_v(float(chi2), int(table.to_numpy().sum()), *table.shape),
        "all_expected_at_least_five": all_five,
        "min_expected": float(expected.min()),
        "tested_shape": list(table.shape),
        "empty_rows_dropped": dropped_rows, "empty_columns_dropped": dropped_cols,
        "note": None,
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)
    memberships = pd.read_csv(MEMBERSHIPS)
    roles = [col for col in memberships.columns if col.startswith("role_")]
    required = ["pid", "year", "team"]
    merged = labels.merge(
        memberships[required + roles + ["dominant_role"]],
        on=required,
        how="inner",
        validate="one_to_one",
    )
    merged = merged[merged["position"].isin(POSITIONS)].copy()
    if len(merged) != 195:
        raise RuntimeError(f"Expected 195 labelled player-seasons after exact join; found {len(merged)}.")

    merged.to_csv(OUT / "court_nmf_labeled_memberships.csv", index=False)
    role_names = {idx: f"Role {idx + 1}" for idx in range(len(roles))}
    merged["dominant_role_name"] = merged["dominant_role"].map(role_names)
    crosstab = pd.crosstab(merged["dominant_role_name"], merged["position"]).reindex(
        index=list(role_names.values()), columns=POSITIONS, fill_value=0
    )
    test = safe_chi_square(crosstab)
    crosstab.to_csv(OUT / "court_nmf_dominant_role_crosstab.csv")
    if test["expected"] is not None:
        test["expected"].to_csv(OUT / "court_nmf_dominant_role_expected.csv")
    if test["empty_rows_dropped"]:
        print(f"WARNING: {len(test['empty_rows_dropped'])} role(s) are nobody's dominant "
              f"role in the labelled sample and were excluded from the test: "
              f"{', '.join(test['empty_rows_dropped'])}")
    if not test["all_expected_at_least_five"] and test["degrees_of_freedom"]:
        print(f"WARNING: smallest expected count is {test['min_expected']:.2f} (< 5); "
              f"reporting a permutation p-value instead of the asymptotic one.")

    membership_means = merged.groupby("position")[roles].mean().reindex(POSITIONS)
    membership_means.to_csv(OUT / "court_nmf_soft_membership_means_by_position.csv")
    soft_tests = []
    for role in roles:
        values = [merged.loc[merged["position"] == pos, role].to_numpy() for pos in POSITIONS]
        statistic, role_p = kruskal(*values)
        soft_tests.append({
            "role": role,
            "kruskal_wallis_H": float(statistic),
            "p_value": float(role_p),
            "bonferroni_p_value": float(min(role_p * len(roles), 1.0)),
        })
    soft_tests_df = pd.DataFrame(soft_tests)
    soft_tests_df.to_csv(OUT / "court_nmf_soft_membership_tests.csv", index=False)

    summary = {
        "labelled_player_seasons": int(len(merged)),
        "dominant_role_chi_square": test["chi_square"],
        "degrees_of_freedom": test["degrees_of_freedom"],
        "p_value": test["p_value"],
        "p_value_method": test["p_value_method"],
        "cramers_v": test["cramers_v"],
        "all_expected_at_least_five": test["all_expected_at_least_five"],
        "min_expected_count": test["min_expected"],
        "tested_table_shape": test["tested_shape"],
        "empty_roles_dropped_from_test": test["empty_rows_dropped"],
        "empty_positions_dropped_from_test": test["empty_columns_dropped"],
        "degeneracy_note": test["note"],
        "position_counts": merged["position"].value_counts().reindex(POSITIONS).to_dict(),
        "dominant_role_counts": merged["dominant_role_name"].value_counts().reindex(list(role_names.values()), fill_value=0).astype(int).to_dict(),
    }
    (OUT / "court_nmf_position_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nSoft-membership Kruskal--Wallis tests:")
    print(soft_tests_df.to_string(index=False))


if __name__ == "__main__":
    main()
