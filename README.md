# COURT-NMF: Player Role Discovery

Code and data for **"COURT-NMF: Contextual, Opportunity-Weighted, Role-Trajectory Factorization for Longitudinal Panels"** (Haoran Li, Capital University of Economics and Business; ICMLC 2027 submission). Contact: lihaoran202309@126.com.

COURT-NMF learns basketball role prototypes and each player-season's mixture over them from 45,980 NCAA Division I player-seasons (2009–10 to 2020–21). It extends non-negative matrix factorization with simplex-constrained mixtures, opportunity weights, conference-season context offsets, and a transfer-aware career graph. The repository also keeps the UMAP + HDBSCAN and NMF + K-means baselines and the validation against external position labels.

## Layout

```
.
├── README.md
├── requirements.txt
├── data/
│   └── CollegeBasketballPlayers2009-2021（55columns）.csv   input panel (public Kaggle table)
└── scripts/                                              all analysis code
```

Each script finds the repository root from its own location. It reads `data/` and creates `outputs/` and `figures/` at the root when it runs. Keep all scripts directly in `scripts/`, because several load `court_nmf_experiment.py` from the same folder.

## Setup

Python 3.11 or later (the R3 runs used Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9).

```bash
pip install -r requirements.txt
```

Run every command from the repository root. `matched_multistart_ablations.py` starts its workers with `fork`, so run it on Linux or macOS (on Windows, use WSL).

## Reproducing the paper

Run the steps in order. Sections, tables and figures refer to the revised manuscript.

| # | Command | Output | In the paper |
|---|---|---|---|
| 1 | Position labels, see [Labels](#labels) | `outputs/expanded_validation/` | Sect. 5 |
| 2 | `python scripts/court_nmf_experiment.py` | `outputs/court_nmf/` | Sects. 3–4 |
| 3 | `python scripts/validate_court_nmf_positions.py`<br>`python scripts/court_nmf_multivariate_validation.py` | `outputs/court_nmf_validation/` | Sect. 5 |
| 4 | `python scripts/matched_multistart_ablations.py --workers 4` | `outputs/matched_ablation/` | Sect. 4.1, Tables 1–2 |
| 5 | `python scripts/matched_multistart_ablations.py --variants gate010,gate060,gate090 --workers 4` | `outputs/matched_ablation/` | App. C.2, Table 7 |
| 6 | `python scripts/make_r3_numbers.py` | `outputs/matched_ablation/r3_generated/` | reported numbers, Tables 1–2, 6–7, Fig. 3 |
| 7 | `python scripts/make_r3_convergence_figure.py --variant full --seed 50` | `figures/fig26_block_convergence.pdf` | Sect. 4.2, Fig. 4 |
| 8 | `python scripts/monte_carlo_convergence.py --mode full --runs 60 --iterations 4000`<br>`python scripts/monte_carlo_convergence.py --mode subsample --runs 8000 --subsample-players 600` | `outputs/monte_carlo/` | Sect. 4.3, App. D.1 |
| 9 | `python scripts/make_monte_carlo_figures.py --mode full` | `figures/fig23_mc_convergence.png`, `fig24_mc_distribution.png` | Figs. 5–6 |
| 10 | `python scripts/compare_step_rules.py` | `outputs/solver_comparison.csv` | App. D.2 |

Step 4 fits the full model, its four ablations and the graph-only baseline from seeds 42–51 and keeps every run. Each variant's reported fit is the start with the lowest value of its own objective. `python scripts/matched_multistart_ablations.py --verify` checks that the instrumented solver reproduces the production solver exactly. Fits are deterministic given a seed: seed 50 gives the selected full model. Steps 4, 5 and 8 resume where they stopped if interrupted. Settings not given above are the script defaults, and each Monte Carlo design records its settings in `outputs/monte_carlo/mc_meta_<mode>.json`.

## Labels

Steps 3–5 and 8 need `outputs/expanded_validation/expanded_validation_labeled_sample.csv`, the 195 labelled player-seasons used for validation. It is not in the repository but is available from the author on request. Of the 195 labels, 176 come from the panel's own position field and 19 from public Sports-Reference player pages. To rebuild a sample by the same procedure:

```bash
python scripts/expanded_external_validation.py --use-dataset-position --dataset-position-col pos
```

Sports-Reference look-ups need internet access. If any players stay unresolved, fill in the `manual_labels_template.csv` the script writes and rerun with `--manual-labels`.

## Other scripts

These scripts cover the exploratory analyses and earlier manuscript versions. You don't need them to reproduce the results above.

| Scripts | Purpose |
|---|---|
| `stage1_rerun_V7.py`, `regen_stage1_figures_V9.py`, `regen_fig03_fig04_V9.py` | Stage 1: PCA, UMAP and HDBSCAN density structure (Sect. 3.1, App. A) |
| `NMF_unified_V7.py`, `stage2_scheme_comparison.py` | Stage 2: direct NMF with K-means, and the PCA-shift comparison |
| `adaptive_clustering_benchmark.py`, `soft_membership_benchmark.py` | soft-clustering and soft-membership benchmarks |
| `run_transfer_gate_sensitivity.py`, `diagnose_convergence.py`, `make_ablation_figure.py`, `make_convergence_figure.py`, `make_r3_story_ablation_figure.py` | earlier single-seed checks and figures |
| `SupLearn(K=5).py`, `Variable_Orthogonization.py`, `scrape_positions_V7.py` | early exploratory profiling and position scraping |
| `build_submission.py`, `qa_submission.py`, `submission_visuals.py` | manuscript assembly for earlier versions |

`diagnose_convergence.py` and `soft_membership_benchmark.py` still contain absolute paths from the machine they were written on. Some older scripts also need `seaborn`, `pypdf` or `Pillow`.

## Data

- Panel: *College Basketball + NBA Advanced Stats*, Kaggle, https://www.kaggle.com/adityak2003/college-basketball-players-20092021 (public; included here as `data/`).
- Supplementary position labels: Sports Reference, https://www.sports-reference.com/cbb/.

No private data are used. Run outputs, block traces and the labelled validation sample are available from the author on request.
