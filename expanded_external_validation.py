"""
Expanded external validation for the NMF + K-means player clusters.

This script builds a 200--250 player validation sample, retrieves traditional
position labels, and runs a chi-square test of independence for:

    cluster label (0--4) x traditional position (guard/forward/center)

The default path recomputes the Stage-2 model before sampling. Alternatively,
provide a CSV containing pid, cluster, and silhouette_score with --labeled-input.

Typical use from the submission(V10) directory:

    python scripts/expanded_external_validation.py --players-per-cluster 40

If automated Sports-Reference scraping leaves unresolved players, fill the
generated manual_labels_template.csv and rerun with:

    python scripts/expanded_external_validation.py --manual-labels outputs/expanded_validation/manual_labels_template.csv
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BASE_DIR / "data" / "CollegeBasketballPlayers2009-2021（55columns）.csv"
DEFAULT_OUTDIR = BASE_DIR / "outputs" / "expanded_validation"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Map K-means labels to the reported archetype order.
RELABEL = {0: 0, 4: 1, 2: 2, 1: 3, 3: 4}
POSITION_ORDER = ["guard", "forward", "center"]


def ensure_scraping_dependencies():
    """Import scraper-only dependencies with a clear message if missing."""
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Automated scraping requires the 'requests' and 'beautifulsoup4' packages. "
            "Install them, or rerun with --no-scrape plus --manual-labels or "
            "--use-dataset-position."
        ) from exc
    return requests, BeautifulSoup


def log(message: str) -> None:
    """Print progress immediately so long scraping runs are transparent."""
    print(message, flush=True)


def normalize_slug(value: str) -> str:
    """Convert a player name to Sports-Reference CBB slug stem (without -N suffix)."""
    # Normalize Unicode and remove accents
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.lower()

    # 1. Remove all punctuation EXCEPT spaces (and maybe hyphens)
    #    We want to remove dots, apostrophes, commas, etc.
    value = re.sub(r"[^\w\s]", "", value)   # keep only letters, digits, underscores, spaces

    # 2. Replace underscores with spaces (just in case)
    value = value.replace("_", " ")

    # 3. Split into words
    parts = value.split()
    if not parts:
        return ""

    # 4. Handle common suffixes (Jr., Sr., II, III, IV) – attach to previous word
    suffix_set = {"jr", "sr", "ii", "iii", "iv"}
    if len(parts) >= 2 and parts[-1] in suffix_set:
        parts[-2] = parts[-2] + parts[-1]   # e.g., "watson" + "jr" -> "watsonjr"
        parts = parts[:-1]

    # 5. Join with hyphens
    slug = "-".join(parts)

    # 6. Remove multiple consecutive hyphens and trim
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


def normalize_position(raw_value: object) -> str:
    """Map Sports-Reference or dataset position strings to guard/forward/center."""
    if raw_value is None or pd.isna(raw_value):
        return "Unknown"

    text = str(raw_value).strip().lower()
    if not text:
        return "Unknown"

    # Preserve slash/combo labels by assigning the broad role that appears first.
    tokens = re.split(r"[^a-z]+", text)
    tokens = [token for token in tokens if token]

    guard_words = {"g", "pg", "sg", "guard", "combo"}
    forward_words = {"f", "sf", "pf", "forward", "wing"}
    center_words = {"c", "center", "centre"}

    for token in tokens:
        if token in guard_words:
            return "guard"
        if token in forward_words:
            return "forward"
        if token in center_words:
            return "center"

    if "guard" in text:
        return "guard"
    if "forward" in text:
        return "forward"
    if "center" in text or "centre" in text:
        return "center"
    return "Unknown"


def possible_sports_ref_ids(row: pd.Series, id_col: str | None = None) -> list[str]:
    """
    Build candidate Sports-Reference player ids.

    Sports-Reference CBB usually uses name slugs such as brandon-davies-1.
    Some local datasets may already include a slug column; if so, pass it with
    --sports-ref-id-col and it will be tried before name-derived candidates.
    """
    candidates: list[str] = []
    if id_col and id_col in row and pd.notna(row[id_col]):
        raw = str(row[id_col]).strip()
        if raw:
            raw = raw.removesuffix(".html").split("/")[-1]
            candidates.append(raw)

    stem = normalize_slug(row.get("player_name", ""))
    if stem:
        candidates.extend([f"{stem}-{i}" for i in range(1, 6)])
        candidates.append(stem)

    # Keep order while removing duplicates.
    return list(dict.fromkeys(candidates))


def extract_position_from_html(html: str) -> str:
    """Extract and normalize a position from a Sports-Reference CBB page."""
    _, BeautifulSoup = ensure_scraping_dependencies()
    soup = BeautifulSoup(html, "html.parser")

    meta = soup.find(id="meta")
    if meta:
        meta_text = " ".join(meta.get_text(" ", strip=True).split())
        match = re.search(r"Position:\s*([A-Za-z /\-]+)", meta_text)
        if match:
            pos = normalize_position(match.group(1))
            if pos != "Unknown":
                return pos

    # Some pages expose per-season abbreviations in standard stat tables.
    for cell in soup.select('[data-stat="pos"]'):
        pos = normalize_position(cell.get_text(" ", strip=True))
        if pos != "Unknown":
            return pos

    return "Unknown"


def fetch_sports_ref_position(
    row: pd.Series,
    session,
    cache: dict[str, dict[str, str]],
    id_col: str | None,
    sleep_seconds: float,
) -> tuple[str, str, str]:
    """
    Return (position, matched_id, status) for one player.

    The cache is keyed by candidate id so interrupted runs can be resumed
    without refetching successful pages.
    """
    for candidate in possible_sports_ref_ids(row, id_col):
        if candidate in cache:
            cached = cache[candidate]
            if cached.get("status") == "ok":
                return cached["position"], candidate, "ok_cached"
            continue

        url = f"https://www.sports-reference.com/cbb/players/{candidate}.html"
        try:
            response = session.get(url, timeout=30)
        except Exception as exc:
            cache[candidate] = {"position": "Unknown", "status": f"request_error:{exc}"}
            time.sleep(sleep_seconds)
            continue

        if response.status_code == 404:
            cache[candidate] = {"position": "Unknown", "status": "not_found"}
            time.sleep(sleep_seconds)
            continue
        if response.status_code == 429:
            cache[candidate] = {"position": "Unknown", "status": "rate_limited"}
            return "Unknown", candidate, "rate_limited"
        if response.status_code >= 400:
            cache[candidate] = {"position": "Unknown", "status": f"http_{response.status_code}"}
            time.sleep(sleep_seconds)
            continue

        position = extract_position_from_html(response.text)
        status = "ok" if position != "Unknown" else "missing_position"
        cache[candidate] = {"position": position, "status": status}
        time.sleep(sleep_seconds)
        if position != "Unknown":
            return position, candidate, status

    return "Unknown", "", "not_found"


def load_manual_labels(path: Path | None) -> dict[str, str]:
    """Load optional manual labels keyed by pid or player_name."""
    if path is None or not path.exists():
        return {}

    manual = pd.read_csv(path)
    if "position" not in manual.columns:
        raise ValueError(f"{path} must contain a 'position' column.")

    label_map: dict[str, str] = {}
    for _, row in manual.iterrows():
        position = normalize_position(row.get("position"))
        if position == "Unknown":
            continue
        for key_col in ("pid", "player_name"):
            if key_col in row and pd.notna(row[key_col]):
                label_map[str(row[key_col]).strip()] = position
    return label_map


def build_stage2_labeled_frame(data_path: Path) -> pd.DataFrame:
    """
    Recompute the direct-NMF + K-means solution used for validation.

    Returns one row per analytic observation with pid, player metadata, cluster,
    and silhouette_score.
    """
    df0 = pd.read_csv(data_path, low_memory=False)
    log(f"Raw observations: {len(df0):,}")
    log(f"Unique player ids: {df0['pid'].nunique():,}")

    rotation_mask = (
        (pd.to_numeric(df0["Min_per"], errors="coerce") > 5)
        & (pd.to_numeric(df0["GP"], errors="coerce") >= 10)
        & (pd.to_numeric(df0["pts"], errors="coerce") >= 1.0)
    )

    df = df0.drop(["type", "ht"], axis=1, errors="ignore")
    mice_cols = [
        "rimmade",
        "rimmade+rimmiss",
        "midmade",
        "midmade+midmiss",
        "rimmade/(rimmade+rimmiss)",
        "midmade/(midmade+midmiss)",
        "dunksmade",
        "dunksmiss+dunksmade",
    ]
    mice_cols = [col for col in mice_cols if col in df.columns]

    numeric_cols = [
        "GP",
        "Min_per",
        "Ortg",
        "usg",
        "eFG",
        "TS_per",
        "ORB_per",
        "DRB_per",
        "AST_per",
        "TO_per",
        "FTM",
        "FTA",
        "FT_per",
        "twoPM",
        "twoPA",
        "twoP_per",
        "TPM",
        "TPA",
        "TP_per",
        "blk_per",
        "stl_per",
        "ftr",
        "porpag",
        "adjoe",
        "pfr",
        "ast/tov",
        "drtg",
        "adrtg",
        "dporpag",
        "stops",
        "bpm",
        "obpm",
        "dbpm",
        "gbpm",
        "mp",
        "ogbpm",
        "dgbpm",
        "oreb",
        "dreb",
        "treb",
        "ast",
        "stl",
        "blk",
        "pts",
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns and col not in mice_cols]
    categorical_cols = [col for col in ["player_name", "team", "conf", "yr"] if col in df.columns]

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for col in numeric.columns:
        numeric[col] = numeric[col].fillna(numeric[col].median())

    categorical = df[categorical_cols].copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical.columns:
        mode = categorical[col].mode()
        categorical[col] = categorical[col].fillna(mode[0] if not mode.empty else "Unknown")
        encoder = LabelEncoder()
        categorical[col] = encoder.fit_transform(categorical[col].astype(str))
        encoders[col] = encoder

    log("Running MICE imputation for shot-location columns...")
    mice = pd.DataFrame(
        IterativeImputer(max_iter=25, random_state=42).fit_transform(
            df[mice_cols].apply(pd.to_numeric, errors="coerce")
        ),
        columns=mice_cols,
        index=df.index,
    )
    for col in ["rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)"]:
        if col in mice.columns:
            mice[col] = mice[col].clip(0, 1)

    for col in categorical.columns:
        categorical[col] = encoders[col].inverse_transform(categorical[col].astype(int))

    other = df.drop(columns=mice_cols + numeric_cols + categorical_cols + ["pid"], errors="ignore")
    df_model = pd.concat([mice, numeric, categorical, other.reset_index(drop=True)], axis=1)
    df_model.insert(0, "pid", df0["pid"].values)
    df_model = df_model[rotation_mask.values].copy()
    log(f"Analytic observations after rotation filter: {len(df_model):,}")

    for old_col, new_col in {
        "pts": "pts_per100",
        "treb": "treb_per100",
        "ast": "ast_per100",
        "stl": "stl_per100",
        "blk": "blk_per100",
        "oreb": "oreb_per100",
        "dreb": "dreb_per100",
    }.items():
        if old_col in df_model.columns:
            df_model[new_col] = df_model[old_col] / df_model["Min_per"] * 100
            numeric_cols.append(new_col)

    for col in [
        "Min_per",
        "ORB_per",
        "DRB_per",
        "AST_per",
        "TO_per",
        "FT_per",
        "twoP_per",
        "TP_per",
        "blk_per",
        "stl_per",
        "usg",
        "eFG",
        "TS_per",
        "GP",
        "FTM",
        "FTA",
        "twoPM",
        "twoPA",
        "TPM",
        "TPA",
        "ftr",
        "porpag",
        "adjoe",
        "pfr",
        "ast/tov",
        "drtg",
        "adrtg",
        "dporpag",
        "stops",
        "bpm",
        "obpm",
        "dbpm",
        "gbpm",
        "mp",
        "ogbpm",
        "dgbpm",
        "Ortg",
    ]:
        if col in df_model.columns:
            numeric_cols.append(col)
    numeric_cols = sorted({col for col in numeric_cols if col in df_model.columns})

    scaled = df_model[numeric_cols].dropna().copy()
    scaled[:] = RobustScaler().fit_transform(scaled)

    nmf_cols = [
        "pts_per100",
        "treb_per100",
        "ast_per100",
        "stl_per100",
        "blk_per100",
        "oreb_per100",
        "dreb_per100",
        "usg",
        "eFG",
        "TS_per",
        "ORB_per",
        "DRB_per",
        "AST_per",
        "TO_per",
        "FT_per",
        "twoP_per",
        "TP_per",
        "blk_per",
        "stl_per",
        "ftr",
        "bpm",
        "obpm",
        "dbpm",
    ]
    nmf_cols = [col for col in nmf_cols if col in scaled.columns]
    nmf_input = MinMaxScaler().fit_transform(scaled[nmf_cols])

    log("Fitting NMF(n_components=5) and KMeans(n_clusters=5)...")
    nmf_weights = NMF(
        n_components=5,
        random_state=42,
        init="nndsvd",
        max_iter=500,
    ).fit_transform(nmf_input)
    raw_labels = KMeans(n_clusters=5, random_state=42, n_init=25).fit_predict(nmf_weights)
    labels = np.array([RELABEL[int(label)] for label in raw_labels])

    log("Computing K=5 silhouette scores...")
    silhouettes = silhouette_samples(nmf_weights, raw_labels)

    output = df_model.loc[scaled.index].copy()
    output["cluster"] = labels
    output["silhouette_score"] = silhouettes
    return output


def sample_validation_players(
    df: pd.DataFrame,
    players_per_cluster: int,
    method: str,
    random_state: int,
) -> pd.DataFrame:
    """Select unique players within each cluster using top silhouettes or random sampling."""
    if not 40 <= players_per_cluster <= 50:
        raise ValueError("--players-per-cluster must be between 40 and 50.")

    required = {"pid", "player_name", "cluster", "silhouette_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    # Keep each player's most representative observation before cluster sampling.
    unique_players = (
        df.sort_values("silhouette_score", ascending=False)
        .drop_duplicates(subset=["pid"], keep="first")
        .copy()
    )

    sampled_frames: list[pd.DataFrame] = []
    for cluster_id in range(5):
        pool = unique_players[unique_players["cluster"] == cluster_id].copy()
        if len(pool) < players_per_cluster:
            raise ValueError(
                f"Cluster {cluster_id} has only {len(pool)} unique players; "
                f"cannot sample {players_per_cluster}."
            )

        if method == "top":
            selected = pool.nlargest(players_per_cluster, "silhouette_score")
        else:
            selected = pool.sample(n=players_per_cluster, random_state=random_state + cluster_id)

        sampled_frames.append(selected)

    sample = pd.concat(sampled_frames, ignore_index=True)
    sample = sample.sort_values(["cluster", "silhouette_score"], ascending=[True, False])
    if sample["pid"].duplicated().any():
        raise AssertionError("Validation sample contains duplicate pid values.")
    return sample


def apply_position_labels(
    sample: pd.DataFrame,
    args: argparse.Namespace,
    outdir: Path,
) -> pd.DataFrame:
    """Attach position labels from manual CSV, Sports-Reference, or optional dataset fallback."""
    sample = sample.copy()
    sample["position"] = "Unknown"
    sample["position_source"] = ""
    sample["sports_ref_id"] = ""
    sample["scrape_status"] = ""

    manual = load_manual_labels(args.manual_labels)
    for idx, row in sample.iterrows():
        for key in (str(row.get("pid", "")).strip(), str(row.get("player_name", "")).strip()):
            if key in manual:
                sample.at[idx, "position"] = manual[key]
                sample.at[idx, "position_source"] = "manual"
                break

    if args.use_dataset_position and args.dataset_position_col in sample.columns:
        unknown_mask = sample["position"].eq("Unknown")
        sample.loc[unknown_mask, "position"] = sample.loc[unknown_mask, args.dataset_position_col].map(
            normalize_position
        )
        sample.loc[
            unknown_mask & sample["position"].ne("Unknown"),
            "position_source",
        ] = f"dataset:{args.dataset_position_col}"

    if args.no_scrape:
        return sample

    requests, _ = ensure_scraping_dependencies()
    cache_path = outdir / "sports_ref_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    unresolved = sample.index[sample["position"].eq("Unknown")].tolist()
    for count, idx in enumerate(unresolved, start=1):
        row = sample.loc[idx]
        position, matched_id, status = fetch_sports_ref_position(
            row=row,
            session=session,
            cache=cache,
            id_col=args.sports_ref_id_col,
            sleep_seconds=args.sleep_seconds,
        )
        sample.at[idx, "position"] = position
        sample.at[idx, "position_source"] = "sports-reference" if position != "Unknown" else ""
        sample.at[idx, "sports_ref_id"] = matched_id
        sample.at[idx, "scrape_status"] = status
        if count % 10 == 0 or position == "Unknown":
            log(
                f"Scraped {count}/{len(unresolved)} unresolved players; "
                f"latest={row.get('player_name')} position={position} status={status}"
            )
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    return sample


def build_manual_template(labeled: pd.DataFrame, outdir: Path) -> None:
    """Write a fill-in CSV for players whose automated position lookup failed."""
    unresolved = labeled[labeled["position"].eq("Unknown")].copy()
    cols = [
        "pid",
        "player_name",
        "team",
        "yr",
        "cluster",
        "silhouette_score",
        "sports_ref_id",
        "scrape_status",
        "position",
    ]
    cols = [col for col in cols if col in unresolved.columns]
    unresolved[cols].to_csv(outdir / "manual_labels_template.csv", index=False)


def run_chi_square(labeled: pd.DataFrame, outdir: Path, alpha: float) -> None:
    """Build observed/expected contingency tables and print the test conclusion."""
    known = labeled[labeled["position"].isin(POSITION_ORDER)].copy()
    observed = pd.crosstab(known["cluster"], known["position"])
    observed = observed.reindex(index=range(5), columns=POSITION_ORDER, fill_value=0)
    observed.to_csv(outdir / "expanded_validation_crosstab.csv")

    if observed.values.sum() == 0 or (observed.sum(axis=1) == 0).any() or (observed.sum(axis=0) == 0).any():
        warning = (
            "Chi-square test was not run because at least one cluster row or "
            "position column has no known labels."
        )
        log(f"WARNING: {warning}")
        (outdir / "chi_square_summary.txt").write_text(warning + "\n", encoding="utf-8")
        return

    chi2, p_value, dof, expected = chi2_contingency(observed)
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    expected_df.to_csv(outdir / "expanded_validation_expected_frequencies.csv")

    low_expected = expected_df < 5
    conclusion = (
        "significant association"
        if p_value < alpha
        else "no statistically significant association"
    )

    lines = [
        "Observed contingency table:",
        observed.to_string(),
        "",
        "Expected frequencies:",
        expected_df.round(3).to_string(),
        "",
        f"Chi-square = {chi2:.6f}",
        f"Degrees of freedom = {dof}",
        f"p-value = {p_value:.6g}",
        f"alpha = {alpha}",
        f"Conclusion: {conclusion} between cluster and position at alpha = {alpha}.",
        f"Known labels used: {len(known)} of {len(labeled)} sampled players.",
        f"Unknown labels: {int(labeled['position'].eq('Unknown').sum())}",
    ]

    if low_expected.any().any():
        lines.extend(
            [
                "",
                "WARNING: At least one expected frequency is below 5.",
                "Use Fisher's exact test for small tables where applicable, or a Monte Carlo/permutation chi-square test.",
            ]
        )

    summary = "\n".join(lines) + "\n"
    log(summary)
    (outdir / "chi_square_summary.txt").write_text(summary, encoding="utf-8")

    json_summary = {
        "chi_square": float(chi2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
        "known_labels": int(len(known)),
        "sample_size": int(len(labeled)),
        "unknown_labels": int(labeled["position"].eq("Unknown").sum()),
        "any_expected_below_5": bool(low_expected.any().any()),
    }
    (outdir / "chi_square_summary.json").write_text(
        json.dumps(json_summary, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--labeled-input",
        type=Path,
        help="Optional CSV that already contains pid, cluster, and silhouette_score.",
    )
    parser.add_argument("--players-per-cluster", type=int, default=40)
    parser.add_argument("--method", choices=["top", "random"], default="top")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--sports-ref-id-col",
        default=None,
        help="Optional column containing Sports-Reference slug/id values.",
    )
    parser.add_argument("--manual-labels", type=Path)
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Do not contact Sports-Reference; use manual/dataset labels only.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Delay between Sports-Reference requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--use-dataset-position",
        action="store_true",
        help="Use a position column already present in the dataset as a fallback label source.",
    )
    parser.add_argument(
        "--dataset-position-col",
        default="Unnamed: 64",
        help="Dataset column to use when --use-dataset-position is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.labeled_input:
        frame = pd.read_csv(args.labeled_input, low_memory=False)
    else:
        frame = build_stage2_labeled_frame(args.data)
        frame.to_csv(args.outdir / "expanded_validation_all_labeled_observations.csv", index=False)

    sample = sample_validation_players(
        frame,
        players_per_cluster=args.players_per_cluster,
        method=args.method,
        random_state=args.random_state,
    )
    sample.to_csv(args.outdir / "expanded_validation_sample.csv", index=False)
    log(f"Validation sample written: {len(sample)} players")

    labeled = apply_position_labels(sample, args, args.outdir)
    labeled.to_csv(args.outdir / "expanded_validation_labeled_sample.csv", index=False)
    build_manual_template(labeled, args.outdir)
    run_chi_square(labeled, args.outdir, args.alpha)

    log(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
