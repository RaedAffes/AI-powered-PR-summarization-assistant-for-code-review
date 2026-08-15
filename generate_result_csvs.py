"""
generate_result_csvs.py
========================
Turns the .pkl result files produced by ACR_vLLM.py / CTR_vLLM.py / CL_vLLM.py /
SI_vLLM.py (from the CodeReviewQA repo) into CSV tables that mirror the tables
published in the paper (Tables 8-13 style: models x languages + Overall,
and Table 2 style: models x tasks).

Expected input layout (unchanged from the repo's own save_dir conventions):

    results/acr/<lang>/acr_<lang>_<model>.pkl
    results/ctr/<lang>/ctr_<lang>_<model>.pkl
    results/cl/<lang>/easy/cl_easy_<lang>_<model>.pkl
    results/cl/<lang>/hard/cl_hard_<lang>_<model>.pkl
    results/si/<lang>/easy/si_easy_<lang>_<model>.pkl
    results/si/<lang>/hard/si_hard_<lang>_<model>.pkl

<lang> in {c, cpp, csharp, go, java, javascript, php, python, ruby}

Output (written to --outdir, default "results/csv"):

  Per-task tables (rows = model, columns = 9 languages + Overall),
  matching Tables 8-13 in the paper:
    acr_scores.csv
    ctr_scores.csv
    cl_easy_scores.csv
    cl_hard_scores.csv
    si_easy_scores.csv
    si_hard_scores.csv

  Per-task "Overall only" tables (rows = model, one Overall column):
    acr_overall.csv
    ctr_overall.csv
    cl_easy_overall.csv
    cl_hard_overall.csv
    si_easy_overall.csv
    si_hard_overall.csv

  One global summary (rows = model, columns = ACR / CTR / CL_Easy / CL_Hard /
  SI_Easy / SI_Hard), matching the shape of Table 2:
    global_summary.csv

Usage:
    python generate_result_csvs.py --results-dir results --outdir results/csv
    python generate_result_csvs.py --acr-metric em_trim   # change ACR metric

Notes on scoring (taken directly from the repo's own code, not re-derived):
  - ACR: for each example, `em` = 1 iff the whitespace-tokenised generation
    exactly equals the whitespace-tokenised gold revision (utils.get_em).
    Score = 100 * mean(em) over the 100 examples for that language.
    --acr-metric lets you switch to em_trim / em_no_space / em_no_comment
    if you want one of the other three variants utils.myeval() computes.
  - CTR / CL / SI: each example was run through every N! ordering of its
    answer options (N=3 for CTR, N=4 for CL/SI). "Invariant accuracy" (what
    the paper tables report) counts an example correct only if the model
    picked the correct option symbol in *every* permutation:
        correct  <=>  row.model_answers == row.correct_answers
    Score = 100 * mean(correct) over the 100 examples for that language.
  - "Overall" per task = pooled score across all languages available for
    that model (equivalent to the mean of the per-language scores here,
    since each language contributes exactly 100 examples).
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

LANGS = ["c", "cpp", "csharp", "go", "java", "javascript", "php", "python", "ruby"]
LANG_DISPLAY = {
    "c": "C", "cpp": "C++", "csharp": "CSharp", "go": "Go", "java": "Java",
    "javascript": "JavaScript", "php": "PHP", "python": "Python", "ruby": "Ruby",
}
MODES = ["easy", "hard"]


# --------------------------------------------------------------------------
# Filename -> model name parsing
# --------------------------------------------------------------------------
def _model_name_from_filename(filename: str, prefix: str) -> str:
    """Strip the known task/lang/mode prefix and the .pkl suffix to recover
    the model_name_short the original scripts embedded in the filename."""
    stem = filename[:-4] if filename.endswith(".pkl") else filename
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected filename '{filename}' for prefix '{prefix}'")
    return stem[len(prefix):]


# --------------------------------------------------------------------------
# Per-task scoring functions (mirrors utils.py / the *_vLLM.py scripts)
# --------------------------------------------------------------------------
def _score_acr_file(path: Path, metric: str) -> tuple[float, int]:
    df = pd.read_pickle(path)
    if metric not in df.columns:
        raise ValueError(f"'{metric}' not found in {path} (columns: {list(df.columns)})")
    n = len(df)
    score = 100.0 * df[metric].sum() / n if n else float("nan")
    return score, n


def _score_mcqa_file(path: Path) -> tuple[float, int]:
    """Shared by CTR / CL / SI: invariant accuracy = fraction of examples
    where the model got every permutation's answer right."""
    df = pd.read_pickle(path)
    n = len(df)
    if n == 0:
        return float("nan"), 0
    correct = (df["model_answers"] == df["correct_answers"]).sum()
    return 100.0 * correct / n, n


# --------------------------------------------------------------------------
# Collectors: walk the results dir for one task and build a long table of
# (model, language, score, n_examples), then pivot it.
# --------------------------------------------------------------------------
def collect_acr(results_dir: Path, metric: str) -> pd.DataFrame:
    rows = []
    for lang in LANGS:
        lang_dir = results_dir / "acr" / lang
        if not lang_dir.is_dir():
            continue
        prefix = f"acr_{lang}_"
        for f in lang_dir.glob("*.pkl"):
            model = _model_name_from_filename(f.name, prefix)
            score, n = _score_acr_file(f, metric)
            rows.append({"model": model, "lang": lang, "score": score, "n": n})
    return pd.DataFrame(rows)


def collect_ctr(results_dir: Path) -> pd.DataFrame:
    rows = []
    for lang in LANGS:
        lang_dir = results_dir / "ctr" / lang
        if not lang_dir.is_dir():
            continue
        prefix = f"ctr_{lang}_"
        for f in lang_dir.glob("*.pkl"):
            model = _model_name_from_filename(f.name, prefix)
            score, n = _score_mcqa_file(f)
            rows.append({"model": model, "lang": lang, "score": score, "n": n})
    return pd.DataFrame(rows)


def collect_cl_or_si(results_dir: Path, task: str, mode: str) -> pd.DataFrame:
    """task = 'cl' or 'si', mode = 'easy' or 'hard'."""
    rows = []
    for lang in LANGS:
        mode_dir = results_dir / task / lang / mode
        if not mode_dir.is_dir():
            continue
        prefix = f"{task}_{mode}_{lang}_"
        for f in mode_dir.glob("*.pkl"):
            model = _model_name_from_filename(f.name, prefix)
            score, n = _score_mcqa_file(f)
            rows.append({"model": model, "lang": lang, "score": score, "n": n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pivoting: long (model, lang, score) -> wide table matching the paper
# --------------------------------------------------------------------------
def pivot_with_overall(long_df: pd.DataFrame) -> pd.DataFrame:
    """rows = model, columns = language display names + 'Overall'.
    'Overall' is the pooled score across whichever languages are present
    for that model (weighted by n, so it's correct even if a language is
    missing or has fewer than 100 rows)."""
    if long_df.empty:
        return pd.DataFrame(columns=["Model"] + [LANG_DISPLAY[l] for l in LANGS] + ["Overall"])

    wide = long_df.pivot_table(index="model", columns="lang", values="score", aggfunc="first")
    wide = wide.reindex(columns=LANGS)
    wide = wide.rename(columns=LANG_DISPLAY)

    # Pooled overall = sum(score_i * n_i) / sum(n_i) per model, i.e. re-derive
    # correct counts from (score, n) rather than a naive mean of percentages.
    counts = long_df.assign(correct=lambda d: d["score"] / 100.0 * d["n"])
    totals = counts.groupby("model").agg(correct=("correct", "sum"), n=("n", "sum"))
    overall = (100.0 * totals["correct"] / totals["n"]).rename("Overall")

    wide = wide.join(overall)
    wide.index.name = "Model"
    wide = wide.reset_index().sort_values("Model")
    return wide


def overall_only(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame(columns=["Model", "Overall"])
    return wide_df[["Model", "Overall"]].copy()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results", help="Root of the repo's results/ folder (default: results)")
    parser.add_argument("--outdir", default="results/csv", help="Where to write the CSVs (default: results/csv)")
    parser.add_argument(
        "--acr-metric",
        default="em",
        choices=["em", "em_trim", "em_no_space", "em_no_comment"],
        help="Which of utils.myeval()'s four ACR match variants to report (default: em)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    task_tables = {}

    # ACR
    acr_long = collect_acr(results_dir, args.acr_metric)
    acr_wide = pivot_with_overall(acr_long)
    task_tables["ACR"] = acr_wide

    # CTR
    ctr_long = collect_ctr(results_dir)
    ctr_wide = pivot_with_overall(ctr_long)
    task_tables["CTR"] = ctr_wide

    # CL / SI, easy + hard
    for task in ["cl", "si"]:
        for mode in MODES:
            long_df = collect_cl_or_si(results_dir, task, mode)
            wide_df = pivot_with_overall(long_df)
            label = f"{task.upper()}_{mode.capitalize()}"
            task_tables[label] = wide_df

    # -- write per-task language tables + per-task overall-only tables --
    filename_map = {
        "ACR": "acr", "CTR": "ctr",
        "CL_Easy": "cl_easy", "CL_Hard": "cl_hard",
        "SI_Easy": "si_easy", "SI_Hard": "si_hard",
    }
    for label, wide_df in task_tables.items():
        fname = filename_map[label]
        scores_path = outdir / f"{fname}_scores.csv"
        overall_path = outdir / f"{fname}_overall.csv"
        wide_df.to_csv(scores_path, index=False)
        overall_only(wide_df).to_csv(overall_path, index=False)
        print(f"[{label}] wrote {scores_path} ({len(wide_df)} models) and {overall_path}")

    # -- global summary: rows = model, columns = task Overall scores --
    global_df = None
    for label, wide_df in task_tables.items():
        if wide_df.empty:
            continue
        col = wide_df[["Model", "Overall"]].rename(columns={"Overall": label})
        global_df = col if global_df is None else global_df.merge(col, on="Model", how="outer")

    if global_df is None:
        global_df = pd.DataFrame(columns=["Model", "ACR", "CTR", "CL_Easy", "CL_Hard", "SI_Easy", "SI_Hard"])
    else:
        for col in ["ACR", "CTR", "CL_Easy", "CL_Hard", "SI_Easy", "SI_Hard"]:
            if col not in global_df.columns:
                global_df[col] = float("nan")
        global_df = global_df[["Model", "ACR", "CTR", "CL_Easy", "CL_Hard", "SI_Easy", "SI_Hard"]]
        global_df = global_df.sort_values("Model")

    global_path = outdir / "global_summary.csv"
    global_df.to_csv(global_path, index=False)
    print(f"[GLOBAL] wrote {global_path} ({len(global_df)} models)")


if __name__ == "__main__":
    main()
