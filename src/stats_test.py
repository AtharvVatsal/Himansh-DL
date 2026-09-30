"""
src/stats_test.py

Paired significance tests: proposed model (dual_attention_bigru / full) vs.
every other model and ablation arm, on the subject-independent (LOSO) folds.

Unit of analysis = one LOSO fold. Each fold's score is the mean over seeds
(seeds on the same fold are not independent samples, so they are averaged
rather than treated as extra pairs). For each comparator:
  - Wilcoxon signed-rank test (two-sided, primary) and paired t-test
  - mean difference (proposed - comparator) and rank-biserial effect size
  - Holm correction across comparators within each (dataset, metric)
Tests are run per dataset and pooled over all datasets ("ALL", folds from
every dataset stacked). Note: with 6 folds the smallest possible two-sided
Wilcoxon p-value is 0.031, with 8 folds 0.0078 -- the pooled test is the
better-powered one; say so in the report.

Usage:
    python -m src.stats_test
    python -m src.stats_test --metric macro_f1 accuracy --proposed dual_attention_bigru:full
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, rankdata

from src.utils import load_results, RAW_RESULTS_CSV, RESULTS_DIR, NO_VARIANT

SIGNIFICANCE_CSV = RESULTS_DIR / "tables" / "significance.csv"


def label(model: str, variant: str) -> str:
    return model if variant == NO_VARIANT else f"{model}:{variant}"


def fold_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """rows = (dataset, fold_id), columns = model[:variant], values = seed-mean metric."""
    d = df[df.split_mode == "loso"].copy()
    d["arm"] = [label(m, v) for m, v in zip(d.model, d.variant)]
    return d.pivot_table(index=["dataset", "fold_id"], columns="arm", values=metric, aggfunc="mean")


def rank_biserial(diff: np.ndarray) -> float:
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    w_pos, w_neg = ranks[diff > 0].sum(), ranks[diff < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    out = dict(n_folds=len(diff), mean_proposed=a.mean(), mean_comparator=b.mean(),
               mean_diff=diff.mean(), std_diff=diff.std(ddof=1) if len(diff) > 1 else np.nan,
               proposed_wins=int((diff > 0).sum()), ties=int((diff == 0).sum()),
               rank_biserial=rank_biserial(diff))
    if len(diff) < 2 or np.all(diff == 0):
        out.update(wilcoxon_p=1.0, ttest_p=1.0)
        return out
    out["wilcoxon_p"] = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    out["ttest_p"] = float(ttest_rel(a, b).pvalue)
    return out


def holm(pvals: pd.Series) -> pd.Series:
    order = pvals.sort_values().index
    m = len(pvals)
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[idx]))
        adj[idx] = running
    return pd.Series(adj).reindex(pvals.index)


def run_tests(df: pd.DataFrame, proposed: str, metrics) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        scores = fold_scores(df, metric)
        if proposed not in scores.columns:
            print(f"[stats_test] no LOSO results for {proposed}; nothing to test")
            return pd.DataFrame()
        comparators = [c for c in scores.columns if c != proposed]
        groups = [(ds, scores.xs(ds, level="dataset")) for ds in scores.index.get_level_values(0).unique()]
        groups.append(("ALL", scores))
        for ds, s in groups:
            block = []
            for comp in comparators:
                pair = s[[proposed, comp]].dropna()
                if len(pair) == 0:
                    continue
                block.append(dict(dataset=ds, metric=metric, proposed=proposed, comparator=comp,
                                  **paired_test(pair[proposed].to_numpy(), pair[comp].to_numpy())))
            if block:
                b = pd.DataFrame(block)
                b["wilcoxon_p_holm"] = holm(b["wilcoxon_p"]).to_numpy()
                rows.append(b)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_csv", default=str(RAW_RESULTS_CSV))
    p.add_argument("--proposed", default="dual_attention_bigru:full")
    p.add_argument("--metric", nargs="+", default=["macro_f1", "accuracy"])
    p.add_argument("--out", default=str(SIGNIFICANCE_CSV))
    args = p.parse_args()

    res = run_tests(load_results(args.results_csv), args.proposed, args.metric)
    if res.empty:
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(res[["dataset", "metric", "comparator", "n_folds", "mean_diff", "proposed_wins",
                   "rank_biserial", "wilcoxon_p", "wilcoxon_p_holm", "ttest_p"]].round(4).to_string(index=False))
    print(f"wrote {len(res)} rows -> {args.out}")


if __name__ == "__main__":
    main()
