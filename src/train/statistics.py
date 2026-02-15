"""
statistics.py — Statistical significance tests for model comparison
===================================================================
McNemar's test for pairwise architecture comparison.
Paired bootstrap confidence intervals for metric differences.
"""

import json
import numpy as np
from itertools import combinations


def mcnemar_test(preds_a, preds_b, labels):
    """
    McNemar's test comparing two classifiers on the same test set.

    Null hypothesis: Both models have the same error rate.

    Args:
        preds_a: Predictions from model A [N]
        preds_b: Predictions from model B [N]
        labels: True labels [N]

    Returns:
        dict with statistic, p_value, b (A correct & B wrong), c (A wrong & B correct)
    """
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    labels = np.array(labels)

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # b: A correct, B wrong
    b = np.sum(correct_a & ~correct_b)
    # c: A wrong, B correct
    c = np.sum(~correct_a & correct_b)

    # Use exact binomial test for small counts, chi2 for large
    n = b + c

    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": int(b), "c": int(c), "n_discordant": int(n)}

    if n < 25:
        # Exact binomial test
        from scipy.stats import binom_test
        try:
            p_value = binom_test(b, n, 0.5)
        except Exception:
            # Fallback for newer scipy
            from scipy.stats import binomtest
            result = binomtest(b, n, 0.5)
            p_value = result.pvalue
    else:
        # Chi-squared approximation with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(statistic, df=1)

    statistic = (abs(b - c) - 1) ** 2 / max(b + c, 1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "b": int(b),
        "c": int(c),
        "n_discordant": int(n),
    }


def pairwise_mcnemar(results, significance_level=0.05):
    """
    Run McNemar's test for all pairs of architectures.

    Args:
        results: List of result dicts, each containing 'arch', 'test_preds', 'test_labels'
        significance_level: p-value threshold for significance

    Returns:
        List of comparison dicts
    """
    comparisons = []

    # Group results by seed, then compare within each seed
    by_seed = {}
    for r in results:
        seed = r.get("seed", 0)
        if seed not in by_seed:
            by_seed[seed] = []
        by_seed[seed].append(r)

    for seed, seed_results in by_seed.items():
        for r_a, r_b in combinations(seed_results, 2):
            test_result = mcnemar_test(
                r_a["test_preds"], r_b["test_preds"], r_a["test_labels"]
            )
            significant = test_result["p_value"] < significance_level
            comp = {
                "seed": seed,
                "model_a": r_a["arch"],
                "model_b": r_b["arch"],
                "f1_a": r_a["f1_macro"],
                "f1_b": r_b["f1_macro"],
                **test_result,
                "significant": significant,
            }
            comparisons.append(comp)

    return comparisons


def print_mcnemar_table(comparisons):
    """Print McNemar's test results as a formatted table."""
    print("\n" + "=" * 90)
    print("McNEMAR'S TEST — PAIRWISE MODEL COMPARISON")
    print("=" * 90)

    header = (f"{'Seed':>6} | {'Model A':<14} | {'Model B':<14} | "
              f"{'F1_A':>6} | {'F1_B':>6} | {'p-value':>8} | {'Sig?':>5} | {'Discordant':>10}")
    print(header)
    print("-" * len(header))

    for c in comparisons:
        sig_mark = "***" if c["p_value"] < 0.001 else ("**" if c["p_value"] < 0.01 else ("*" if c["significant"] else ""))
        print(f"{c['seed']:>6} | {c['model_a']:<14} | {c['model_b']:<14} | "
              f"{c['f1_a']:>6.4f} | {c['f1_b']:>6.4f} | {c['p_value']:>8.4f} | {sig_mark:>5} | {c['n_discordant']:>10}")

    print("=" * 90)
    print("  * p < 0.05  |  ** p < 0.01  |  *** p < 0.001")
    print()


def run_statistical_tests(results_path, output_path=None):
    """
    Load full results and run all statistical tests.

    Args:
        results_path: Path to full_results.json
        output_path: Path to save test results (optional)
    """
    with open(results_path) as f:
        results = json.load(f)

    comparisons = pairwise_mcnemar(results)
    print_mcnemar_table(comparisons)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(comparisons, f, indent=2)
        print(f"Statistical test results saved to: {output_path}")

    return comparisons


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run statistical significance tests")
    parser.add_argument("--results", type=str, default="data/results/full_results.json",
                        help="Path to full_results.json")
    parser.add_argument("--output", type=str, default="data/results/mcnemar_tests.json",
                        help="Output path for test results")
    args = parser.parse_args()

    run_statistical_tests(args.results, args.output)
