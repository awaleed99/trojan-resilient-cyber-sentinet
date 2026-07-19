"""
evaluation/statistical_tests.py
─────────────────────────────────────────────────────────────────────────────
Statistical significance testing — replicates and extends the original
paper's protocol (Table 13: paired t-test, Wilcoxon, McNemar).

Tests implemented:
  1. Paired t-test       — parametric comparison of two models across folds
  2. Wilcoxon signed-rank — non-parametric alternative to paired t-test
  3. McNemar's test       — comparison of two classifiers' error patterns
                            on the SAME test set (exact test for 2×2 table)
  4. Bootstrap CI         — confidence intervals for ASR differences across defenses

Usage
-----
    from evaluation.statistical_tests import StatisticalTests
    st = StatisticalTests()
    result = st.mcnemar(y_true, preds_defense_A, preds_defense_B)
    result = st.paired_ttest(scores_A, scores_B)
    ci     = st.bootstrap_ci(asr_values_A, asr_values_B)
"""

import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
from scipy import stats
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)


class StatisticalTests:
    """
    Collection of statistical tests matching the paper's protocol.
    All tests use alpha=0.05 by default.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # ── 1. Paired t-test ──────────────────────────────────────────────────────
    def paired_ttest(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        label_a: str = "A",
        label_b: str = "B",
        metric_name: str = "metric",
    ) -> Dict:
        """
        Two-sided paired t-test. Both arrays must have same length.
        Tests H0: mean(A) == mean(B).
        """
        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        significant = p_val < self.alpha

        result = {
            "test":         "paired_ttest",
            "t_statistic":  float(t_stat),
            "p_value":      float(p_val),
            "significant":  bool(significant),
            "mean_a":       float(np.mean(scores_a)),
            "mean_b":       float(np.mean(scores_b)),
            "mean_diff":    float(np.mean(scores_a) - np.mean(scores_b)),
        }
        self._print_result(result, label_a, label_b, metric_name)
        return result

    # ── 2. Wilcoxon signed-rank ───────────────────────────────────────────────
    def wilcoxon(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        label_a: str = "A",
        label_b: str = "B",
        metric_name: str = "metric",
    ) -> Dict:
        """Non-parametric alternative to paired t-test."""
        try:
            stat, p_val = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
        except ValueError as e:
            log.warning(f"Wilcoxon test failed: {e}")
            return {"test": "wilcoxon", "error": str(e)}

        significant = p_val < self.alpha
        result = {
            "test":        "wilcoxon",
            "statistic":   float(stat),
            "p_value":     float(p_val),
            "significant": bool(significant),
            "mean_a":      float(np.mean(scores_a)),
            "mean_b":      float(np.mean(scores_b)),
        }
        self._print_result(result, label_a, label_b, metric_name)
        return result

    # ── 3. McNemar's test ─────────────────────────────────────────────────────
    def mcnemar(
        self,
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        label_a: str = "Model A",
        label_b: str = "Model B",
        exact: bool = True,
    ) -> Dict:
        """
        McNemar's test: compare two classifiers on the SAME test set.
        Tests whether the models make different errors.

        Contingency table:
          b00 = both correct
          b01 = A correct, B wrong
          b10 = A wrong, B correct
          b11 = both wrong

        Statistic: (|b01 - b10| - 1)² / (b01 + b10)  (with continuity correction)
        """
        correct_a = (preds_a == y_true)
        correct_b = (preds_b == y_true)

        b00 = int(( correct_a &  correct_b).sum())
        b01 = int(( correct_a & ~correct_b).sum())  # A right, B wrong
        b10 = int((~correct_a &  correct_b).sum())  # A wrong, B right
        b11 = int((~correct_a & ~correct_b).sum())

        n_discordant = b01 + b10

        if n_discordant == 0:
            p_val = 1.0
            stat  = 0.0
        elif exact:
            # Exact binomial test
            p_val = float(2 * stats.binom.cdf(min(b01, b10), n_discordant, 0.5))
            stat  = float(b10)  # test statistic for exact test
        else:
            # Chi-squared with continuity correction
            stat  = (abs(b01 - b10) - 1) ** 2 / n_discordant
            p_val = float(stats.chi2.sf(stat, df=1))

        significant = p_val < self.alpha
        result = {
            "test":         "mcnemar",
            "b00": b00, "b01": b01, "b10": b10, "b11": b11,
            "n_discordant": n_discordant,
            "statistic":    float(stat),
            "p_value":      float(p_val),
            "significant":  bool(significant),
            "accuracy_a":   float(correct_a.mean()),
            "accuracy_b":   float(correct_b.mean()),
        }

        table = Table(title=f"McNemar's Test: {label_a} vs {label_b}")
        table.add_column("Cell",         style="bold")
        table.add_column("Count",        style="cyan")
        table.add_column("Meaning",      style="white")
        table.add_row("b00", str(b00), "Both correct")
        table.add_row("b01", str(b01), f"{label_a} correct, {label_b} wrong")
        table.add_row("b10", str(b10), f"{label_a} wrong, {label_b} correct")
        table.add_row("b11", str(b11), "Both wrong")
        console.print(table)

        sig_str = "[green]SIGNIFICANT[/green]" if significant else "[red]not significant[/red]"
        console.print(
            f"  McNemar p={p_val:.4f}  α={self.alpha}  → {sig_str}\n"
            f"  Accuracy: {label_a}={result['accuracy_a']:.4f}  {label_b}={result['accuracy_b']:.4f}"
        )
        return result

    # ── 4. Bootstrap CI ───────────────────────────────────────────────────────
    def bootstrap_ci(
        self,
        values_a: np.ndarray,
        values_b: np.ndarray,
        n_bootstrap: int = 10_000,
        ci: float = 0.95,
        label_a: str = "A",
        label_b: str = "B",
        metric_name: str = "ASR",
    ) -> Dict:
        """
        Bootstrap confidence interval for the difference in means (A - B).
        Useful for ASR differences across defenses.
        """
        rng = np.random.RandomState(42)
        diffs = []
        for _ in range(n_bootstrap):
            sample_a = rng.choice(values_a, size=len(values_a), replace=True)
            sample_b = rng.choice(values_b, size=len(values_b), replace=True)
            diffs.append(sample_a.mean() - sample_b.mean())

        diffs = np.array(diffs)
        lower = np.percentile(diffs, 100 * (1 - ci) / 2)
        upper = np.percentile(diffs, 100 * (1 - (1 - ci) / 2))
        obs_diff = float(values_a.mean() - values_b.mean())

        result = {
            "test":        "bootstrap_ci",
            "observed_diff": obs_diff,
            "ci_lower":    float(lower),
            "ci_upper":    float(upper),
            "ci_level":    ci,
            "n_bootstrap": n_bootstrap,
            "significant": not (lower <= 0 <= upper),
        }

        sig_str = (
            "[green]significantly different (CI excludes 0)[/green]"
            if result["significant"]
            else "[red]not significantly different (CI includes 0)[/red]"
        )
        console.print(
            f"\n  Bootstrap CI ({int(100*ci)}%) for Δ{metric_name} ({label_a}−{label_b}):\n"
            f"  Observed diff = {obs_diff:+.4f}  "
            f"95% CI = [{lower:+.4f}, {upper:+.4f}]\n"
            f"  → {sig_str}"
        )
        return result

    # ── ANOVA across multiple defenses ────────────────────────────────────────
    def anova(self, groups: Dict[str, np.ndarray], metric_name: str = "ASR") -> Dict:
        """
        One-way ANOVA across multiple defense groups.
        groups: dict mapping defense_name → array of metric values across runs/folds.
        """
        names   = list(groups.keys())
        arrays  = [groups[n] for n in names]
        f_stat, p_val = stats.f_oneway(*arrays)
        significant = p_val < self.alpha

        console.print(
            f"\n  ANOVA ({metric_name}) across {len(names)} defenses:\n"
            f"  F={f_stat:.4f}  p={p_val:.4f}  "
            + ("[green]SIGNIFICANT[/green]" if significant else "[red]not significant[/red]")
        )
        return {"test": "anova", "f_statistic": float(f_stat), "p_value": float(p_val),
                "significant": bool(significant), "groups": names}

    # ── Utility ───────────────────────────────────────────────────────────────
    def _print_result(self, result: Dict, label_a: str, label_b: str, metric: str):
        sig_str = (
            "[green]SIGNIFICANT[/green]"
            if result["significant"]
            else "[red]not significant[/red]"
        )
        console.print(
            f"\n  {result['test'].replace('_',' ').title()} — {metric}: "
            f"{label_a} vs {label_b}\n"
            f"  mean({label_a})={result['mean_a']:.4f}  "
            f"mean({label_b})={result['mean_b']:.4f}  "
            f"diff={result.get('mean_diff', result['mean_a']-result['mean_b']):+.4f}\n"
            f"  p={result['p_value']:.4f}  α={self.alpha}  → {sig_str}"
        )

    # ── Run full test suite on two defenses ───────────────────────────────────
    def full_comparison(
        self,
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        asr_a: float,
        asr_b: float,
        label_a: str = "Defense A",
        label_b: str = "Defense B",
    ) -> Dict:
        """Run all three tests (t-test, Wilcoxon, McNemar) for two defenses."""
        console.rule(f"[bold]Statistical Comparison: {label_a} vs {label_b}[/bold]")
        results = {}

        # McNemar (classification error patterns on test set)
        results["mcnemar"] = self.mcnemar(y_true, preds_a, preds_b, label_a, label_b)

        # t-test on per-sample accuracy (1=correct, 0=wrong)
        acc_a = (preds_a == y_true).astype(float)
        acc_b = (preds_b == y_true).astype(float)
        results["ttest"]    = self.paired_ttest(acc_a, acc_b, label_a, label_b, "per-sample accuracy")
        results["wilcoxon"] = self.wilcoxon(acc_a, acc_b, label_a, label_b, "per-sample accuracy")

        # Bootstrap CI on ASR difference
        results["bootstrap"] = self.bootstrap_ci(
            np.array([asr_a]), np.array([asr_b]),
            label_a=label_a, label_b=label_b, metric_name="ASR",
        )

        return results
