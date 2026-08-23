"""Data-quality and model-validation checks.

Every check takes a DataFrame and returns a list of violation strings; an
empty list means the check passed. Returning a list rather than raising lets
one function serve two runners:

  - pytest, at development time, asserts the list is empty:
        assert null_violations(events, ["customer_id"]) == []
  - a notebook task, at run time, collects the lists and calls fail_on
    to raise, which fails the task and stops the tasks downstream of it.

The AUC function at the end is pure Python. It is used by the model tests
against the labelled holdout set, where labels exist; the run-time model
checks use only the label-free guards.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Data-quality checks — run on input data
# ---------------------------------------------------------------------------


def missing_columns(df: DataFrame, required: list[str]) -> list[str]:
    """Each required column that is absent from the schema is a violation."""
    present = set(df.columns)
    return [f"missing column: {c}" for c in required if c not in present]


def null_violations(df: DataFrame, columns: list[str]) -> list[str]:
    """Each listed column that contains a null is a violation, with a count."""
    out = []
    for c in columns:
        n = df.filter(F.col(c).isNull()).count()
        if n > 0:
            out.append(f"column {c}: {n} null values")
    return out


def range_violations(
    df: DataFrame, column: str, lo: float | None = None, hi: float | None = None
) -> list[str]:
    """Values below lo or above hi in the column are violations, with a count."""
    out = []
    if lo is not None:
        n = df.filter(F.col(column) < lo).count()
        if n > 0:
            out.append(f"column {column}: {n} values below {lo}")
    if hi is not None:
        n = df.filter(F.col(column) > hi).count()
        if n > 0:
            out.append(f"column {column}: {n} values above {hi}")
    return out


def duplicate_key_violations(df: DataFrame, key: str) -> list[str]:
    """A key value appearing on more than one row is a violation."""
    n = df.groupBy(key).count().filter(F.col("count") > 1).count()
    return [f"column {key}: {n} duplicated values"] if n > 0 else []


def row_count_violations(df: DataFrame, min_rows: int = 1) -> list[str]:
    """Fewer rows than min_rows is a violation.

    The volume expectation: an empty or truncated input passes every
    column-level check, so completeness includes checking that the data
    arrived at all.
    """
    n = df.count()
    if n < min_rows:
        return [f"row count {n} below minimum {min_rows}"]
    return []


# ---------------------------------------------------------------------------
# Model-validation guards — run on scored output, no labels required
# ---------------------------------------------------------------------------


def score_range_violations(df: DataFrame, column: str = "churn_risk") -> list[str]:
    """A churn risk that is NaN, null, or outside [0, 100] is a violation.

    NaN is counted on its own rather than left to the range comparison: Spark
    orders NaN greater than any number, so an upper bound would report it as
    "above 100.0" and a check with only a lower bound would miss it entirely.
    Counting it explicitly makes detection independent of which bounds exist,
    and names the actual fault.
    """
    out = []
    n = df.filter(F.isnan(F.col(column)) | F.col(column).isNull()).count()
    if n > 0:
        out.append(f"column {column}: {n} NaN or null scores")
    out.extend(range_violations(df.filter(~F.isnan(F.col(column))), column, lo=0.0, hi=100.0))
    return out


def band_distribution_violations(
    df: DataFrame, max_high_share: float = 0.5
) -> list[str]:
    """More than max_high_share of customers in the high band is a violation.

    A scoring run that marks most of the customer base high risk is more
    likely to be a data or code fault than a genuine change in the base.
    """
    total = df.count()
    if total == 0:
        return ["scored output is empty"]
    high = df.filter(F.col("risk_band") == "high").count()
    share = high / total
    if share > max_high_share:
        return [f"high band holds {share:.0%} of customers, limit {max_high_share:.0%}"]
    return []


# ---------------------------------------------------------------------------
# Runner for the notebook tasks
# ---------------------------------------------------------------------------


def fail_on(*violation_lists: list[str]) -> None:
    """Raise ValueError listing every violation, or return if there are none."""
    violations = [v for vs in violation_lists for v in vs]
    if violations:
        raise ValueError("checks failed:\n" + "\n".join(f"  - {v}" for v in violations))


# ---------------------------------------------------------------------------
# Offline evaluation — pure Python, used by the test suite
# ---------------------------------------------------------------------------


def auc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve by the rank statistic.

    The probability that a randomly chosen positive outranks a randomly
    chosen negative, counting ties as one half. Pure Python so the model
    test needs no additional dependency.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        raise ValueError("both classes must be present")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))
