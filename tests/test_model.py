"""Model-validation tests.

The metric-threshold test runs here, at development time, against the
labelled holdout set in tests/fixtures/holdout.csv — labels exist here and
not at scoring time. The label-free guards (score range, band distribution)
are also exercised here against fixture data; at run time the same guard
functions run inside the model_checks task.
"""
import csv
from pathlib import Path

import pytest

from churn import checks, data
from churn.data import AS_OF
from churn.features import apply_scores, build_features, churn_risk_score, risk_band

HOLDOUT = Path(__file__).parent / "fixtures" / "holdout.csv"

AUC_FLOOR = 0.70


def test_holdout_auc_above_floor():
    """The scoring rule must rank churned customers above retained ones.

    The holdout is a labelled sample: customers scored ninety days before,
    with churned recording what actually happened. The floor is deliberately
    below the current value, so the test fails on a genuine regression and
    not on a small legitimate change.
    """
    with HOLDOUT.open() as f:
        rows = list(csv.DictReader(f))
    scores = [
        churn_risk_score(int(r["recency_days"]), int(r["event_count"]), float(r["revenue"]))
        for r in rows
    ]
    labels = [int(r["churned"]) for r in rows]
    assert checks.auc(scores, labels) >= AUC_FLOOR


def test_scores_stay_in_range_across_the_input_grid():
    """A property over 125 input combinations rather than one example."""
    for recency in (0, 1, 15, 30, 10_000):
        for count in (0, 1, 10, 20, 50):
            for revenue in (0.0, 100.0, 500.0, 1000.0, 5000.0):
                score = churn_risk_score(recency, count, revenue)
                assert 0.0 <= score <= 100.0


@pytest.mark.spark
def test_nan_scores_are_detected(spark):
    """NaN evades a range comparison, so the guard tests NaN explicitly."""
    scored = spark.createDataFrame(
        [("c1", 42.0), ("c2", float("nan"))],
        "customer_id string, churn_risk double",
    )
    assert checks.score_range_violations(scored) == [
        "column churn_risk: 1 NaN or null scores"
    ]


@pytest.mark.spark
def test_clean_data_passes_the_distribution_guard(spark):
    scored = apply_scores(
        build_features(data.make_events(spark), data.make_revenue(spark), as_of=AS_OF)
    )
    assert checks.band_distribution_violations(scored, max_high_share=0.5) == []


@pytest.mark.spark
def test_drift_is_caught_by_the_distribution_guard(spark):
    """The guard itself is under test: plant the fault, expect the report."""
    scored = apply_scores(
        build_features(
            data.make_events(spark, corrupt="drift"),
            data.make_revenue(spark, corrupt="drift"),
            as_of=AS_OF,
        )
    )
    violations = checks.band_distribution_violations(scored, max_high_share=0.5)
    assert len(violations) == 1
    assert "high band" in violations[0]


def test_auc_is_one_for_a_perfect_ranking():
    assert checks.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0


def test_auc_is_zero_for_a_reversed_ranking():
    assert checks.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_auc_counts_ties_as_one_half():
    assert checks.auc([0.5, 0.5], [1, 0]) == 0.5


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        checks.auc([0.5, 0.6], [1, 1])


@pytest.mark.spark
def test_band_column_is_consistent_with_the_thresholds(spark):
    """Row-level consistency: the stored band equals the band of the stored
    score, with the pure function as the oracle."""
    scored = apply_scores(
        build_features(data.make_events(spark), data.make_revenue(spark), as_of=AS_OF)
    )
    for row in scored.collect():
        assert row["risk_band"] == risk_band(row["churn_risk"])


@pytest.mark.spark
def test_an_empty_scored_output_is_detected(spark):
    scored = spark.createDataFrame([], "customer_id string, churn_risk double, risk_band string")
    assert checks.band_distribution_violations(scored) == ["scored output is empty"]
