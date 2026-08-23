"""Tests for the Spark transformations.

DataFrames go in, DataFrames come out; nothing here reads or writes a table.
The input frames are the four-customer fixtures in conftest.py: c1 recent and
active, c2 the worked middle case, c3 with no revenue rows, c4 with its only
event outside the 90-day window.
"""
from datetime import date

import pytest
from pyspark.testing import assertDataFrameEqual

from churn.features import apply_scores, build_features, churn_risk_score, risk_band

pytestmark = pytest.mark.spark


def test_missing_revenue_becomes_zero_not_null(events_df, revenue_df, as_of):
    result = build_features(events_df, revenue_df, as_of=as_of)
    c3 = result.filter(result.customer_id == "c3").first()
    assert c3["revenue"] == 0.0


def test_events_outside_the_window_are_excluded(events_df, revenue_df, as_of):
    result = build_features(events_df, revenue_df, as_of=as_of)
    assert result.filter(result.customer_id == "c4").count() == 0


def test_recency_counts_from_the_last_event(events_df, revenue_df, as_of):
    # c2's most recent event is 2026-06-15; as_of is 2026-06-30
    result = build_features(events_df, revenue_df, as_of=as_of)
    c2 = result.filter(result.customer_id == "c2").first()
    assert c2["recency_days"] == 15


def test_one_row_per_customer(events_df, revenue_df, as_of):
    result = build_features(events_df, revenue_df, as_of=as_of)
    assert result.count() == result.select("customer_id").distinct().count()


def test_build_features_whole_frame(spark, events_df, revenue_df, as_of):
    """The whole output frame, rows and schema, in one comparison."""
    expected = spark.createDataFrame(
        [
            ("c1", 1, 3, 1200.0),
            ("c2", 15, 3, 500.0),
            ("c3", 40, 1, 0.0),
        ],
        "customer_id string, recency_days int, event_count bigint, revenue double",
    )
    result = build_features(events_df, revenue_df, as_of=as_of)
    assertDataFrameEqual(result, expected)


def test_spark_scores_match_the_pure_function(spark):
    """Parity: the Spark expressions and churn_risk_score are one rule.

    apply_scores restates the scoring arithmetic in column expressions so it
    runs without a Python worker. This test scores a grid of inputs both
    ways and compares. The tolerance is one step of the final digit, because
    Spark's round and Python's round can differ at an exact half.
    """
    grid = [
        (r, c, v)
        for r in (0, 1, 15, 29, 30, 100)
        for c in (0, 1, 10, 19, 20, 50)
        for v in (0.0, 100.0, 500.0, 999.0, 1000.0, 5000.0)
    ]
    df = spark.createDataFrame(
        grid, "recency_days int, event_count bigint, revenue double"
    )
    for row in apply_scores(df).collect():
        expected = churn_risk_score(
            row["recency_days"], row["event_count"], row["revenue"]
        )
        assert row["churn_risk"] == pytest.approx(expected, abs=0.1)
        assert row["risk_band"] == risk_band(row["churn_risk"])


def test_apply_scores_adds_the_two_columns(events_df, revenue_df, as_of):
    features = build_features(events_df, revenue_df, as_of=as_of)
    scored = apply_scores(features)
    fields = dict(scored.dtypes)
    assert fields["churn_risk"] == "double"
    assert fields["risk_band"] == "string"


def test_empty_inputs_yield_an_empty_frame(spark, as_of):
    events = spark.createDataFrame([], "customer_id string, event_ts date")
    revenue = spark.createDataFrame([], "customer_id string, amount double")
    result = build_features(events, revenue, as_of=as_of)
    assert result.count() == 0
    assert result.columns == ["customer_id", "recency_days", "event_count", "revenue"]


def test_revenue_without_events_is_excluded(spark, events_df, as_of):
    # the join starts from activity, so revenue for an inactive customer
    # contributes no feature row
    revenue = spark.createDataFrame(
        [("c9", 900.0)], "customer_id string, amount double"
    )
    result = build_features(events_df, revenue, as_of=as_of)
    assert result.filter(result.customer_id == "c9").count() == 0


def test_window_boundary_is_inclusive(spark, revenue_df, as_of):
    # as_of 2026-06-30, lookback 90 -> window_start 2026-04-01; an event
    # exactly on it is inside the window (the filter is >=)
    events = spark.createDataFrame(
        [("c5", date(2026, 4, 1))], "customer_id string, event_ts date"
    )
    result = build_features(events, revenue_df, as_of=as_of)
    c5 = result.filter(result.customer_id == "c5").first()
    assert c5 is not None
    assert c5["recency_days"] == 90


def test_revenue_is_summed_per_customer(events_df, revenue_df, as_of):
    # c1 has two revenue rows, 800 and 400
    result = build_features(events_df, revenue_df, as_of=as_of)
    assert result.filter(result.customer_id == "c1").first()["revenue"] == 1200.0


def test_events_on_the_same_day_are_all_counted(spark, revenue_df, as_of):
    events = spark.createDataFrame(
        [("c6", date(2026, 6, 20)), ("c6", date(2026, 6, 20)), ("c6", date(2026, 6, 20))],
        "customer_id string, event_ts date",
    )
    result = build_features(events, revenue_df, as_of=as_of)
    assert result.filter(result.customer_id == "c6").first()["event_count"] == 3


def test_apply_scores_preserves_the_row_count(events_df, revenue_df, as_of):
    features = build_features(events_df, revenue_df, as_of=as_of)
    assert apply_scores(features).count() == features.count()
