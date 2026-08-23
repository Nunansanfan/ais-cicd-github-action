"""Tests for the data-quality checks.

Two things are under test. First, that clean data passes every check — the
empty list is the contract. Second, that each check detects the fault it
exists for, demonstrated against small frames built inline with the fault
planted. A check that has never been seen to report a violation gives the
same false comfort as a test that has never been seen to fail.
"""
import pytest

from churn import checks, data

pytestmark = pytest.mark.spark


def test_clean_inputs_pass_every_check(spark):
    events = data.make_events(spark)
    revenue = data.make_revenue(spark)
    assert checks.missing_columns(events, ["customer_id", "event_ts"]) == []
    assert checks.null_violations(events, ["customer_id", "event_ts"]) == []
    assert checks.null_violations(revenue, ["customer_id", "amount"]) == []
    assert checks.range_violations(revenue, "amount", lo=0.0) == []


def test_null_ids_are_detected_and_counted(spark):
    events = data.make_events(spark, corrupt="null_ids")
    assert checks.null_violations(events, ["customer_id"]) == [
        "column customer_id: 3 null values"
    ]


def test_negative_amounts_are_detected(spark):
    revenue = spark.createDataFrame(
        [("c1", 120.0), ("c2", -35.0)],
        "customer_id string, amount double",
    )
    assert checks.range_violations(revenue, "amount", lo=0.0) == [
        "column amount: 1 values below 0.0"
    ]


def test_a_missing_column_is_detected(spark):
    events = spark.createDataFrame([("c1",)], "customer_id string")
    assert checks.missing_columns(events, ["customer_id", "event_ts"]) == [
        "missing column: event_ts"
    ]


def test_duplicate_keys_are_detected(spark):
    features = spark.createDataFrame(
        [("c1", 10), ("c1", 12), ("c2", 3)],
        "customer_id string, event_count int",
    )
    assert checks.duplicate_key_violations(features, "customer_id") == [
        "column customer_id: 1 duplicated values"
    ]


def test_fail_on_raises_with_every_violation_listed(spark):
    events = data.make_events(spark, corrupt="null_ids")
    with pytest.raises(ValueError, match="customer_id: 3 null values"):
        checks.fail_on(checks.null_violations(events, ["customer_id"]))


def test_zero_is_not_a_range_violation(spark):
    # the boundary value itself is valid: lo is inclusive
    revenue = spark.createDataFrame(
        [("c1", 0.0)], "customer_id string, amount double"
    )
    assert checks.range_violations(revenue, "amount", lo=0.0) == []


def test_values_above_the_upper_bound_are_detected(spark):
    scored = spark.createDataFrame(
        [("c1", 101.5)], "customer_id string, churn_risk double"
    )
    assert checks.range_violations(scored, "churn_risk", hi=100.0) == [
        "column churn_risk: 1 values above 100.0"
    ]


def test_expected_volume_passes(spark):
    events = data.make_events(spark)
    assert checks.row_count_violations(events, min_rows=1) == []


def test_an_empty_input_is_detected_by_the_volume_check(spark):
    # an empty frame passes every column-level check; only the volume
    # expectation reports it
    events = spark.createDataFrame([], "customer_id string, event_ts date")
    assert checks.null_violations(events, ["customer_id"]) == []
    assert checks.row_count_violations(events, min_rows=1) == [
        "row count 0 below minimum 1"
    ]


def test_fail_on_returns_silently_when_there_are_no_violations(spark):
    events = data.make_events(spark)
    assert checks.fail_on(checks.null_violations(events, ["customer_id"])) is None
