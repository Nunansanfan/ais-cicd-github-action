"""Unit tests for the pure scoring logic.

No Spark, no I/O, no fixtures. Every expected value is derived by hand in a
comment, never by calling the function under test.
"""
import pytest

from churn.features import (
    W_FREQUENCY,
    W_RECENCY,
    W_VALUE,
    churn_risk_score,
    risk_band,
)


def test_known_middle_case():
    # recency    min(15/30, 1)      = 0.5   * 0.5 = 0.25
    # frequency  1 - min(10/20, 1)  = 0.5   * 0.3 = 0.15
    # value      1 - min(500/1000,1) = 0.5  * 0.2 = 0.10
    #                                        total 0.50 -> 50.0
    assert churn_risk_score(15, 10, 500.0) == 50.0


def test_best_customer_scores_zero():
    # seen today, at the frequency cap, at the value cap: every part is 0
    assert churn_risk_score(0, 20, 1000.0) == 0.0


def test_worst_customer_scores_hundred():
    # at the recency cap, no events, no revenue: every part is 1
    assert churn_risk_score(30, 0, 0.0) == 100.0


def test_inputs_past_the_caps_do_not_exceed_hundred():
    # 10000 days is far past the 30-day cap; the cap must hold
    assert churn_risk_score(10_000, 0, 0.0) == 100.0


def test_result_is_rounded_to_one_decimal():
    # recency    min(1/30, 1) = 0.03333... * 0.5 = 0.016666...
    # frequency                                    0.15
    # value                                        0.10
    #            total 0.266666... -> 26.666... -> 26.7
    assert churn_risk_score(1, 10, 500.0) == 26.7


@pytest.mark.parametrize(
    "recency_days,event_count,revenue",
    [
        (-1, 10, 500.0),
        (15, -1, 500.0),
        (15, 10, -0.01),
    ],
)
def test_negative_inputs_raise(recency_days, event_count, revenue):
    with pytest.raises(ValueError):
        churn_risk_score(recency_days, event_count, revenue)


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "low"),
        (39.9, "low"),
        (40.0, "medium"),
        (69.9, "medium"),
        (70.0, "high"),
        (100.0, "high"),
    ],
)
def test_band_boundaries(score, expected):
    assert risk_band(score) == expected


def test_score_is_monotonic_in_recency():
    """More days since the last event must never lower the risk."""
    scores = [churn_risk_score(d, 10, 500.0) for d in range(0, 40, 5)]
    assert scores == sorted(scores)


def test_recency_cap_boundary():
    # below the 30-day cap the component still grows; at and past it, it is flat
    # 29 days: 29/30 = 0.96667 * 0.5 = 0.48333; + 0.15 + 0.10 -> 73.3
    assert churn_risk_score(29, 10, 500.0) == 73.3
    assert churn_risk_score(30, 10, 500.0) == 75.0
    assert churn_risk_score(31, 10, 500.0) == 75.0


def test_frequency_cap_boundary():
    # 19 events: 1 - 19/20 = 0.05 * 0.3 = 0.015; + 0.25 + 0.10 -> 36.5
    assert churn_risk_score(15, 19, 500.0) == 36.5
    assert churn_risk_score(15, 20, 500.0) == 35.0
    assert churn_risk_score(15, 25, 500.0) == 35.0


def test_value_cap_boundary():
    # 950 revenue: 1 - 950/1000 = 0.05 * 0.2 = 0.01; + 0.25 + 0.15 -> 41.0
    assert churn_risk_score(15, 10, 950.0) == 41.0
    assert churn_risk_score(15, 10, 1000.0) == 40.0
    assert churn_risk_score(15, 10, 1500.0) == 40.0


def test_zero_is_a_valid_input_everywhere():
    # zero is on the valid side of every guard: seen today, no events, no revenue
    # recency 0 -> 0.0; frequency 1 * 0.3; value 1 * 0.2 -> 50.0
    assert churn_risk_score(0, 0, 0.0) == 50.0


def test_score_is_monotonic_in_event_count():
    """More events must never raise the risk."""
    scores = [churn_risk_score(15, c, 500.0) for c in range(0, 30, 5)]
    assert scores == sorted(scores, reverse=True)


def test_score_is_monotonic_in_revenue():
    """More revenue must never raise the risk."""
    scores = [churn_risk_score(15, 10, v) for v in (0.0, 250.0, 500.0, 750.0, 1000.0)]
    assert scores == sorted(scores, reverse=True)


def test_weights_sum_to_one():
    # a configuration test: the weights are a partition of the score
    assert abs(W_RECENCY + W_FREQUENCY + W_VALUE - 1.0) < 1e-9


def test_score_is_a_float():
    assert isinstance(churn_risk_score(15, 10, 500.0), float)
