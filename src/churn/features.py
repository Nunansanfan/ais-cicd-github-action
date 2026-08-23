"""Churn scoring, arranged for testability.

The module has three layers, and the boundary between them is the design
taught in Module 5, Part B:

  1. Pure logic. churn_risk_score and risk_band hold every business rule.
     They import nothing from Spark and perform no I/O, so they can be
     executed and tested in a plain Python process.
  2. Spark transformations. build_features and apply_scores take DataFrames
     as arguments and return DataFrames. They never read or write a table.
  3. I/O. load_table, write_scores and main are the only functions that name
     a table. main composes the layers and holds no logic of its own.

Behaviour is identical to the original notebook (src/churn_scoring_notebook.py
in the module assets); only the arrangement differs.
"""
from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# 1. Pure logic
# ---------------------------------------------------------------------------

# Weights, agreed with the CRM team. They must sum to 1.0.
W_RECENCY = 0.5
W_FREQUENCY = 0.3
W_VALUE = 0.2

RECENCY_CAP_DAYS = 30.0
FREQUENCY_CAP_EVENTS = 20.0
VALUE_CAP_REVENUE = 1000.0

HIGH_THRESHOLD = 70.0
MEDIUM_THRESHOLD = 40.0


def churn_risk_score(recency_days: int, event_count: int, revenue: float) -> float:
    """Return a churn risk score from 0.0 to 100.0.

    Risk increases with days since the last event and decreases with event
    count and revenue. Each component is capped, weighted, and summed.

    Raises ValueError on a negative input: a negative recency, count or
    revenue is a data fault, and scoring it would hide the fault.
    """
    if recency_days < 0:
        raise ValueError(f"recency_days must be >= 0, got {recency_days}")
    if event_count < 0:
        raise ValueError(f"event_count must be >= 0, got {event_count}")
    if revenue < 0:
        raise ValueError(f"revenue must be >= 0, got {revenue}")

    recency_part = min(recency_days / RECENCY_CAP_DAYS, 1.0)
    frequency_part = 1.0 - min(event_count / FREQUENCY_CAP_EVENTS, 1.0)
    value_part = 1.0 - min(revenue / VALUE_CAP_REVENUE, 1.0)

    score = 100.0 * (
        W_RECENCY * recency_part
        + W_FREQUENCY * frequency_part
        + W_VALUE * value_part
    )
    return round(score, 1)


def risk_band(score: float) -> str:
    """Bucket a score into the three bands the CRM team acts on."""
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# 2. Spark transformations — DataFrame in, DataFrame out
# ---------------------------------------------------------------------------


def build_features(
    events: DataFrame,
    revenue: DataFrame,
    as_of: date,
    lookback_days: int = 90,
) -> DataFrame:
    """Aggregate raw events and revenue into one feature row per customer.

    A customer with no revenue rows receives revenue 0.0. A customer with no
    events inside the lookback window does not appear in the result.
    """
    window_start = as_of - timedelta(days=lookback_days)

    activity = (
        events.filter(F.col("event_ts") >= F.lit(window_start))
        .groupBy("customer_id")
        .agg(
            F.max("event_ts").alias("last_event_ts"),
            F.count("*").alias("event_count"),
        )
        .withColumn("recency_days", F.datediff(F.lit(as_of), F.col("last_event_ts")))
    )

    totals = revenue.groupBy("customer_id").agg(F.sum("amount").alias("revenue"))

    return (
        activity.join(totals, on="customer_id", how="left")
        .fillna({"revenue": 0.0})
        .select("customer_id", "recency_days", "event_count", "revenue")
    )


def apply_scores(features: DataFrame) -> DataFrame:
    """Attach churn_risk and risk_band columns.

    The arithmetic restates churn_risk_score in Spark column expressions, so
    scoring runs in the JVM and no worker has to import this package — a
    Python UDF would require the churn package on every worker's Python
    path, which depends on how the cluster is configured. The caps, weights
    and thresholds are the same constants the pure function reads, and
    test_transform.py holds a parity test that pins the two implementations
    to each other.
    """
    recency_part = F.least(F.col("recency_days") / RECENCY_CAP_DAYS, F.lit(1.0))
    frequency_part = 1.0 - F.least(F.col("event_count") / FREQUENCY_CAP_EVENTS, F.lit(1.0))
    value_part = 1.0 - F.least(F.col("revenue") / VALUE_CAP_REVENUE, F.lit(1.0))

    score = F.round(
        100.0 * (W_RECENCY * recency_part + W_FREQUENCY * frequency_part + W_VALUE * value_part),
        1,
    )

    return features.withColumn("churn_risk", score).withColumn(
        "risk_band",
        F.when(F.col("churn_risk") >= HIGH_THRESHOLD, "high")
        .when(F.col("churn_risk") >= MEDIUM_THRESHOLD, "medium")
        .otherwise("low"),
    )


# ---------------------------------------------------------------------------
# 3. I/O — the only functions that name a table
# ---------------------------------------------------------------------------


def load_table(spark: SparkSession, table: str) -> DataFrame:
    return spark.table(table)


def write_scores(df: DataFrame, table: str, mode: str = "overwrite") -> None:
    df.write.mode(mode).saveAsTable(table)


def main(
    spark: SparkSession,
    events_table: str,
    revenue_table: str,
    output_table: str,
    as_of: date | None = None,
) -> None:
    as_of = as_of or date.today()
    events = load_table(spark, events_table)
    revenue = load_table(spark, revenue_table)
    features = build_features(events, revenue, as_of=as_of)
    scored = apply_scores(features)
    write_scores(scored, output_table)
