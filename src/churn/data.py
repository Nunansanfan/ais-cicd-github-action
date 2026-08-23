"""Deterministic synthetic input data for the m5demo job.

The training workspace grants no writable Unity Catalog catalog, so the job
cannot read production tables. Every task instead rebuilds the same input
DataFrames from this module. The build is deterministic — no randomness — so
two tasks in one run, or two runs on one day, see identical data, and the
numbers on the Module 5 slides can be reproduced exactly.

The corrupt parameter plants one of two faults, so the class can watch the
in-job checks catch them:

  none        clean data (the default)
  null_ids    three event rows carry a null customer_id — caught by the
              data_checks task
  drift       every event is 60 days older than it should be — every score
              rises, and the model_checks task rejects the band distribution

The fault is selected per run:  databricks bundle run scoring --var corrupt=drift
"""
from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession

AS_OF = date(2026, 6, 30)

# customer_id, days between events, days from as_of back to the last event,
# events in the window, total revenue
_PROFILES = [
    ("c1", 3, 1, 28, 1450.0),   # active, high value
    ("c2", 8, 15, 10, 500.0),   # the worked middle case: score 50.0
    ("c3", 6, 5, 14, 260.0),
    ("c4", 20, 40, 3, 0.0),     # no revenue rows at all
    ("c5", 12, 25, 6, 120.0),
    ("c6", 4, 2, 22, 980.0),
    ("c7", 15, 55, 2, 40.0),
    ("c8", 10, 8, 9, 700.0),
]


def make_events(spark: SparkSession, corrupt: str = "none") -> DataFrame:
    """One row per event: customer_id, event_ts (a date)."""
    shift = 60 if corrupt == "drift" else 0
    rows = []
    for cid, gap, recency, count, _ in _PROFILES:
        last = AS_OF - timedelta(days=recency + shift)
        for i in range(count):
            rows.append((cid, last - timedelta(days=i * gap)))
    if corrupt == "null_ids":
        rows[3] = (None, rows[3][1])
        rows[17] = (None, rows[17][1])
        rows[31] = (None, rows[31][1])
    return spark.createDataFrame(rows, "customer_id string, event_ts date")


def make_revenue(spark: SparkSession, corrupt: str = "none") -> DataFrame:
    """One row per purchase: customer_id, amount. c4 has no rows."""
    rows = []
    for cid, _, _, _, total in _PROFILES:
        if total <= 0:
            continue
        half = round(total / 2, 2)
        rows.append((cid, half))
        rows.append((cid, round(total - half, 2)))
    return spark.createDataFrame(rows, "customer_id string, amount double")
