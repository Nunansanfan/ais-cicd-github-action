# Databricks notebook source
# model_checks — validate the scored output before it is published.
#
# These are the label-free guards: score range, one row per customer, and the
# band distribution. The metric-threshold test (AUC against the labelled
# holdout) lives in the test suite, because labels exist at development time
# and not at scoring time.
#
# The input DataFrames are rebuilt from src/churn/data.py rather than read
# from the score task: the training workspace grants no writable catalog, so
# tasks cannot hand each other a table. The build is deterministic, so this
# task validates the same frame the score task computed.

# COMMAND ----------

dbutils.widgets.text("src_root", "")
dbutils.widgets.text("corrupt", "none")

import sys

src_root = dbutils.widgets.get("src_root")
if src_root and src_root not in sys.path:
    sys.path.append(src_root)

from churn import checks, data
from churn.data import AS_OF
from churn.features import apply_scores, build_features

# COMMAND ----------

corrupt = dbutils.widgets.get("corrupt")
events = data.make_events(spark, corrupt)
revenue = data.make_revenue(spark, corrupt)

features = build_features(events, revenue, as_of=AS_OF)
scored = apply_scores(features)

checks.fail_on(
    checks.duplicate_key_violations(features, "customer_id"),
    checks.score_range_violations(scored),
    checks.band_distribution_violations(scored, max_high_share=0.5),
)

# COMMAND ----------

high = scored.filter(scored.risk_band == "high").count()

dbutils.notebook.exit(
    f"model checks passed: {scored.count()} rows, "
    f"high band {high}/{scored.count()}"
)
