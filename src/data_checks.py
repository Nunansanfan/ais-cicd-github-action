# Databricks notebook source
# data_checks — validate the input data before anything is computed.
#
# The checks come from src/churn/checks.py, the same functions the test suite
# runs locally against fixture data. Here they run against the data the job
# is actually given. fail_on raises on any violation, which fails this task
# and leaves every downstream task UPSTREAM_FAILED.

# COMMAND ----------

dbutils.widgets.text("src_root", "")
dbutils.widgets.text("corrupt", "none")

import sys

src_root = dbutils.widgets.get("src_root")
if src_root and src_root not in sys.path:
    sys.path.append(src_root)

from churn import checks, data

# COMMAND ----------

corrupt = dbutils.widgets.get("corrupt")
events = data.make_events(spark, corrupt)
revenue = data.make_revenue(spark, corrupt)

checks.fail_on(
    checks.missing_columns(events, ["customer_id", "event_ts"]),
    checks.missing_columns(revenue, ["customer_id", "amount"]),
    checks.row_count_violations(events, min_rows=1),
    checks.row_count_violations(revenue, min_rows=1),
    checks.null_violations(events, ["customer_id", "event_ts"]),
    checks.null_violations(revenue, ["customer_id", "amount"]),
    checks.range_violations(revenue, "amount", lo=0.0),
)

# COMMAND ----------

dbutils.notebook.exit(
    f"data checks passed: {events.count()} event rows, "
    f"{revenue.count()} revenue rows, corrupt={corrupt}"
)
