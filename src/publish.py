# Databricks notebook source
# publish — the production step the checks are guarding.
#
# In production this task writes the scored table:
#
#     scored.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.churn_scores")
#
# The training workspace grants no writable catalog, so the write is not
# executed; the task reports the destination it was given instead. The
# teaching point is its position: publish depends on model_checks, and its
# run_if is the default ALL_SUCCESS, so a failure in either checks task
# means the run ends without publishing anything.

# COMMAND ----------

dbutils.widgets.text("src_root", "")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")

import sys

src_root = dbutils.widgets.get("src_root")
if src_root and src_root not in sys.path:
    sys.path.append(src_root)

from churn import data
from churn.data import AS_OF
from churn.features import apply_scores, build_features

# COMMAND ----------

events = data.make_events(spark)
revenue = data.make_revenue(spark)
scored = apply_scores(build_features(events, revenue, as_of=AS_OF))

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

dbutils.notebook.exit(
    f"would publish {scored.count()} rows to {catalog}.{schema}.churn_scores"
)
