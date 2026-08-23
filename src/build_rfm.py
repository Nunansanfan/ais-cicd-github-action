# Databricks notebook source
# MAGIC %md
# MAGIC # Customer RFM from samples.tpch
# MAGIC
# MAGIC Builds a Recency / Frequency / Monetary segmentation of the TPC-H
# MAGIC customer base and writes it to Unity Catalog.
# MAGIC
# MAGIC RFM (Hughes, *Strategic Database Marketing*, 1994) ranks each customer on
# MAGIC three axes and combines the ranks into a cell. It is the same decomposition
# MAGIC Module 5's churn rule scores at weights 0.5 / 0.3 / 0.2; there the inputs
# MAGIC were synthetic, here they are computed from data.
# MAGIC
# MAGIC | Measure | Source | Definition |
# MAGIC |---|---|---|
# MAGIC | Recency | orders | days from the customer's last order to the reference date |
# MAGIC | Frequency | orders | number of distinct orders |
# MAGIC | Monetary | lineitem | `SUM(l_extendedprice * (1 - l_discount))` |
# MAGIC
# MAGIC **The reference date is `MAX(o_orderdate)`, not today.** TPC-H order dates
# MAGIC run from 1992 to 1998. Measured against `current_date()` every customer
# MAGIC scores about eleven thousand days, the quintiles collapse, and the output
# MAGIC stops being reproducible — which would also stop CI being able to assert
# MAGIC anything about it.

# COMMAND ----------

dbutils.widgets.text("catalog", "ctl_training_dev")
dbutils.widgets.text("schema", "m6")
dbutils.widgets.text("table_suffix", "")
dbutils.widgets.dropdown("enrich", "true", ["true", "false"])

import json

from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SUFFIX = dbutils.widgets.get("table_suffix").strip()
ENRICH = dbutils.widgets.get("enrich") == "true"

if not SUFFIX:
    raise ValueError("table_suffix is required: two people writing one table is not a result")

RFM_TABLE = f"{CATALOG}.{SCHEMA}.customer_rfm_{SUFFIX}"
SUMMARY_TABLE = f"{CATALOG}.{SCHEMA}.rfm_segment_summary_{SUFFIX}"

SEGMENTS = ["Champions", "Loyal", "Potential", "At Risk", "Hibernating"]

print(f"writing {RFM_TABLE}")
print(f"writing {SUMMARY_TABLE}")

# COMMAND ----------

# MAGIC %md ## 1 · The customer dimension — customer, nation, region

# COMMAND ----------

customer = spark.table("samples.tpch.customer")
nation = spark.table("samples.tpch.nation")
region = spark.table("samples.tpch.region")

customer_dim = (
    customer.join(nation, customer.c_nationkey == nation.n_nationkey)
    .join(region, nation.n_regionkey == region.r_regionkey)
    .select(
        F.col("c_custkey").alias("customer_key"),
        F.col("c_name").alias("customer_name"),
        F.col("c_mktsegment").alias("market_segment"),
        F.col("n_name").alias("nation"),
        F.col("r_name").alias("region"),
    )
)

# COMMAND ----------

# MAGIC %md ## 2 · The revenue fact — orders joined to lineitem
# MAGIC
# MAGIC `l_extendedprice * (1 - l_discount)` is the standard TPC-H net-revenue
# MAGIC expression. Revenue lives on the line, not on the order, so monetary value
# MAGIC has to come from `lineitem` rather than from `o_totalprice`.

# COMMAND ----------

orders = spark.table("samples.tpch.orders")
lineitem = spark.table("samples.tpch.lineitem")

revenue = (
    orders.join(lineitem, orders.o_orderkey == lineitem.l_orderkey)
    .withColumn("net_revenue", F.col("l_extendedprice") * (1 - F.col("l_discount")))
    .select("o_custkey", "o_orderkey", "o_orderdate", "net_revenue")
)

# COMMAND ----------

# MAGIC %md ## 3 · The three measures, against a reference date taken from the data

# COMMAND ----------

REFERENCE_DATE = orders.agg(F.max("o_orderdate")).collect()[0][0]
print("reference date:", REFERENCE_DATE)

rfm_base = revenue.groupBy("o_custkey").agg(
    F.max("o_orderdate").alias("last_order_date"),
    F.countDistinct("o_orderkey").alias("frequency"),
    F.sum("net_revenue").alias("monetary"),
)

rfm_base = rfm_base.withColumn(
    "recency_days", F.datediff(F.lit(REFERENCE_DATE), F.col("last_order_date"))
)

# COMMAND ----------

# MAGIC %md ## 4 · Quintile scores, and the segment
# MAGIC
# MAGIC `NTILE(5)` over each measure. Recency is ordered descending so that the
# MAGIC most recent customers land in bucket 5: on every axis, 5 is good.

# COMMAND ----------

scored = (
    rfm_base.withColumn("r_score", F.ntile(5).over(Window.orderBy(F.col("recency_days").desc())))
    .withColumn("f_score", F.ntile(5).over(Window.orderBy(F.col("frequency").asc())))
    .withColumn("m_score", F.ntile(5).over(Window.orderBy(F.col("monetary").asc())))
)

scored = scored.withColumn(
    "rfm_cell",
    F.concat(F.col("r_score").cast("string"), F.col("f_score").cast("string"),
             F.col("m_score").cast("string")),
).withColumn(
    "segment",
    F.when((F.col("r_score") >= 4) & (F.col("f_score") >= 4) & (F.col("m_score") >= 4), "Champions")
    .when((F.col("r_score") >= 3) & (F.col("f_score") >= 3), "Loyal")
    .when((F.col("r_score") >= 4) & (F.col("f_score") <= 2), "Potential")
    .when((F.col("r_score") <= 2) & (F.col("f_score") >= 3), "At Risk")
    .otherwise("Hibernating"),
)

# COMMAND ----------

# MAGIC %md ## 5 · Enrichment — part, partsupp, supplier
# MAGIC
# MAGIC Breadth of purchasing, and margin. `partsupp` carries `ps_supplycost` for
# MAGIC a part-supplier pair, which is what turns revenue into margin — the same
# MAGIC join TPC-H query 9 uses.

# COMMAND ----------

if ENRICH:
    part = spark.table("samples.tpch.part")
    partsupp = spark.table("samples.tpch.partsupp")
    supplier = spark.table("samples.tpch.supplier")

    breadth = (
        orders.join(lineitem, orders.o_orderkey == lineitem.l_orderkey)
        .join(part, lineitem.l_partkey == part.p_partkey)
        .join(supplier, lineitem.l_suppkey == supplier.s_suppkey)
        .join(
            partsupp,
            (lineitem.l_partkey == partsupp.ps_partkey)
            & (lineitem.l_suppkey == partsupp.ps_suppkey),
        )
        .withColumn(
            "margin",
            F.col("l_extendedprice") * (1 - F.col("l_discount"))
            - F.col("ps_supplycost") * F.col("l_quantity"),
        )
        .groupBy("o_custkey")
        .agg(
            F.countDistinct("p_type").alias("distinct_part_types"),
            F.countDistinct("s_suppkey").alias("distinct_suppliers"),
            F.sum("margin").alias("total_margin"),
        )
    )
    scored = scored.join(breadth, "o_custkey", "left")

# COMMAND ----------

# MAGIC %md ## 6 · Write, and report

# COMMAND ----------

result = (
    scored.withColumnRenamed("o_custkey", "customer_key")
    .join(customer_dim, "customer_key", "inner")
    .withColumn("reference_date", F.lit(REFERENCE_DATE))
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
result.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(RFM_TABLE)

summary = result.groupBy("segment").count()
summary.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SUMMARY_TABLE)

# COMMAND ----------

# MAGIC %md ## 7 · Exit with a result CI can assert on
# MAGIC
# MAGIC `print()` does not reach the Jobs API. `dbutils.notebook.exit` is the only
# MAGIC channel to `notebook_output.result`, so the summary the workflow checks
# MAGIC must leave through it. Every segment name is present even when its count
# MAGIC is zero, so the assertion on the number of segments is stable.

# COMMAND ----------

counts = {row["segment"]: row["count"] for row in summary.collect()}
payload = {
    "rows": result.count(),
    "reference_date": str(REFERENCE_DATE),
    "segments": {name: counts.get(name, 0) for name in SEGMENTS},
    "tables": [RFM_TABLE, SUMMARY_TABLE],
}
print(json.dumps(payload, indent=2))
dbutils.notebook.exit(json.dumps(payload))
