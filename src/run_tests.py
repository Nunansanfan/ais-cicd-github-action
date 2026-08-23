# Databricks notebook source
# MAGIC %md
# MAGIC # run_tests — the deployed suite, executed on the cluster
# MAGIC
# MAGIC This notebook is the job task of the `run_tests` resource. It runs the
# MAGIC **same `tests/` folder that was deployed with the bundle**, against the
# MAGIC Databricks Runtime's own Python and Spark.
# MAGIC
# MAGIC It works because `tests/` is part of the bundle. Nothing excludes it in
# MAGIC `dev`, so `databricks bundle deploy` uploads `tests/`, `pytest.ini` and
# MAGIC `src/` together, and `${workspace.file_path}` is where they landed. The
# MAGIC job passes that path in as the `bundle_root` parameter.
# MAGIC
# MAGIC **Local pytest and this task are not the same gate.** Local pytest runs
# MAGIC before deployment, in seconds, against your laptop's Python. This runs
# MAGIC after deployment, against the runtime that will actually execute the job —
# MAGIC the runtime's Spark version, the cluster's installed libraries, and the
# MAGIC workspace's own permissions. A suite that is green locally and red here
# MAGIC has found an environment difference, which is exactly what it is for.

# COMMAND ----------

# MAGIC %pip install pytest==8.2.0
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("bundle_root", "")

import os
import sys

# Where the bundle was deployed. The job passes ${workspace.file_path}; the
# default keeps the notebook runnable interactively from its own location.
bundle_root = dbutils.widgets.get("bundle_root") or os.path.dirname(os.getcwd())

# pytest resolves testpaths and pythonpath relative to the rootdir it finds,
# so the run must start from the folder holding pytest.ini.
os.chdir(bundle_root)

# The deployed folder is not a scratch directory. Bytecode caching is disabled
# here and pytest's own cache plugin is disabled at the call below; without
# both, the run can fail on a write error instead of on a test.
sys.dont_write_bytecode = True

print("bundle_root:", bundle_root)
print("contents   :", sorted(os.listdir(bundle_root)))

# COMMAND ----------

import pytest

retcode = int(pytest.main(["-q", "-p", "no:cacheprovider", "tests/"]))
print("pytest exit code:", retcode)

# COMMAND ----------

# The task must FAIL when the suite fails. A notebook that merely prints red
# text finishes SUCCESS, and the job would report a green run over a broken
# build. Raising is what turns the suite into a gate.
assert retcode == 0, f"test suite failed: pytest exit code {retcode}"

dbutils.notebook.exit(f"test suite passed (pytest exit code {retcode})")
