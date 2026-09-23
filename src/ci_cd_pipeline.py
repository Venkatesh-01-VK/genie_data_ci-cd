# Databricks notebook source
# DBTITLE 1,CI/CD Pipeline Overview
# MAGIC %md
# MAGIC # CI/CD Pipeline for Data Integration
# MAGIC
# MAGIC This notebook implements the CI/CD process for the `genie_data_ci-cd` project.
# MAGIC
# MAGIC ## Workflow
# MAGIC
# MAGIC 1. **Development (dev branch)**: Code changes are made on the `dev_genie_data_cicd` branch
# MAGIC 2. **Pull Request**: A PR is created from `dev_genie_data_cicd` → `prd`
# MAGIC 3. **Production (prd branch)**: After PR review and merge, code is deployed to production
# MAGIC 4. **Bundle Deployment**: DAB is deployed using `databricks bundle deploy --target prd`
# MAGIC
# MAGIC ## Environments
# MAGIC
# MAGIC | Target | Branch | Mode |
# MAGIC |--------|--------|------|
# MAGIC | dev | dev_genie_data_cicd | development |
# MAGIC | prd | prd | production |

# COMMAND ----------

# DBTITLE 1,Environment Detection
import os
import json

# Detect the current environment based on bundle target
# In CI/CD, this is passed as a job parameter or environment variable
current_target = os.environ.get("DATABRICKS_BUNDLE_TARGET", "dev")
current_user = os.environ.get("DATABRICKS_BUNDLE_RUN_AS", "venkateshvasu07@gmail.com")

print(f"=== CI/CD Pipeline Environment ===")
print(f"Target: {current_target}")
print(f"Run As: {current_user}")
print(f"Bundle Name: genie_data_ci_cd")
print(f"Workspace: https://dbc-80c9bc92-a375.cloud.databricks.com")
print(f"Branch: {current_target}_genie_data_cicd" if current_target == "dev" else f"Branch: {current_target}")
print(f"Mode: {'development' if current_target == 'dev' else 'production'}")
print("=" * 40)

# COMMAND ----------

# DBTITLE 1,Data Integration Pipeline
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType

# Simulate source data - in a real scenario this would come from an external source
source_data = [
    ("ORD-001", "CUST-101", "Electronics", 2, 599.99, "2026-09-01"),
    ("ORD-002", "CUST-102", "Clothing", 3, 149.50, "2026-09-02"),
    ("ORD-003", "CUST-103", "Electronics", 1, 1299.00, "2026-09-03"),
    ("ORD-004", "CUST-104", "Books", 5, 89.75, "2026-09-04"),
    ("ORD-005", "CUST-105", "Electronics", 2, 449.99, "2026-09-05"),
    ("ORD-006", "CUST-106", "Clothing", 4, 299.00, "2026-09-06"),
    ("ORD-007", "CUST-107", "Books", 2, 45.50, "2026-09-07"),
    ("ORD-008", "CUST-108", "Electronics", 3, 899.99, "2026-09-08"),
]

schema = StructType([
    StructField("order_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("category", StringType(), False),
    StructField("quantity", IntegerType(), False),
    StructField("unit_price", DoubleType(), False),
    StructField("order_date", StringType(), False),
])

# Create the source DataFrame
source_df = spark.createDataFrame(source_data, schema)

# Transform: add calculated columns and metadata
target_df = (source_df
    .withColumn("total_amount", col("quantity") * col("unit_price"))
    .withColumn("processed_at", current_timestamp())
    .withColumn("environment", lit(current_target))
    .withColumn("load_date", current_timestamp())
)

print(f"Source records: {source_df.count()}")
print(f"Target records: {target_df.count()}")
target_df.display()

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Data Quality Validation Checks
print("=== Running Data Quality Checks ===")

# Check 1: No null values in critical columns
critical_cols = ["order_id", "customer_id", "category"]
null_count = 0
for c in critical_cols:
    nulls = target_df.filter(col(c).isNull()).count()
    null_count += nulls
    status = "PASS" if nulls == 0 else "FAIL"
    print(f"  Check NULL - {c}: {nulls} nulls -> {status}")

# Check 2: Total amount is positive
negative_amounts = target_df.filter(col("total_amount") <= 0).count()
print(f"  Check POSITIVE_AMOUNT: {negative_amounts} non-positive -> {'PASS' if negative_amounts == 0 else 'FAIL'}")

# Check 3: Quantity is at least 1
invalid_qty = target_df.filter(col("quantity") < 1).count()
print(f"  Check MIN_QUANTITY: {invalid_qty} invalid -> {'PASS' if invalid_qty == 0 else 'FAIL'}")

# Check 4: Row count match
source_count = source_df.count()
target_count = target_df.count()
print(f"  Check ROW_COUNT: source={source_count}, target={target_count} -> {'PASS' if source_count == target_count else 'FAIL'}")

# Overall result
total_failures = null_count + negative_amounts + invalid_qty + (0 if source_count == target_count else 1)
if total_failures == 0:
    print("\n✅ ALL DATA QUALITY CHECKS PASSED")
else:
    print(f"\n❌ {total_failures} DATA QUALITY CHECKS FAILED")
    raise Exception(f"Data quality validation failed with {total_failures} errors")

# COMMAND ----------

# DBTITLE 1,Aggregated Metrics
# Generate aggregated metrics for the dashboard/reporting layer
from pyspark.sql.functions import sum as spark_sum, avg, count as spark_count, round as spark_round

print("=== Aggregated Metrics ===")

# Revenue by category
category_metrics = (target_df
    .groupBy("category")
    .agg(
        spark_count("order_id").alias("order_count"),
        spark_sum("quantity").alias("total_units"),
        spark_round(spark_sum("total_amount"), 2).alias("total_revenue"),
        spark_round(avg("total_amount"), 2).alias("avg_order_value"),
    )
    .orderBy(col("total_revenue").desc())
)

print("\nRevenue by Category:")
category_metrics.display()

# Daily summary
daily_metrics = (target_df
    .groupBy("order_date")
    .agg(
        spark_count("order_id").alias("daily_orders"),
        spark_round(spark_sum("total_amount"), 2).alias("daily_revenue"),
    )
    .orderBy("order_date")
)

print("\nDaily Summary:")
daily_metrics.display()

# Overall summary
total_revenue = target_df.agg(spark_sum("total_amount")).collect()[0][0]
total_orders = target_df.count()
total_units = target_df.agg(spark_sum("quantity")).collect()[0][0]

print(f"\n{'='*40}")
print(f"Total Orders:     {total_orders}")
print(f"Total Units:      {total_units}")
print(f"Total Revenue:    ${total_revenue:,.2f}")
print(f"Environment:      {current_target}")
print(f"{'='*40}")
print("\n✅ CI/CD Pipeline completed successfully!")

# COMMAND ----------

# DBTITLE 1,CI/CD Workflow Summary
# MAGIC %md
# MAGIC ## CI/CD Workflow Summary
# MAGIC
# MAGIC ### Branch Strategy
# MAGIC
# MAGIC ```mermaid
# MAGIC graph LR
# MAGIC     A[dev_genie_data_cicd] -->|Pull Request| B[prd]
# MAGIC     B -->|Bundle Deploy| C[Production Job]
# MAGIC ```
# MAGIC
# MAGIC ### Steps
# MAGIC
# MAGIC 1. **Develop**: Make code changes on the `dev_genie_data_cicd` branch
# MAGIC 2. **Commit & Push**: Push changes to the remote dev branch
# MAGIC 3. **Create PR**: Open a pull request from `dev_genie_data_cicd` → `prd` on GitHub
# MAGIC 4. **Review**: Review and approve the PR
# MAGIC 5. **Merge**: Merge the PR into the `prd` branch
# MAGIC 6. **Deploy**: Run `databricks bundle deploy --target prd` to deploy the production job
# MAGIC 7. **Run**: The production job runs on schedule (daily at 08:00 UTC)
# MAGIC
# MAGIC ### Bundle Commands
# MAGIC
# MAGIC ```bash
# MAGIC # Validate the bundle
# MAGIC databricks bundle validate --target dev
# MAGIC
# MAGIC # Deploy to dev
# MAGIC databricks bundle deploy --target dev
# MAGIC
# MAGIC # Deploy to production (after PR merge)
# MAGIC databricks bundle deploy --target prd
# MAGIC
# MAGIC # Run the job
# MAGIC databricks bundle run ci_cd_pipeline_job --target prd
# MAGIC ```

# COMMAND ----------

