# Databricks notebook source
!pip install databricks-feature-engineering
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text(
    "input_table_path",
    "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
    label="Input Table Name",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "output_table_name",
    "mlops_dbx_talk_dev.churn.telco_cust_features",
    label="Output Feature Table Name",
)

dbutils.widgets.text(
    "label_table_name",
    "mlops_dbx_talk_dev.churn.telco_cust_labels",
    label="Output Feature Table Name",
)



# Primary Keys columns for the feature table;
dbutils.widgets.text(
    "primary_keys",
    "customer_id",
    label="Primary keys columns for the feature table, comma separated.",
)

# COMMAND ----------

import os
import os, sys
from pyspark.sql import functions as F

sys.path.append('..')

from features.compute_features import compute_features_fn
from databricks.feature_engineering import FeatureEngineeringClient
# notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
# %cd $notebook_path
# %cd ../features

# COMMAND ----------

# DBTITLE 1,Define input and output variables
input_table_path = dbutils.widgets.get("input_table_path")
output_table_name = dbutils.widgets.get("output_table_name")
label_table_name = dbutils.widgets.get("label_table_name")
pk_columns = dbutils.widgets.get("primary_keys")

# COMMAND ----------

# DBTITLE 1, Read input data.
df_raw = spark.table(input_table_path)

# COMMAND ----------

# DBTITLE 1,Save labels
# Labels (NO van a Feature Store; van a una tabla aparte para training sets)
df_labels = (
    df_raw
    .select(
        F.col("customerID").alias("customer_id"),
        F.when(F.col("Churn") == "Yes", F.lit(1)).otherwise(F.lit(0)).cast("int").alias("label")
    )
    .dropDuplicates(["customer_id"])
)

df_labels.write.mode("overwrite").saveAsTable(label_table_name)

display(df_labels.groupBy("label").count())


# COMMAND ----------

# DBTITLE 1,Computar features
# Compute features
features_df = compute_features_fn(df_raw)
display(features_df.limit(5))

# COMMAND ----------

# DBTITLE 1, Write computed features.
fe = FeatureEngineeringClient()

# Create the feature table if it does not exist first.
# Note that this is a no-op if a table with the same name and schema already exists.
fe.create_table(
    name=output_table_name,    
    primary_keys=pk_columns,  # Include timeseries column in primary_keys
    df=features_df,
    description="Telco churn - customer-level features."
)

# Write the computed features dataframe.
fe.write_table(
    name=output_table_name,
    df=features_df,
    mode="merge",
)
