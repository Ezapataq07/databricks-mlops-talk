# Databricks notebook source
##################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook shows an example of a Model Training pipeline using Databricks Feature Store tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``mlops_dbx/resources/model-workflow-resource.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the trained model in Unity Catalog. 
#  
##################################################################################

# COMMAND ----------

# %pip install -r ../../requirements.txt

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["dev","staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "training_data_raw",
    "mlops_dbx_talk_dev.churn.telco_churn_train_raw",
    label="Path to the training data",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-mlops_dbx-experiment",
    label="MLflow experiment name",
)


# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "mlops_dbx_talk_dev.churn.telco_churn_model", label="Full (Three-Level) Model Name"
)

# Pickup features table name
dbutils.widgets.text(
    "features_table",
    "mlops_dbx_talk_dev.churn.telco_cust_features",
    label="Features Table",
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
input_table_name = dbutils.widgets.get("training_data_raw")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import pyspark.sql.functions as f
import mlflow
from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature
import time

# COMMAND ----------

# DBTITLE 1, Set experiment
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri('databricks-uc')

if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name=experiment_name)
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1, Load raw data
raw_data = spark.table(input_table_name).select(
        f.col("customerID").alias("customer_id"),
        f.when(f.col("Churn") == "Yes", f.lit(1)).otherwise(f.lit(0)).cast("int").alias("churn")
    )
raw_data.display()

# COMMAND ----------

# DBTITLE 1, Create FeatureLookups
features_table = dbutils.widgets.get("features_table")

feature_lookups = [
    FeatureLookup(
        table_name=features_table,
        feature_names=None,
        lookup_key=["customer_id"],
    ),
]

# COMMAND ----------

# DBTITLE 1, Create Training Dataset


# Since the rounded timestamp columns would likely cause the model to overfit the data
# unless additional feature engineering was performed, exclude them to avoid training on them.

fe = FeatureEngineeringClient()

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fe.create_training_set(
    df=raw_data, # specify the df 
    feature_lookups=feature_lookups, 
    label="churn",
)


# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the Feature Store
training_df.display()

# COMMAND ----------

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

numeric_features = ["tenure_months","tenure_years","monthly_charges","total_charges_filled","avg_monthly_charge_lifetime","abs_charges_gap"]
categorical_features = ["gender","internet_service","contract_type","payment_method","tenure_bucket","monthly_charge_bucket"]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OrdinalEncoder(), categorical_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest()),
    ('classifier', LogisticRegression(random_state=29, n_jobs=-1))
])

# COMMAND ----------

# DBTITLE 1, Train model

from sklearn.model_selection import GridSearchCV, train_test_split


param_grid = [
    {
		'classifier': [ LogisticRegression(random_state=29, n_jobs=-1)],
		'classifier__max_iter': [500, 750,1000],
		# 'classifier__max_depth': [5, 10, 20],
        'feature_selection__k': ["all", 30, 15]
	}
]

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, train_size=0.8,random_state=123)
X_train = train.drop(["customer_id","churn"], axis=1)
X_test = test.drop(["churn"], axis=1)
y_train = train.churn
y_test = test.churn




mlflow.sklearn.autolog(log_input_examples=True, log_models=False, max_tuning_runs=30)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

y_pred = grid_search.predict(X_test)
y_pred_proba = grid_search.predict_proba(X_test)[:,1]

# COMMAND ----------

# DBTITLE 1, Log model and return output.
# Log the trained model with MLflow and package it with feature lookup information.
signature = infer_signature(X_train, grid_search.predict(X_train))

model_info = fe.log_model(
    model=grid_search, #specify model
    artifact_path="model_packaged",
    flavor=mlflow.sklearn,
    training_set=training_set,
    signature=signature,
    registered_model_name=model_name,
)

eval_data = X_test
eval_data["churn"] = y_test
eval_data["predicted_churn"] = y_pred

mlflow.evaluate(
    # model_info.model_uri,
    data=eval_data,
    targets = "churn",
    predictions="predicted_churn",
    model_type = "classifier"
)

mlflow.end_run()    

# client = MlflowClient()
# client.set_registered_model_alias(name=model_name, alias="staging", version=model_info.version)

# # The returned model URI is needed by the model deployment notebook.
# model_version = get_latest_model_version(model_name)
# model_uri = f"models:/{model_name}/{model_version}"
# dbutils.jobs.taskValues.set("model_uri", model_uri)
# dbutils.jobs.taskValues.set("model_name", model_name)
# dbutils.jobs.taskValues.set("model_version", model_version)
# dbutils.notebook.exit(model_uri)

# COMMAND ----------

client = MlflowClient()
run_id = model_info.run_id

def find_version_by_run(model_name, run_id, max_wait_s=60):
    for _ in range(max_wait_s):
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if getattr(mv, "run_id", None) == run_id or str(mv.source).endswith('/model'):
                return int(mv.version)
        time.sleep(1)
    return None

version = getattr(model_info, "registered_model_version", None)
if version is None:
    version = find_version_by_run(model_name, run_id)
    
client.set_registered_model_alias(name=model_name, alias="staging", version=version)

# COMMAND ----------

spark.createDataFrame(train).write.mode("overwrite").saveAsTable("mlops_dbx_talk_dev.churn.telco_churn_train")
spark.createDataFrame(test).write.mode("overwrite").saveAsTable("mlops_dbx_talk_dev.churn.telco_churn_validation")
