# Databricks notebook source
# MAGIC %pip install ../dist/marvel_characters-0.1.2-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import time
from importlib.metadata import version
from pathlib import Path

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel
from marvel_characters.models.ab_model import log_ab_model
from marvel_characters.serving.model_serving import ModelServing
from marvel_characters.utils import is_databricks

# COMMAND ----------

# Set up Databricks or local MLflow tracking
spark = SparkSession.builder.getOrCreate()

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    w = WorkspaceClient(profile=profile)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
else:
    w = WorkspaceClient()

config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
# Define tags (customize as needed)
tags = Tags(git_sha="dev", branch="ab-testing")

# COMMAND ----------
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Train model A
basic_model_a = BasicModel(config=config, tags=tags, spark=spark)
basic_model_a.load_data()
basic_model_a.prepare_features()
basic_model_a.train()
basic_model_a.log_model()
model_a_version = basic_model_a.register_model()
model_A_uri = f"models:/{basic_model_a.model_name}/{model_a_version}"

# COMMAND ----------
# Train model B (with different hyperparameters or features)
basic_model_b = BasicModel(config=config, tags=tags, spark=spark)
basic_model_b.parameters = {"learning_rate": 0.01, "n_estimators": 1000, "max_depth": 6}
basic_model_b.model_name = f"{catalog_name}.{schema_name}.marvel_character_model_basic_B"
basic_model_b.load_data()
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
model_b_version = basic_model_b.register_model()
model_B_uri = f"models:/{basic_model_b.model_name}/{model_b_version}"

# COMMAND ----------
# The packaged MarvelABModelWrapper shares portable artifact loading with the
# custom wrapper. log_ab_model stages A/B in distinct folders before logging.

# COMMAND ----------
# Prepare data
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
X_train = train_set[config.num_features + config.cat_features + ["Id"]]
X_test = test_set[config.num_features + config.cat_features + ["Id"]]

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/marvel-characters-ab-testing")
model_name = f"{catalog_name}.{schema_name}.marvel_character_model_pyfunc_ab_test"
wheel_path = Path("../dist") / f"marvel_characters-{version('marvel-characters')}-py3-none-any.whl"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 1, "model": "Model B"})
    dataset = mlflow.data.from_spark(
        train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version=basic_model_a.train_data_version
    )
    mlflow.log_input(dataset, context="training")
    model_info = log_ab_model(
        model_a_uri=model_A_uri,
        model_b_uri=model_B_uri,
        input_example=X_test.iloc[:1],
        wheel_path=str(wheel_path),
        signature=signature,
    )
model_version = mlflow.register_model(
    model_uri=model_info.model_uri, name=model_name
)

# COMMAND ----------
# Model serving setup
endpoint_name = "marvel-characters-ab-testing"
entity_version = model_version.version

# This also updates an existing failed endpoint when re-running the notebook.
ModelServing(model_name=model_name, endpoint_name=endpoint_name).deploy_or_update_serving_endpoint(version=entity_version)

# COMMAND ----------
# Create sample request body
sampled_records = train_set[config.num_features + config.cat_features + ["Id"]].sample(n=1000, replace=True)

import numpy as np
sampled_records = sampled_records.replace({np.nan: None}).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

print(train_set.dtypes)
print(dataframe_records[0])

# COMMAND ----------
# Call the endpoint with one sample record
def call_endpoint(record):
    """Calls the model serving endpoint with a given input record."""
    serving_endpoint = f"{w.config.host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers=w.config.authenticate(),
        json={"dataframe_records": record},
        timeout=120,
    )
    response.raise_for_status()
    return response.status_code, response.text

status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    #time.sleep(0.2)
# COMMAND ----------
