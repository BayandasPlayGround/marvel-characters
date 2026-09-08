import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env

from marvel_characters.config import Tags


def adjust_predictions(predictions: np.ndarray | list[int]) -> dict[str, list[str]]:
    """Adjust predictions to human-readable format."""
    return {"Survival prediction": ["alive" if pred == 1 else "dead" for pred in predictions]}


class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for LightGBM model."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the LightGBM model."""
        # MLflow 3.1.1 can persist Windows separators in artifact metadata.
        # On Linux, /model/artifacts\\. must resolve to /model/artifacts;
        # the artifact key does not imply a subdirectory with the same name.
        artifact_path = context.artifacts["lightgbm-pipeline"]
        # During logging MLflow also calls load_context before downloading
        # artifacts. Preserve remote URIs; a one-letter scheme is a drive.
        if len(urlparse(artifact_path).scheme) > 1:
            self.model = mlflow.sklearn.load_model(artifact_path)
            return
        model_path = os.path.normpath(artifact_path.replace("\\", "/"))
        if not (Path(model_path) / "MLmodel").is_file():
            raise FileNotFoundError(
                f"LightGBM pipeline MLmodel file not found at {model_path!r} "
                f"(artifact path: {artifact_path!r}). Re-log the wrapper with the complete pipeline artifact."
            )
        self.model = mlflow.sklearn.load_model(model_path)

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        """Predict the survival of a character."""
        predictions = self.model.predict(model_input)
        return adjust_predictions(predictions)

    def log_register_model(
        self,
        wrapped_model_uri: str,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],
        input_example: pd.DataFrame,
    ) -> str:
        """Log and register the model.
        :param wrapped_model_uri: URI of the wrapped model
        :param pyfunc_model_name: Name of the PyFunc model
        :param experiment_name: Name of the experiment
        :param tags: Tags for the model
        :param code_paths: List of code paths
        :param input_example: Input example for the model
        """
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.replace("\\", "/").rsplit("/", 1)[-1]
                additional_pip_deps.append(f"code/{whl_name}")
            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            signature = infer_signature(model_input=input_example, model_output={"Survival prediction": ["alive"]})
            model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={"lightgbm-pipeline": wrapped_model_uri},
                signature=signature,
                code_paths=code_paths,
                conda_env=conda_env,
            )
        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=pyfunc_model_name,
            tags=tags.to_dict(),
        )
        latest_version = registered_model.version
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
