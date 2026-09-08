"""Regression tests for cross-platform custom-model artifact loading."""

import posixpath
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from sklearn.dummy import DummyClassifier

from marvel_characters.models.custom_model import MarvelModelWrapper


class TestCustomModelArtifacts(unittest.TestCase):
    """Exercise Windows metadata and real model save/load round trips."""

    def test_logging_preserves_model_uris(self) -> None:
        """Allow MLflow's pre-download load_context call during logging."""
        for uri in ["models:/m-example", "models:/catalog.schema.model/2", "runs:/run-id/pipeline"]:
            with self.subTest(uri=uri), patch("mlflow.sklearn.load_model") as load_model:
                MarvelModelWrapper().load_context(PythonModelContext({"lightgbm-pipeline": uri}, model_config=None))
                load_model.assert_called_once_with(uri)

    def test_linux_context_with_windows_separators(self) -> None:
        """Normalise mixed separators without inventing an artifact directory."""
        for source, expected in [
            (r"/model/artifacts\.", "/model/artifacts"),
            (r"/model/artifacts\lightgbm-pipeline", "/model/artifacts/lightgbm-pipeline"),
            ("/model/artifacts/lightgbm-pipeline", "/model/artifacts/lightgbm-pipeline"),
        ]:
            with self.subTest(source=source):
                context = PythonModelContext({"lightgbm-pipeline": source}, model_config=None)
                with (
                    patch("marvel_characters.models.custom_model.os.path.normpath", side_effect=posixpath.normpath),
                    patch("pathlib.Path.is_file", return_value=True),
                    patch("mlflow.sklearn.load_model") as load_model,
                ):
                    wrapper = MarvelModelWrapper()
                    wrapper.load_context(context)
                    load_model.assert_called_once_with(expected)
                    self.assertIs(wrapper.model, load_model.return_value)

    def test_missing_pipeline_fails_clearly(self) -> None:
        """Reject incomplete artifacts before invoking MLflow's loader."""
        with tempfile.TemporaryDirectory() as directory:
            context = PythonModelContext({"lightgbm-pipeline": directory}, model_config=None)
            with patch("mlflow.sklearn.load_model") as load_model:
                with self.assertRaisesRegex(FileNotFoundError, "pipeline MLmodel file not found"):
                    MarvelModelWrapper().load_context(context)
                load_model.assert_not_called()

    def test_saved_wrapper_loads_and_predicts(self) -> None:
        """Load real packaged artifacts and preserve the readable batch output."""
        features = pd.DataFrame({"Height": [1.7, 1.9]})
        classifier = DummyClassifier(strategy="constant", constant=1).fit(features, [0, 1])
        expected = {"Survival prediction": ["alive", "alive"]}

        with tempfile.TemporaryDirectory() as directory:
            basic_path = Path(directory) / "basic"
            wrapper_path = Path(directory) / "wrapper"
            mlflow.sklearn.save_model(classifier, str(basic_path), pip_requirements=[])
            # Reproduce the reported artifacts\\. layout using a real saved model.
            context = PythonModelContext({"lightgbm-pipeline": str(basic_path) + "\\."}, model_config=None)
            wrapper = MarvelModelWrapper()
            wrapper.load_context(context)
            self.assertEqual(wrapper.predict(context, features), expected)

            mlflow.pyfunc.save_model(
                path=str(wrapper_path),
                python_model=MarvelModelWrapper(),
                artifacts={"lightgbm-pipeline": str(basic_path)},
                signature=infer_signature(features, expected),
                pip_requirements=[],
            )
            loaded = mlflow.pyfunc.load_model(str(wrapper_path))
            self.assertEqual(loaded.predict(features), expected)
