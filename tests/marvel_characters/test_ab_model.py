"""Regression coverage for portable, distinct A/B model artifacts."""

import hashlib
import posixpath
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from sklearn.tree import DecisionTreeClassifier

from marvel_characters.models.ab_model import MarvelABModelWrapper, stage_ab_artifacts


class TestABModel(unittest.TestCase):
    """Verify that A/B routing survives packaging without aliasing the models."""

    def test_load_context_normalises_both_artifacts(self) -> None:
        """Handle Windows separators for both artifact keys in a Linux context."""
        model_a, model_b = Mock(), Mock()
        context = PythonModelContext(
            {
                "sklearn-pipeline-model-A": r"/model/artifacts\model_A",
                "sklearn-pipeline-model-B": r"/model/artifacts\model_B",
            },
            model_config=None,
        )
        with (
            patch("marvel_characters.models.custom_model.os.path.normpath", side_effect=posixpath.normpath),
            patch("pathlib.Path.is_file", return_value=True),
            patch("mlflow.sklearn.load_model", side_effect=[model_a, model_b]) as load,
        ):
            wrapper = MarvelABModelWrapper()
            wrapper.load_context(context)
            self.assertEqual(
                [call.args[0] for call in load.call_args_list],
                ["/model/artifacts/model_A", "/model/artifacts/model_B"],
            )
            self.assertIs(wrapper.model_a, model_a)
            self.assertIs(wrapper.model_b, model_b)

    def test_distinct_artifacts_and_routing_survive_round_trip(self) -> None:
        """Package models with identical directory basenames and opposite predictions."""
        features = pd.DataFrame({"Height": [1.7, 1.9]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a, source_b = root / "A" / "artifacts", root / "B" / "artifacts"
            for target, labels in [(source_a, [0, 0]), (source_b, [1, 1])]:
                mlflow.sklearn.save_model(
                    DecisionTreeClassifier().fit(features, labels), str(target), pip_requirements=[]
                )
            request = features.iloc[:1].assign(Id="example")
            wrapper_path = root / "wrapper"
            with stage_ab_artifacts(str(source_a), str(source_b)) as artifacts:
                self.assertNotEqual(artifacts["sklearn-pipeline-model-A"], artifacts["sklearn-pipeline-model-B"])
                mlflow.pyfunc.save_model(
                    path=str(wrapper_path),
                    python_model=MarvelABModelWrapper(),
                    artifacts=artifacts,
                    signature=infer_signature(request, {"Prediction": 0, "model": "Model A"}),
                    pip_requirements=[],
                )
            # Staging is gone; the packaged model must contain both pipelines.
            metadata = mlflow.models.Model.load(str(wrapper_path)).flavors["python_function"]["artifacts"]
            self.assertNotEqual(
                metadata["sklearn-pipeline-model-A"]["path"], metadata["sklearn-pipeline-model-B"]["path"]
            )
            loaded = mlflow.pyfunc.load_model(str(wrapper_path))
            ids = {
                parity: next(
                    str(i) for i in range(100) if int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 2 == parity
                )
                for parity in (0, 1)
            }
            for parity, label, prediction in [(1, "Model A", 0), (0, "Model B", 1)]:
                with self.subTest(label=label):
                    row = request.assign(Id=ids[parity])
                    expected = {"Prediction": prediction, "model": label}
                    self.assertEqual(loaded.predict(row), expected)
                    self.assertEqual(loaded.predict(row), expected)

    def test_incomplete_artifact_fails_before_logging(self) -> None:
        """Do not package directories missing the MLflow model descriptor."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Model A artifact does not contain"):
                with stage_ab_artifacts(directory, directory):
                    self.fail("Incomplete model artifact should not be yielded")
