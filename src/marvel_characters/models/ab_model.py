"""Portable packaging and deterministic routing for the A/B demonstration."""

import hashlib
import shutil
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env

from marvel_characters.models.custom_model import load_sklearn_artifact

AB_MODEL_MODULE_ENTRY = "marvel_characters/models/ab_model.py"


class MarvelABModelWrapper(mlflow.pyfunc.PythonModel):
    """Choose A for odd MD5 hashes of Id and B for even hashes."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the two separately packaged pipelines on Windows or Linux."""
        self.model_a = load_sklearn_artifact(context.artifacts["sklearn-pipeline-model-A"])
        self.model_b = load_sklearn_artifact(context.artifacts["sklearn-pipeline-model-B"])

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> dict[str, int | str]:
        """Preserve the course wrapper's first-record routing and output."""
        page_id = str(model_input["Id"].values[0])
        hashed_id = hashlib.md5(page_id.encode(encoding="UTF-8")).hexdigest()
        model, label = (self.model_a, "Model A") if int(hashed_id, 16) % 2 else (self.model_b, "Model B")
        predictions = model.predict(model_input.drop(["Id"], axis=1))
        return {"Prediction": int(predictions[0]), "model": label}


@contextmanager
def stage_ab_artifacts(model_a_uri: str, model_b_uri: str) -> Iterator[dict[str, str]]:
    """Give artifacts unique directories even when both downloads have the same basename."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        artifacts = {}
        for label, uri in [("A", model_a_uri), ("B", model_b_uri)]:
            download_dir = root / f"download_{label}"
            download_dir.mkdir()
            downloaded = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(download_dir))
            staged = root / f"model_{label}"
            shutil.copytree(downloaded, staged)
            if not (staged / "MLmodel").is_file():
                raise FileNotFoundError(f"Model {label} artifact does not contain an MLmodel file: {uri}")
            artifacts[f"sklearn-pipeline-model-{label}"] = str(staged)
        yield artifacts


def _require_module_in_wheel(wheel: Path, module_entry: str) -> None:
    """Fail fast if the wheel about to be packaged does not contain ``module_entry``.

    A stale wheel that predates a source change (such as one built before ``ab_model.py``
    existed) installs without error in the serving container's pip step, then fails only
    when MLflow tries to import the missing module while loading the model. Checking the
    wheel's contents here turns that into an immediate, actionable error at logging time.
    """
    with zipfile.ZipFile(wheel) as archive:
        if module_entry not in archive.namelist():
            raise ModuleNotFoundError(
                f"{wheel} does not contain {module_entry}. Rebuild the project wheel with "
                "`uv build` so it matches the current source before logging the A/B model."
            )


def resolve_ab_wheel_path(dist_dir: str | Path, version_file: str | Path) -> Path:
    """Resolve the project wheel from the version committed to ``version.txt``.

    Building the filename from ``version.txt`` (the package's actual build-time version
    source, per ``pyproject.toml``) instead of the version metadata of whatever happens to
    be installed in the current kernel avoids silently resolving a stale wheel left over in
    ``dist_dir`` from an earlier package version.
    """
    project_version = Path(version_file).read_text(encoding="utf-8").strip()
    wheel = Path(dist_dir) / f"marvel_characters-{project_version}-py3-none-any.whl"
    if not wheel.is_file():
        raise FileNotFoundError(f"Build the project wheel before logging the A/B model: {wheel}")
    _require_module_in_wheel(wheel, AB_MODEL_MODULE_ENTRY)
    return wheel


def log_ab_model(
    model_a_uri: str,
    model_b_uri: str,
    input_example: pd.DataFrame,
    wheel_path: str,
    signature: ModelSignature | None = None,
) -> ModelInfo:
    """Log both pipelines and the wrapper wheel within the caller's MLflow run."""
    wheel = Path(wheel_path)
    if not wheel.is_file():
        raise FileNotFoundError(f"Build the project wheel before logging the A/B model: {wheel}")
    _require_module_in_wheel(wheel, AB_MODEL_MODULE_ENTRY)
    signature = signature or infer_signature(input_example, {"Prediction": 1, "model": "Model B"})
    with stage_ab_artifacts(model_a_uri, model_b_uri) as artifacts:
        return mlflow.pyfunc.log_model(
            name="pyfunc-marvel-character-model-ab",
            python_model=MarvelABModelWrapper(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            code_paths=[str(wheel)],
            conda_env=_mlflow_conda_env(additional_pip_deps=[f"code/{wheel.name}"]),
            metadata={"model_a_uri": model_a_uri, "model_b_uri": model_b_uri},
        )
