from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import mlflow


def setup_mlflow(experiment_name: str = "urban-mlops") -> None:
    mlflow.set_experiment(experiment_name)


def log_environment() -> None:
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("platform", platform.platform())


def log_dict_as_artifact(data: dict[str, Any], artifact_path: str, filename: str = "data.json") -> None:
    tmp = Path(".mlflow_tmp")
    tmp.mkdir(exist_ok=True)
    p = tmp / filename
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(p), artifact_path=artifact_path)
