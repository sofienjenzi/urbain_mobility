from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class RegistryEntry:
    objective: str
    version: str
    created_at: str
    model_path: Path
    metadata_path: Path


def _now_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(exist_ok=True)
    return d


def register_model(objective: str, model: Any, metadata: dict[str, Any], *, version: str | None = None) -> RegistryEntry:
    version = version or _now_version()
    base = models_dir() / objective / version
    base.mkdir(parents=True, exist_ok=True)

    model_path = base / "model.joblib"
    metadata_path = base / "metadata.json"

    joblib.dump(model, model_path)
    metadata_full = {
        **metadata,
        "objective": objective,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata_full, ensure_ascii=False, indent=2), encoding="utf-8")

    latest = models_dir() / objective / "latest"
    if latest.exists():
        if latest.is_dir():
            shutil.rmtree(latest, ignore_errors=True)
        else:
            latest.unlink(missing_ok=True)
    shutil.copytree(base, latest)

    return RegistryEntry(objective=objective, version=version, created_at=metadata_full["created_at"], model_path=model_path, metadata_path=metadata_path)


def load_latest_model(objective: str) -> tuple[Any, dict[str, Any]]:
    latest = models_dir() / objective / "latest"
    model_path = latest / "model.joblib"
    metadata_path = latest / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"No latest model for {objective}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def list_versions(objective: str) -> list[str]:
    base = models_dir() / objective
    if not base.exists():
        return []
    versions = [p.name for p in base.iterdir() if p.is_dir() and p.name != "latest"]
    return sorted(versions, reverse=True)
