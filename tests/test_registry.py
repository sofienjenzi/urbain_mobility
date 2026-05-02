from __future__ import annotations

import json
from pathlib import Path

from mlops.registry import load_latest_model, register_model


def test_register_and_load_latest(tmp_path, monkeypatch):
    # Point the registry to a temp project root by monkeypatching models_dir()
    from mlops import registry as reg

    def _tmp_root() -> Path:
        return tmp_path

    monkeypatch.setattr(reg, "project_root", _tmp_root)

    entry = register_model("objective_test", {"a": 1}, {"foo": "bar"}, version="v1")
    assert entry.model_path.exists()
    assert entry.metadata_path.exists()

    model, meta = load_latest_model("objective_test")
    assert model == {"a": 1}
    assert meta["foo"] == "bar"
    assert meta["objective"] == "objective_test"
    assert meta["version"] == "v1"

    # Metadata should be valid json
    json.loads(entry.metadata_path.read_text(encoding="utf-8"))
