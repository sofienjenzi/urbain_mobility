from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Allow running as `python pipelines/train_objective.py ...` without PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow

from mlops.mlflow_utils import log_environment, log_dict_as_artifact, setup_mlflow
from mlops.registry import list_versions


SCRIPT_BY_OBJECTIVE = {
    "1": ROOT / "Objectif1_Prediction_Trafic_Urbain_V2.py",
    "2": ROOT / "Objectif2_Recommandation_Trajets_V2.py",
    "3": ROOT / "Objectif3_Classification_Risques_V2.py",
    "4": ROOT / "Objectif4_Estimation_CO2_Energie.py",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["1", "2", "3", "4"])
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "urban-mlops"))
    args = p.parse_args()

    script = SCRIPT_BY_OBJECTIVE[args.objective]
    if not script.exists():
        raise FileNotFoundError(script)

    # Run the script as-is, but ask it to export/register model artifacts.
    env = os.environ.copy()
    env.setdefault("MLOPS_EXPORT", "1")
    env.setdefault("MLOPS_OBJECTIVE", args.objective)
    env.setdefault("MPLBACKEND", "Agg")
    # Windows default consoles often use cp1252; force UTF-8 to avoid crashes
    # when objective scripts print non-ASCII characters.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    setup_mlflow(args.experiment)
    with mlflow.start_run(run_name=f"objective_{args.objective}"):
        log_environment()
        mlflow.log_param("objective", args.objective)
        mlflow.log_param("script", str(script.name))
        mlflow.log_param("fast_mode", env.get("FAST_MODE", ""))

        before = list_versions(f"objective{args.objective}")
        if args.objective == "3":
            # Import-based execution avoids pickling custom classes under '__main__'.
            cmd = [
                sys.executable,
                "-c",
                "import Objectif3_Classification_Risques_V2 as m; print(m.mlops_export_latest(force_retrain=False))",
            ]
        else:
            cmd = [sys.executable, str(script)]

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        mlflow.log_text(proc.stdout[-20000:], artifact_file="stdout_tail.txt")
        mlflow.log_text(proc.stderr[-20000:], artifact_file="stderr_tail.txt")
        if proc.returncode != 0:
            raise RuntimeError(f"Training script failed (code={proc.returncode}). See stderr_tail.txt")

        after = list_versions(f"objective{args.objective}")
        mlflow.log_param("registry_versions_before", len(before))
        mlflow.log_param("registry_versions_after", len(after))
        new_version = after[0] if after and (after[0] not in before) else (after[0] if after else None)
        log_dict_as_artifact({"versions": after, "new_version": new_version}, artifact_path="registry", filename="registry.json")

        print(f"OK objective={args.objective} new_version={new_version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
