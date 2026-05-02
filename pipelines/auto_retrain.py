from __future__ import annotations

import argparse
from datetime import datetime, timezone

import os
import subprocess
import sys
from pathlib import Path

from mlops.registry import load_latest_model


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _age_days(created_at_iso: str) -> float:
    dt = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True, choices=["1", "2", "3", "4"])
    p.add_argument("--max-age-days", type=float, default=7.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    def _run_train() -> int:
        env = os.environ.copy()
        env.setdefault("MLOPS_EXPORT", "1")
        env.setdefault("MPLBACKEND", "Agg")
        cmd = [sys.executable, str(ROOT / "pipelines" / "train_objective.py"), "--objective", args.objective]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
        return int(proc.returncode)

    if args.force:
        return _run_train()

    try:
        _, meta = load_latest_model(f"objective{args.objective}")
        created_at = str(meta.get("created_at", ""))
        if not created_at:
            raise ValueError("missing created_at")
        age = _age_days(created_at)
        if age <= args.max_age_days:
            print(f"SKIP objective={args.objective} age_days={age:.2f} <= {args.max_age_days}")
            return 0
    except Exception:
        # No model yet or invalid metadata -> retrain
        pass

    # Re-run training
    return _run_train()


if __name__ == "__main__":
    raise SystemExit(main())
