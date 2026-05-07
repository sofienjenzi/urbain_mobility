"""
DAG 2 - dag_ml_pipeline.py

Pipeline MLOps: PostgreSQL -> 4 modeles ML -> registry local/MLflow -> logs PostgreSQL.
Schedule: tous les jours a 08:00, apres le chargement DWH.

Modeles:
  - Objectif 1: Prediction Trafic Urbain
  - Objectif 2: Recommandation Trajets
  - Objectif 3: Classification Risques
  - Objectif 4: Estimation CO2 & Energie
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule


logger = logging.getLogger(__name__)

DB_CONN_ID = os.getenv("AIRFLOW_POSTGRES_CONN_ID", "postgres_urbain_dw")
PROJECT_ROOT = Path(os.getenv("MLOPS_PROJECT_PATH", "/opt/airflow/ml_scripts"))
TRAIN_PIPELINE = PROJECT_ROOT / "pipelines" / "train_objective.py"

DEFAULT_ARGS = {
    "owner": "ml_team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}


@dataclass(frozen=True)
class ObjectiveConfig:
    objective_id: str
    task_id: str
    label: str
    required_tables: tuple[str, ...]
    optional_tables: tuple[str, ...] = ()
    timeout_seconds: int = 3600

    @property
    def model_name(self) -> str:
        return f"objective{self.objective_id}"

    @property
    def xcom_key(self) -> str:
        return f"{self.model_name}_result"


OBJECTIVES: tuple[ObjectiveConfig, ...] = (
    ObjectiveConfig(
        objective_id="1",
        task_id="run_objectif1_trafic",
        label="Objectif1 - Prediction Trafic Urbain",
        required_tables=("fact_circulation", "dim_trafic", "dim_zone", "dim_time"),
    ),
    ObjectiveConfig(
        objective_id="2",
        task_id="run_objectif2_trajets",
        label="Objectif2 - Recommandation Trajets",
        required_tables=("fact_circulation", "dim_zone"),
        optional_tables=("dim_ligne", "dim_segment", "dim_arret", "dim_trafic", "dim_time"),
    ),
    ObjectiveConfig(
        objective_id="3",
        task_id="run_objectif3_risques",
        label="Objectif3 - Classification Risques",
        required_tables=("fact_safetyroad", "dim_zone"),
        optional_tables=("dim_accidents", "dim_delinquence"),
    ),
    ObjectiveConfig(
        objective_id="4",
        task_id="run_objectif4_co2",
        label="Objectif4 - Estimation CO2 & Energie",
        required_tables=("dim_zone",),
        optional_tables=(
            "fact_energieconsomation",
            "fact_energiecondomation",
            "fact_pollution",
            "dim_emission_co2",
            "dim_energietransport",
        ),
    ),
)


def on_failure_callback(context: dict[str, Any]) -> None:
    task_id = context["task_instance"].task_id
    dag_id = context["dag"].dag_id
    exception = context.get("exception")
    logger.error("ECHEC - DAG: %s | Tache: %s | Erreur: %s", dag_id, task_id, exception)


def _db_env_from_airflow_conn() -> dict[str, str]:
    conn = PostgresHook(postgres_conn_id=DB_CONN_ID).get_connection(DB_CONN_ID)
    return {
        "DB_HOST": conn.host or "host.docker.internal",
        "DB_PORT": str(conn.port or 5432),
        "DB_NAME": conn.schema or "urbain_dw",
        "DB_USER": conn.login or "postgres",
        "DB_PASSWORD": conn.password or "admin",
        "DB_SCHEMA": (conn.extra_dejson or {}).get("schema", "public"),
    }


def _table_exists(cursor: Any, schema: str, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        );
        """,
        (schema, table_name),
    )
    return bool(cursor.fetchone()[0])


def _table_count(cursor: Any, schema: str, table_name: str) -> int | None:
    if not _table_exists(cursor, schema, table_name):
        return None
    cursor.execute(f'SELECT COUNT(*) FROM "{schema}"."{table_name}";')
    return int(cursor.fetchone()[0])


def check_data_ready(**context: Any) -> dict[str, Any]:
    """Verifie les donnees DWH necessaires aux 4 modeles MLOps."""
    logger.info("Verification des donnees DWH pour les 4 objectifs MLOps...")
    hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    conn = hook.get_conn()
    cursor = conn.cursor()
    schema = _db_env_from_airflow_conn()["DB_SCHEMA"]

    required_tables = sorted({t for obj in OBJECTIVES for t in obj.required_tables})
    optional_tables = sorted({t for obj in OBJECTIVES for t in obj.optional_tables})
    counts: dict[str, int | None] = {}
    missing_required: list[str] = []
    empty_required: list[str] = []

    try:
        for table_name in required_tables + optional_tables:
            count = _table_count(cursor, schema, table_name)
            counts[table_name] = count
            if count is None:
                logger.warning("Table absente: %s.%s", schema, table_name)
            else:
                logger.info("%s.%s: %s lignes", schema, table_name, count)

        for table_name in required_tables:
            count = counts.get(table_name)
            if count is None:
                missing_required.append(table_name)
            elif count == 0:
                empty_required.append(table_name)

        # Objectif4 accepte l'un des deux noms historiques de la table energie.
        energy_candidates = ("fact_energieconsomation", "fact_energiecondomation")
        has_energy_fact = any((counts.get(t) or 0) > 0 for t in energy_candidates)
        if not has_energy_fact:
            missing_required.append("fact_energieconsomation|fact_energiecondomation")

        if missing_required or empty_required:
            raise ValueError(
                "Donnees DWH non pretes. "
                f"Tables requises absentes={missing_required}; tables vides={empty_required}"
            )

        payload = {
            "schema": schema,
            "counts": counts,
            "objectives": [obj.model_name for obj in OBJECTIVES],
        }
        context["ti"].xcom_push(key="data_counts", value=payload)
        return payload
    finally:
        cursor.close()
        conn.close()


def run_ml_objective(objective_id: str, **context: Any) -> dict[str, Any]:
    """Lance l'entrainement MLOps d'un objectif via pipelines/train_objective.py."""
    objective = next(obj for obj in OBJECTIVES if obj.objective_id == objective_id)
    logger.info("=== Lancement %s ===", objective.label)

    if not TRAIN_PIPELINE.exists():
        raise FileNotFoundError(f"Pipeline MLOps introuvable: {TRAIN_PIPELINE}")

    env = {
        **os.environ,
        **_db_env_from_airflow_conn(),
        "FAST_MODE": os.getenv("FAST_MODE", "1"),
        "MLOPS_EXPORT": "1",
        "MLOPS_OBJECTIVE": objective.objective_id,
        "MPLBACKEND": "Agg",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "MLFLOW_EXPERIMENT": os.getenv("MLFLOW_EXPERIMENT", "urban-mlops"),
    }
    if os.getenv("MLFLOW_TRACKING_URI"):
        env["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")

    cmd = [sys.executable, str(TRAIN_PIPELINE), "--objective", objective.objective_id]
    start = perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=objective.timeout_seconds,
    )
    duration_ms = round((perf_counter() - start) * 1000, 2)

    stdout_tail = proc.stdout[-4000:]
    stderr_tail = proc.stderr[-4000:]
    result = {
        "objective": objective.model_name,
        "label": objective.label,
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "duration_ms": duration_ms,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }

    context["ti"].xcom_push(key=objective.xcom_key, value=result)

    if proc.returncode != 0:
        logger.error("%s echoue. STDERR: %s", objective.label, stderr_tail)
        raise RuntimeError(f"{objective.label} echoue avec code {proc.returncode}")

    logger.info("%s termine en %.2f ms. STDOUT: %s", objective.label, duration_ms, stdout_tail)
    return result


def save_ml_results(**context: Any) -> dict[str, Any]:
    """Sauvegarde les statuts des 4 modeles MLOps dans PostgreSQL."""
    logger.info("Sauvegarde des resultats MLOps en base...")
    hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    conn = hook.get_conn()
    cursor = conn.cursor()
    ti = context["ti"]

    results = {
        obj.model_name: ti.xcom_pull(key=obj.xcom_key, task_ids=obj.task_id)
        or {"objective": obj.model_name, "label": obj.label, "status": "unknown"}
        for obj in OBJECTIVES
    }

    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.ml_pipeline_logs (
                id SERIAL PRIMARY KEY,
                dag_id VARCHAR(150),
                run_id VARCHAR(250),
                execution_date TIMESTAMP,
                objective VARCHAR(100),
                label VARCHAR(250),
                status VARCHAR(50),
                duration_ms NUMERIC,
                returncode INTEGER,
                stdout_tail TEXT,
                stderr_tail TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        for objective_name, result in results.items():
            cursor.execute(
                """
                INSERT INTO public.ml_pipeline_logs (
                    dag_id, run_id, execution_date, objective, label, status,
                    duration_ms, returncode, stdout_tail, stderr_tail, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                """,
                (
                    context["dag"].dag_id,
                    context["run_id"],
                    context["execution_date"],
                    objective_name,
                    result.get("label"),
                    result.get("status"),
                    result.get("duration_ms"),
                    result.get("returncode"),
                    result.get("stdout_tail"),
                    result.get("stderr_tail"),
                    json.dumps({"project_root": str(PROJECT_ROOT)}, ensure_ascii=True),
                ),
            )

        conn.commit()
        logger.info("Resultats MLOps sauvegardes: %s", results)
        return results
    except Exception:
        conn.rollback()
        logger.exception("Erreur pendant la sauvegarde des resultats MLOps")
        raise
    finally:
        cursor.close()
        conn.close()


def generate_ml_report(**context: Any) -> str:
    """Genere le rapport final du pipeline MLOps."""
    ti = context["ti"]
    execution_date = context["execution_date"]

    results = [
        ti.xcom_pull(key=obj.xcom_key, task_ids=obj.task_id)
        or {"label": obj.label, "status": "unknown", "duration_ms": None}
        for obj in OBJECTIVES
    ]

    report = f"\n{'=' * 70}\nRAPPORT MLOPS - {execution_date:%Y-%m-%d %H:%M}\n{'=' * 70}"
    for result in results:
        status = result.get("status", "unknown")
        icon = "OK" if status == "success" else "KO"
        duration = result.get("duration_ms")
        suffix = f" ({duration:.0f} ms)" if isinstance(duration, (int, float)) else ""
        report += f"\n  [{icon}] {result.get('label')} : {status}{suffix}"

    success_count = sum(1 for r in results if r.get("status") == "success")
    report += f"\n\nRESULTAT : {success_count}/{len(OBJECTIVES)} modeles MLOps reussis"
    report += f"\nRegistry attendu : {PROJECT_ROOT / 'models'}/objectiveX/latest"
    report += f"\n{'=' * 70}"

    logger.info(report)
    return report


with DAG(
    dag_id="dag_ml_pipeline",
    description="Pipeline MLOps: 4 modeles depuis PostgreSQL urbain_dw vers registry/MLflow",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 8 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "mlflow", "registry", "urban-mobility"],
    on_failure_callback=on_failure_callback,
) as dag:
    t_check_data = PythonOperator(
        task_id="check_data_ready",
        python_callable=check_data_ready,
        on_failure_callback=on_failure_callback,
    )

    objective_tasks = [
        PythonOperator(
            task_id=obj.task_id,
            python_callable=run_ml_objective,
            op_kwargs={"objective_id": obj.objective_id},
            on_failure_callback=on_failure_callback,
        )
        for obj in OBJECTIVES
    ]

    t_save_results = PythonOperator(
        task_id="save_ml_results",
        python_callable=save_ml_results,
        trigger_rule=TriggerRule.ALL_DONE,
        on_failure_callback=on_failure_callback,
    )

    t_report = PythonOperator(
        task_id="generate_ml_report",
        python_callable=generate_ml_report,
        trigger_rule=TriggerRule.ALL_DONE,
        on_failure_callback=on_failure_callback,
    )

    # Les 4 modeles MLOps partent apres la validation DWH.
    # save_ml_results attend tous les objectifs, meme si l'un echoue.
    t_check_data >> objective_tasks >> t_save_results >> t_report
