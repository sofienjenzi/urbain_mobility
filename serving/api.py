from __future__ import annotations

import os
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from mlops.registry import list_versions, load_latest_model

LOGGER = logging.getLogger(__name__)


# Optional DB defaults (used for Objective2 city dropdown + aggregates)
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "urbain_dw")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "admin")
os.environ.setdefault("DB_SCHEMA", "public")


app = FastAPI(title="Urban MLOps API", version="1.0")


REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests by method, endpoint and status.",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions by objective and model version.",
    ["objective", "version"],
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total prediction errors by objective.",
    ["objective"],
)
MODEL_ACCURACY = Gauge(
    "model_accuracy_score",
    "Current production-like model accuracy estimate.",
    ["objective"],
)
MODEL_BASELINE_ACCURACY = Gauge(
    "model_baseline_accuracy_score",
    "Baseline accuracy used for degradation detection.",
    ["objective"],
)
MODEL_ACCURACY_DROP = Gauge(
    "model_accuracy_drop_ratio",
    "Baseline accuracy minus current estimated accuracy.",
    ["objective"],
)
MODEL_CONFIDENCE = Gauge(
    "model_confidence_score",
    "Current prediction confidence proxy.",
    ["objective"],
)
MODEL_BASELINE_CONFIDENCE = Gauge(
    "model_baseline_confidence_score",
    "Baseline confidence used for confidence degradation detection.",
    ["objective"],
)
DATA_MISSING_RATIO = Gauge(
    "data_missing_ratio",
    "Ratio of missing input values for the model feature set.",
    ["objective"],
)
DATA_FRESHNESS_SECONDS = Gauge(
    "data_freshness_seconds",
    "Age of event data in seconds when a timestamp/year-month is provided.",
    ["objective"],
)
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Simple input drift score based on out-of-range numeric features and missing values.",
    ["objective"],
)
ALERT_EVENTS = Counter(
    "monitoring_alert_events_total",
    "Application-side anomaly events emitted to logs.",
    ["objective", "alert_type"],
)

BASELINES = {
    "1": {"accuracy": 0.92, "confidence": 0.86, "latency_seconds": 0.25},
    "2": {"accuracy": 0.9989, "confidence": 0.90, "latency_seconds": 0.30},
    "3": {"accuracy": 0.667, "confidence": 0.72, "latency_seconds": 0.30},
    "4": {"accuracy": 0.91, "confidence": 0.84, "latency_seconds": 0.35},
}

REFERENCE_RANGES = {
    "1": {"steps": (1, 36)},
    "2": {"vitesse": (1, 140), "temps_trajet_min": (1, 300), "congestion_index": (0, 9)},
    "3": {"volume": (0, 100000), "taux_1000": (0, 1000), "year": (2020, 2035), "month": (1, 12), "zone_id": (0, 100000)},
    "4": {
        "activity_value": (0, 100000),
        "emission_factor": (0, 20),
        "dechets_kg": (0, 50000),
        "station_kwh": (0, 100000),
        "year": (2020, 2035),
        "month": (1, 12),
    },
}

SIMULATION_STATE: dict[str, float] = {"high_latency_until": 0.0, "api_errors_until": 0.0, "model_drift_until": 0.0}


class PredictRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


class SimulateRequest(BaseModel):
    duration_seconds: int = Field(default=120, ge=1, le=3600)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        LOGGER.exception("Unhandled API error", extra={"path": request.url.path})
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(request.method, endpoint, status).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
def models() -> dict[str, Any]:
    return {f"objective{k}": {"versions": list_versions(f"objective{k}")} for k in ["1", "2", "3", "4"]}


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for objective, values in BASELINES.items():
        MODEL_BASELINE_ACCURACY.labels(objective).set(values["accuracy"])
        MODEL_ACCURACY.labels(objective).set(values["accuracy"])
        MODEL_ACCURACY_DROP.labels(objective).set(0.0)
        MODEL_BASELINE_CONFIDENCE.labels(objective).set(values["confidence"])
        MODEL_CONFIDENCE.labels(objective).set(values["confidence"])
        DATA_MISSING_RATIO.labels(objective).set(0.0)
        DATA_FRESHNESS_SECONDS.labels(objective).set(0.0)
        DATA_DRIFT_SCORE.labels(objective).set(0.0)
    LOGGER.info("API startup complete")


@app.post("/simulate/{scenario}")
def simulate(scenario: str, req: SimulateRequest = SimulateRequest()) -> dict[str, Any]:
    scenario = scenario.strip().lower().replace("-", "_")
    now = time.time()
    until = now + req.duration_seconds
    if scenario == "high_traffic":
        return {
            "scenario": scenario,
            "status": "use scripts/simulate_monitoring.py high_traffic to generate concurrent requests",
        }
    if scenario == "high_latency":
        SIMULATION_STATE["high_latency_until"] = until
    elif scenario == "api_errors":
        SIMULATION_STATE["api_errors_until"] = until
    elif scenario in {"model_drift", "drift", "accuracy_drop"}:
        SIMULATION_STATE["model_drift_until"] = until
        for objective, values in BASELINES.items():
            degraded_accuracy = max(0.0, values["accuracy"] - 0.08)
            degraded_confidence = max(0.0, values["confidence"] - 0.15)
            MODEL_ACCURACY.labels(objective).set(degraded_accuracy)
            MODEL_CONFIDENCE.labels(objective).set(degraded_confidence)
            MODEL_ACCURACY_DROP.labels(objective).set(values["accuracy"] - degraded_accuracy)
            DATA_DRIFT_SCORE.labels(objective).set(0.75)
            ALERT_EVENTS.labels(objective, "model_drift").inc()
        LOGGER.warning("Simulation started: model drift and accuracy degradation", extra={"duration_seconds": req.duration_seconds})
    else:
        raise HTTPException(status_code=404, detail="Unknown scenario. Use high_traffic, high_latency, api_errors or model_drift.")

    LOGGER.warning("Simulation started", extra={"scenario": scenario, "duration_seconds": req.duration_seconds})
    return {"scenario": scenario, "active_until_epoch": until, "duration_seconds": req.duration_seconds}


@app.post("/predict/{objective}")
def predict(objective: str, req: PredictRequest) -> dict[str, Any]:
    objective = str(objective).replace("objective", "").replace("objectif", "").strip()
    if objective not in {"1", "2", "3", "4"}:
        raise HTTPException(status_code=404, detail="Unknown objective")

    start = time.perf_counter()
    _apply_active_simulations(objective)
    try:
        model, meta = load_latest_model(f"objective{objective}")
    except FileNotFoundError:
        PREDICTION_ERRORS.labels(objective).inc()
        LOGGER.error("Prediction failed: model not found", extra={"objective": objective})
        raise HTTPException(status_code=404, detail=f"No model registered for objective{objective}. Train it first.")

    try:
        out = _predict_with_loaded(objective, model, meta, req.input)
    except Exception as e:
        PREDICTION_ERRORS.labels(objective).inc()
        LOGGER.exception("Prediction failed", extra={"objective": objective, "input": req.input})
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    dur_ms = (time.perf_counter() - start) * 1000.0
    PREDICTION_COUNT.labels(objective, str(meta.get("version") or "unknown")).inc()
    _record_model_and_data_health(objective, meta, req.input, out, dur_ms / 1000.0)
    row = {
        "ts": time.time(),
        "objective": objective,
        "version": meta.get("version"),
        "duration_ms": dur_ms,
        "input": req.input,
        "output": out,
    }
    _append_jsonl(Path("data") / "predictions.jsonl", row)

    return {"objective": objective, "model_version": meta.get("version"), "duration_ms": dur_ms, **out}


def _apply_active_simulations(objective: str) -> None:
    now = time.time()
    if SIMULATION_STATE.get("api_errors_until", 0.0) > now:
        PREDICTION_ERRORS.labels(objective).inc()
        ALERT_EVENTS.labels(objective, "simulated_api_error").inc()
        LOGGER.error("Simulated API error", extra={"objective": objective})
        raise HTTPException(status_code=503, detail="Simulated API incident")
    if SIMULATION_STATE.get("high_latency_until", 0.0) > now:
        time.sleep(float(os.getenv("SIMULATED_LATENCY_SECONDS", "1.2")))


def _record_model_and_data_health(
    objective: str,
    meta: dict[str, Any],
    input_data: dict[str, Any],
    output: dict[str, Any],
    latency_seconds: float,
) -> None:
    drift_active = SIMULATION_STATE.get("model_drift_until", 0.0) > time.time()
    baseline = BASELINES[objective]
    baseline_accuracy = float(baseline["accuracy"])
    baseline_confidence = float(baseline["confidence"])
    current_accuracy = max(0.0, baseline_accuracy - (0.08 if drift_active else 0.0))
    confidence = _estimate_confidence(objective, output, baseline_confidence)
    if drift_active:
        confidence = max(0.0, confidence - 0.15)

    missing_ratio = _missing_ratio(meta, input_data)
    freshness_seconds = _freshness_seconds(input_data)
    drift_score = max(_drift_score(objective, input_data), 0.75 if drift_active else 0.0)
    accuracy_drop = max(0.0, baseline_accuracy - current_accuracy)

    MODEL_ACCURACY.labels(objective).set(current_accuracy)
    MODEL_CONFIDENCE.labels(objective).set(confidence)
    MODEL_ACCURACY_DROP.labels(objective).set(accuracy_drop)
    DATA_MISSING_RATIO.labels(objective).set(missing_ratio)
    DATA_FRESHNESS_SECONDS.labels(objective).set(freshness_seconds)
    DATA_DRIFT_SCORE.labels(objective).set(drift_score)

    if latency_seconds > float(baseline["latency_seconds"]) * 3:
        ALERT_EVENTS.labels(objective, "high_latency").inc()
        LOGGER.warning("High latency detected", extra={"objective": objective, "latency_seconds": latency_seconds})
    if accuracy_drop > 0.05:
        ALERT_EVENTS.labels(objective, "accuracy_degradation").inc()
        LOGGER.warning("Accuracy degradation detected", extra={"objective": objective, "accuracy_drop": accuracy_drop})
        ALERT_EVENTS.labels(objective, "retraining_trigger").inc()
        LOGGER.error("Retraining trigger emitted", extra={"objective": objective, "reason": "accuracy_degradation"})
    if confidence < baseline_confidence - 0.10:
        ALERT_EVENTS.labels(objective, "confidence_decrease").inc()
        LOGGER.warning("Confidence decrease detected", extra={"objective": objective, "confidence": confidence})
    if drift_score > 0.30:
        ALERT_EVENTS.labels(objective, "drift_detected").inc()
        LOGGER.warning("Data drift detected", extra={"objective": objective, "drift_score": drift_score})
    if missing_ratio > 0.20:
        ALERT_EVENTS.labels(objective, "missing_values").inc()
        LOGGER.warning("High missing value ratio detected", extra={"objective": objective, "missing_ratio": missing_ratio})


def _missing_ratio(meta: dict[str, Any], input_data: dict[str, Any]) -> float:
    feature_cols = meta.get("feature_cols") or meta.get("numeric_cols") or []
    if not feature_cols:
        return 0.0
    missing = sum(1 for col in feature_cols if input_data.get(col) in (None, ""))
    return float(missing / max(1, len(feature_cols)))


def _freshness_seconds(input_data: dict[str, Any]) -> float:
    raw_ts = input_data.get("timestamp") or input_data.get("event_time") or input_data.get("date")
    now = datetime.now(timezone.utc)
    if raw_ts:
        try:
            dt = pd.to_datetime(raw_ts, utc=True).to_pydatetime()
            return float(max(0.0, (now - dt).total_seconds()))
        except Exception:
            return 0.0
    year = input_data.get("year")
    month = input_data.get("month")
    if year and month:
        try:
            dt = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
            return float(max(0.0, (now - dt).total_seconds()))
        except Exception:
            return 0.0
    return 0.0


def _drift_score(objective: str, input_data: dict[str, Any]) -> float:
    ranges = REFERENCE_RANGES.get(objective, {})
    if not ranges:
        return 0.0
    checked = 0
    drifted = 0
    for col, (low, high) in ranges.items():
        if col not in input_data or input_data.get(col) in (None, ""):
            continue
        try:
            value = float(input_data[col])
        except Exception:
            continue
        checked += 1
        if value < low or value > high:
            drifted += 1
    return float(drifted / max(1, checked))


def _estimate_confidence(objective: str, output: dict[str, Any], default: float) -> float:
    if objective == "3" and "risk_score" in output:
        proba = float(output["risk_score"])
        return float(max(proba, 1.0 - proba))
    if objective == "2" and "quality_score" in output:
        score = abs(float(output["quality_score"]))
        return float(np.clip(1.0 / (1.0 + np.exp(-score)), 0.0, 1.0))
    if objective == "4" and isinstance(output.get("prediction"), dict):
        vals = [abs(float(v)) for v in output["prediction"].values()]
        if vals:
            cv = float(np.std(vals) / (np.mean(vals) + 1e-9))
            return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
    if objective == "1" and output.get("forecast"):
        vals = [float(v) for v in output["forecast"]]
        cv = float(np.std(vals) / (abs(np.mean(vals)) + 1e-9))
        return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
    return default





def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


_DB_ENGINE = None


def _db_enabled() -> bool:
    return all(os.getenv(v) for v in ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_SCHEMA"])


def _get_engine():
    global _DB_ENGINE
    if _DB_ENGINE is not None:
        return _DB_ENGINE
    try:
        from sqlalchemy import create_engine  # type: ignore
        from sqlalchemy.engine import URL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("SQLAlchemy not installed (pip install sqlalchemy psycopg2-binary)") from e

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME"),
    )
    _DB_ENGINE = create_engine(url, pool_pre_ping=True)
    return _DB_ENGINE


def _city_aggregates(ville: str, vitesse: float) -> dict[str, float] | None:
    if not _db_enabled():
        return None
    schema = os.getenv("DB_SCHEMA", "public")
    try:
        from sqlalchemy import text  # type: ignore
    except Exception:
        return None

    # Use speed-conditioned neighborhood so outputs vary with vitesse.
    # Also compute city-wide min/max to normalize congestion if it's not in 0..1.
    sql = text(
        f"""
        WITH base AS (
            SELECT f.vitesse, f.temps_trajet_min, f.congestion_index
            FROM {schema}.fact_circulation f
            JOIN {schema}.dim_zone z ON f.fk_zone = z.zone_id
            WHERE z.ville = :ville
              AND f.vitesse IS NOT NULL
              AND f.temps_trajet_min IS NOT NULL
              AND f.congestion_index IS NOT NULL
        ),
        near AS (
            SELECT * FROM base
            ORDER BY ABS(vitesse - :vitesse)
            LIMIT 200
        )
        SELECT
            (SELECT AVG(vitesse) FROM base) AS avg_speed,
            (SELECT AVG(temps_trajet_min) FROM near) AS avg_time_min,
            (SELECT AVG(congestion_index) FROM near) AS avg_congestion,
            (SELECT MIN(congestion_index) FROM base) AS min_congestion,
            (SELECT MAX(congestion_index) FROM base) AS max_congestion
        """
    )
    eng = _get_engine()
    with eng.connect() as c:
        row = c.execute(sql, {"ville": ville, "vitesse": float(vitesse)}).mappings().first()
    if not row:
        return None
    if row.get("avg_time_min") is None or row.get("avg_congestion") is None:
        return None

    avg_cong_raw = float(row["avg_congestion"])
    min_cong = float(row.get("min_congestion") or avg_cong_raw)
    max_cong = float(row.get("max_congestion") or avg_cong_raw)

    # If congestion already looks like 0..1, keep it.
    if 0.0 <= min_cong and max_cong <= 1.0:
        avg_cong_norm = float(np.clip(avg_cong_raw, 0.0, 1.0))
    else:
        denom = (max_cong - min_cong) if max_cong > min_cong else 1.0
        avg_cong_norm = float(np.clip((avg_cong_raw - min_cong) / denom, 0.0, 1.0))

    out = {
        "avg_speed": float(row.get("avg_speed") or 0.0),
        "avg_time_min": float(row["avg_time_min"]),
        "avg_congestion_raw": avg_cong_raw,
        "avg_congestion": avg_cong_norm,
        "min_congestion_raw": min_cong,
        "max_congestion_raw": max_cong,
    }
    return out


def _risk_features_from_db(zone_id: int | None = None) -> dict[str, Any] | None:
    if not _db_enabled():
        return None
    schema = os.getenv("DB_SCHEMA", "public")
    try:
        from sqlalchemy import text  # type: ignore
    except Exception:
        return None

    where = "WHERE f.fk_zone = :zone_id" if zone_id is not None else ""
    sql = text(
        f"""
        WITH joined AS (
            SELECT
                f.fk_zone AS zone_id,
                z.zone_nom,
                f.volume,
                NULLIF(regexp_replace(replace(f.taux_1000::text, ',', '.'), '[^0-9.\\-]', '', 'g'), '')::double precision AS taux_1000,
                a.type,
                d.categorie
            FROM {schema}.fact_safetyroad f
            LEFT JOIN {schema}.dim_zone z ON f.fk_zone = z.zone_id
            LEFT JOIN {schema}.dim_accidents a ON f.fk_accident = a.accident_id
            LEFT JOIN {schema}.dim_delinquence d ON f.fk_crime = d.crime_id
            {where}
        )
        SELECT
            COALESCE(:zone_id, (SELECT zone_id FROM joined WHERE zone_id IS NOT NULL GROUP BY zone_id ORDER BY COUNT(*) DESC LIMIT 1)) AS zone_id,
            AVG(volume)::double precision AS volume,
            AVG(taux_1000)::double precision AS taux_1000,
            MODE() WITHIN GROUP (ORDER BY zone_nom) AS zone_nom,
            MODE() WITHIN GROUP (ORDER BY type) AS type,
            MODE() WITHIN GROUP (ORDER BY categorie) AS categorie
        FROM joined
        """
    )
    try:
        with _get_engine().connect() as c:
            row = c.execute(sql, {"zone_id": zone_id}).mappings().first()
    except Exception:
        LOGGER.exception("DB risk feature lookup failed")
        return None
    if not row or row.get("volume") is None or row.get("taux_1000") is None:
        return None
    return {
        "zone_id": int(row.get("zone_id") or zone_id or 1),
        "volume": float(row["volume"]),
        "taux_1000": float(row["taux_1000"]),
        "zone_nom": str(row.get("zone_nom") or f"Zone_{row.get('zone_id') or zone_id or 1}"),
        "type": str(row.get("type") or "unknown"),
        "categorie": str(row.get("categorie") or "unknown"),
    }


def _canonical_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    value = str(mode).strip()
    if not value:
        return None
    key = value.lower().replace("é", "e").replace("è", "e").replace("ê", "e")
    aliases = {
        "bus": "bus",
        "metro": "métro",
        "tram": "tram",
        "voiture": "voiture",
        "car": "voiture",
    }
    return aliases.get(key, value.lower())


def _mode_aliases(mode: str | None) -> list[str]:
    canonical = _canonical_mode(mode)
    if canonical == "métro":
        return ["métro", "metro"]
    if canonical:
        return [canonical]
    return []


def _energy_features_from_db(
    zone_id: int | None = None,
    mode: str | None = None,
    year: int | None = None,
    month: int | None = None,
) -> dict[str, Any] | None:
    if not _db_enabled():
        return None
    schema = os.getenv("DB_SCHEMA", "public")
    try:
        from sqlalchemy import text  # type: ignore
    except Exception:
        return None

    filters = []
    params: dict[str, Any] = {}
    if zone_id is not None:
        filters.append("f.fk_zone = :zone_id")
        params["zone_id"] = int(zone_id)
    mode_aliases = _mode_aliases(mode)
    if mode_aliases:
        filters.append("LOWER(COALESCE(e.mode, t.mode, '')) = ANY(:mode_aliases)")
        params["mode_aliases"] = mode_aliases
    if month is not None:
        filters.append("t.mois = :month")
        params["month"] = int(month)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    sql = text(
        f"""
        SELECT
            COALESCE(:zone_id, MODE() WITHIN GROUP (ORDER BY f.fk_zone)) AS zone_id,
            AVG(f.activity_value)::double precision AS activity_value,
            AVG(f.emission_factor)::double precision AS emission_factor,
            AVG(f.dechets_kg)::double precision AS dechets_kg,
            AVG(f.station_kwh)::double precision AS station_kwh,
            MODE() WITHIN GROUP (ORDER BY COALESCE(e.mode, t.mode)) AS mode,
            MODE() WITHIN GROUP (ORDER BY e.activity_type) AS activity_type,
            MAX(t.annee)::integer AS latest_year,
            MODE() WITHIN GROUP (ORDER BY t.mois)::integer AS source_month
        FROM {schema}.fact_energiecondomation f
        LEFT JOIN {schema}.dim_emission_co2 e ON f.fk_emco2 = e.emission_id
        LEFT JOIN {schema}.dim_energietransport t ON f.fk_energie = t.energy_id
        {where}
        """
    )
    try:
        with _get_engine().connect() as c:
            row = c.execute(sql, params | {"zone_id": zone_id}).mappings().first()
    except Exception:
        LOGGER.exception("DB energy feature lookup failed")
        return None
    if not row or row.get("activity_value") is None:
        return None
    latest_year = int(row.get("latest_year") or year or 2022)
    requested_year = int(year or latest_year)
    # The DB contains historical measurements. For future selected years, use the selected
    # month's historical profile and apply a modest annual trend so predictions vary over time.
    year_factor = float(np.clip(1.0 + 0.015 * (requested_year - latest_year), 0.70, 1.50))
    activity_value = float(row.get("activity_value") or 0.0) * year_factor
    station_kwh = float(row.get("station_kwh") or 0.0) * year_factor
    return {
        "zone_id": int(row.get("zone_id") or zone_id or 1),
        "activity_value": activity_value,
        "emission_factor": float(row.get("emission_factor") or 0.0),
        "dechets_kg": float(row.get("dechets_kg") or 0.0),
        "station_kwh": station_kwh,
        "mode": _canonical_mode(str(row.get("mode") or mode or "bus")) or "bus",
        "activity_type": str(row.get("activity_type") or "transport"),
        "source_month": int(row.get("source_month") or month or 1),
        "latest_historical_year": latest_year,
        "requested_year": requested_year,
        "year_adjustment_factor": year_factor,
    }


def _objective1_seasonal_profile_forecast(meta: dict[str, Any], steps: int) -> dict[str, Any]:
    hist_vals = [float(x) for x in (meta.get("history_values") or [])]
    hist_idx = pd.to_datetime(meta.get("history_index") or [], errors="coerce")
    if not hist_vals:
        last = float(meta.get("last_value") or 0.0)
        return {
            "forecast": [last] * steps,
            "forecast_method": "NaiveLastFallback",
            "target": meta.get("target_col"),
            "traffic_summary": _traffic_summary([last] * steps, []),
        }

    last = float(hist_vals[-1])
    first = float(hist_vals[0])
    trend_per_month = (last - first) / max(1, len(hist_vals) - 1)
    seasonal = hist_vals[-12:] if len(hist_vals) >= 12 else hist_vals

    last_date = None
    if len(hist_idx) and not pd.isna(hist_idx[-1]):
        last_date = pd.Timestamp(hist_idx[-1])

    out: list[float] = []
    dates: list[str] = []
    for i in range(int(steps)):
        next_date = (last_date + pd.offsets.MonthBegin(i + 1)).normalize() if last_date is not None else None
        if next_date is not None and len(seasonal) >= 12:
            # Map Jan..Dec to the same month observed in the last historical year.
            base = seasonal[int(next_date.month) - 1]
        else:
            base = seasonal[i % len(seasonal)]

        # Keep the seasonal shape and add a restrained trend so forecasts evolve smoothly.
        yhat = float(base + trend_per_month * (i + 1))
        out.append(float(max(0.0, yhat)))
        if next_date is not None:
            dates.append(next_date.strftime("%Y-%m"))

    payload: dict[str, Any] = {
        "forecast": out,
        "forecast_method": "SeasonalProfileWithTrend",
        "target": meta.get("target_col"),
        "traffic_summary": _traffic_summary(out, dates),
        "traffic_classification": [
            {
                "month": dates[i] if i < len(dates) else f"M+{i + 1}",
                "value": float(value),
                **_traffic_level(float(value)),
            }
            for i, value in enumerate(out)
        ],
    }
    if dates:
        payload["forecast_months"] = dates
    return payload


def _traffic_level(value: float) -> dict[str, str]:
    if value < 2.0:
        return {
            "level": "FLUIDE",
            "severity": "low",
            "description": "Circulation confortable, peu de ralentissements attendus.",
        }
    if value < 3.0:
        return {
            "level": "MODÉRÉ",
            "severity": "medium",
            "description": "Trafic normal avec quelques ralentissements possibles.",
        }
    if value < 5.0:
        return {
            "level": "CHARGÉ",
            "severity": "high",
            "description": "Trafic dense, temps de trajet à surveiller.",
        }
    return {
        "level": "CRITIQUE",
        "severity": "critical",
        "description": "Forte congestion probable, action opérationnelle recommandée.",
    }


def _traffic_summary(values: list[float], dates: list[str]) -> dict[str, Any]:
    if not values:
        return {}
    avg = float(np.mean(values))
    peak_idx = int(np.argmax(values))
    low_idx = int(np.argmin(values))
    return {
        "average_value": avg,
        "average_level": _traffic_level(avg)["level"],
        "average_description": _traffic_level(avg)["description"],
        "peak_month": dates[peak_idx] if peak_idx < len(dates) else f"M+{peak_idx + 1}",
        "peak_value": float(values[peak_idx]),
        "peak_level": _traffic_level(float(values[peak_idx]))["level"],
        "lowest_month": dates[low_idx] if low_idx < len(dates) else f"M+{low_idx + 1}",
        "lowest_value": float(values[low_idx]),
    }


def _predict_with_loaded(objective: str, model: Any, meta: dict[str, Any], input_data: dict[str, Any]) -> dict[str, Any]:
    if objective == "3":
        # For Obj3 we saved the sklearn pipeline directly, with its metadata
        feature_cols = meta.get("feature_cols") or []
        if not feature_cols:
            raise ValueError("metadata missing feature_cols")
        db_features: dict[str, Any] = {}
        if any(input_data.get(c) in (None, "") for c in feature_cols):
            zid = input_data.get("zone_id") or input_data.get("fk_zone")
            db_features = _risk_features_from_db(int(zid)) if zid not in (None, "") else (_risk_features_from_db() or {})
        row = {c: input_data.get(c, db_features.get(c, np.nan)) for c in feature_cols}
        for c in ["year", "month"]:
            if c in row and row[c] in (None, ""):
                row[c] = db_features.get(c, input_data.get(c, np.nan))
        X = pd.DataFrame([row], columns=feature_cols)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        out = {"prediction": pred, "risk_score": proba, "risk_level": "HIGH" if pred == 1 else "LOW"}
        if db_features:
            out["derived_from_db"] = db_features
        return out

    if objective == "2":
        feature_cols = meta.get("numeric_cols") or []
        if not feature_cols:
            raise ValueError("metadata missing numeric_cols")
        scaler = model.get("scaler")
        reg = model.get("regressor")
        if scaler is None or reg is None:
            raise ValueError("model bundle missing scaler/regressor")

        # New UX: inputs are (ville, vitesse). We derive (temps_trajet_min, congestion_index)
        # from DB aggregates per city, then feed the model.
        ville = input_data.get("ville") or input_data.get("city")
        vitesse = input_data.get("vitesse") or input_data.get("vitesse_moyenne") or input_data.get("speed")
        temps_trajet_min = input_data.get("temps_trajet_min")
        congestion_index_in = input_data.get("congestion_index")

        # We separate the value used for model features (normalized 0..1)
        # from the value exposed in outputs (raw, typically 1..9 from DB).
        congestion_index_model: float | None = None
        congestion_index_out: float | None = None

        derived: dict[str, Any] = {}
        if temps_trajet_min is None or congestion_index_in is None:
            if not ville:
                raise ValueError("Missing 'ville' (or provide temps_trajet_min + congestion_index)")
            stats = _city_aggregates(str(ville), float(vitesse) if vitesse is not None else 0.0)
            if stats is None:
                raise ValueError(f"No DB aggregates found for ville={ville!r} (check dim_zone/fact_circulation)")

            avg_speed = float(stats.get("avg_speed") or 0.0)
            avg_time = float(stats["avg_time_min"])
            cong_model = float(stats["avg_congestion"])  # normalized 0..1
            cong_raw = float(stats.get("avg_congestion_raw") or 0.0)  # raw DB scale (e.g. 1..9)

            v = float(vitesse) if vitesse is not None else avg_speed
            if v and avg_speed and v > 0 and avg_speed > 0:
                time_est = avg_time * (avg_speed / v)
            else:
                time_est = avg_time

            # clamp to sane ranges
            time_est = float(np.clip(time_est, 1.0, 300.0))
            cong_model = float(np.clip(cong_model, 0.0, 1.0))

            temps_trajet_min = time_est
            congestion_index_model = cong_model
            congestion_index_out = cong_raw
            derived = {
                "temps_trajet_min": time_est,
                "congestion_index_model": cong_model,
                "congestion_index_raw": cong_raw,
                "congestion_raw_range": [
                    float(stats.get("min_congestion_raw") or 0.0),
                    float(stats.get("max_congestion_raw") or 0.0),
                ],
            }
        else:
            # If caller provides congestion_index manually, try to handle both scales:
            # - 0..1: assume already normalized
            # - 1..9: assume raw congestion, normalize to 0..1 for model
            cong_in = float(congestion_index_in)
            if 0.0 <= cong_in <= 1.0:
                congestion_index_model = float(np.clip(cong_in, 0.0, 1.0))
                congestion_index_out = cong_in
            elif 1.0 <= cong_in <= 9.0:
                congestion_index_out = cong_in
                congestion_index_model = float(np.clip((cong_in - 1.0) / 8.0, 0.0, 1.0))
            else:
                # Unknown scale: keep raw for output; clip for model to avoid blowups
                congestion_index_out = cong_in
                congestion_index_model = float(np.clip(cong_in, 0.0, 1.0))

        if vitesse is None:
            raise ValueError("Missing 'vitesse' (km/h)")

        if congestion_index_model is None or congestion_index_out is None:
            raise ValueError("Internal error: congestion_index not resolved")

        # Prepare model input vector
        def _val(col: str) -> float:
            if col in input_data and input_data.get(col) is not None:
                return float(input_data[col])
            if col in {"vitesse", "vitesse_moyenne"}:
                return float(vitesse)
            if col in {"temps_trajet_min", "temps_trajet"}:
                return float(temps_trajet_min)
            if col == "congestion_index":
                return float(congestion_index_model)
            return 0.0

        x = np.asarray([_val(c) for c in feature_cols], dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        raw_score = float(reg.predict(x_scaled)[0])
        quality_score = float(np.clip(100.0 / (1.0 + np.exp(-raw_score)), 0.0, 100.0))

        out: dict[str, Any] = {
            "quality_score": quality_score,
            "raw_quality_score": raw_score,
            "recommendation": "GOOD" if quality_score >= 50.0 else "BAD",
            "inputs": {"ville": str(ville) if ville else None, "vitesse": float(vitesse)},
            "outputs": {
                "temps_trajet_min": float(temps_trajet_min),
                "congestion_index": float(congestion_index_out),
            },
        }
        if derived:
            out["derived_from_db"] = derived
        return out

    if objective == "4":
        feature_cols = meta.get("feature_cols") or []
        targets = meta.get("targets") or []
        if not feature_cols or not targets:
            raise ValueError("metadata missing feature_cols/targets")
        models = model.get("models")
        if not isinstance(models, dict):
            raise ValueError("model bundle missing models")
        db_features: dict[str, Any] = {}
        if any(input_data.get(c) in (None, "") for c in feature_cols):
            zid = input_data.get("zone_id") or input_data.get("fk_zone")
            mode = _canonical_mode(input_data.get("mode") or input_data.get("mode_transport"))
            year = input_data.get("year") or input_data.get("annee")
            month = input_data.get("month") or input_data.get("mois")
            year_int = int(year) if year not in (None, "") else None
            month_int = int(month) if month not in (None, "") else None
            db_features = (
                _energy_features_from_db(int(zid), str(mode) if mode else None, year_int, month_int)
                if zid not in (None, "")
                else (
                    _energy_features_from_db(None, str(mode), year_int, month_int)
                    if mode
                    else (_energy_features_from_db(year=year_int, month=month_int) or {})
                )
            )
        row = {c: input_data.get(c, db_features.get(c, np.nan)) for c in feature_cols}
        if "mode" in row:
            row["mode"] = _canonical_mode(row.get("mode")) or db_features.get("mode", row.get("mode"))
        X = pd.DataFrame([row], columns=feature_cols)
        pred = {t: float(models[t].predict(X)[0]) for t in targets if t in models}
        out = {"prediction": pred}
        if db_features:
            out["derived_from_db"] = db_features
        return out

    if objective == "1":
        kind = meta.get("model_kind")
        if kind == "NaiveLast":
            steps = int(input_data.get("steps", 1) or 1)
            return _objective1_seasonal_profile_forecast(meta, steps)

        if kind == "MovingAvg_3":
            steps = int(input_data.get("steps", 12) or 12)
            return _objective1_seasonal_profile_forecast(meta, steps)

        if kind == "SeasonalNaive":
            steps = int(input_data.get("steps", 12) or 12)
            hist_vals = [float(x) for x in (meta.get("history_values") or [])]
            if len(hist_vals) < 12:
                last = float(meta.get("last_value"))
                return {"forecast": [last] * steps}
            season = hist_vals[-12:]
            out = [float(season[i % 12]) for i in range(steps)]
            return {"forecast": out}

        if kind == "NaiveDrift":
            steps = int(input_data.get("steps", 12) or 12)
            return _objective1_seasonal_profile_forecast(meta, steps)

        if kind == "SARIMAX":
            steps = int(input_data.get("steps", 12) or 12)
            pred = model.get_forecast(steps=steps).predicted_mean
            return {"forecast": [float(x) for x in pred.values]}

        if kind == "ML_Lags_GBR":
            steps = int(input_data.get("steps", 12) or 12)
            lags = tuple(meta.get("lags", []))
            history = pd.Series(meta.get("history_values", []), index=pd.to_datetime(meta.get("history_index", [])))
            if history.empty:
                raise ValueError("missing history")
            # iterative
            hist = history.copy()
            preds = []
            last_date = hist.index[-1]
            for _ in range(steps):
                next_date = (last_date + pd.offsets.MonthBegin(1)).normalize()
                feats = {f"lag_{lag}": float(hist.iloc[-lag]) for lag in lags}
                m = int(next_date.month)
                feats["month"] = m
                feats["month_sin"] = float(np.sin(2 * np.pi * m / 12))
                feats["month_cos"] = float(np.cos(2 * np.pi * m / 12))
                yhat = float(model.predict(pd.DataFrame([feats]))[0])
                preds.append(yhat)
                hist.loc[next_date] = yhat
                last_date = next_date
            return {"forecast": preds}

        # Unknown kind -> fallback to NaiveLast
        steps = int(input_data.get("steps", 12) or 12)
        return _objective1_seasonal_profile_forecast(meta, steps)
