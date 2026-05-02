from __future__ import annotations

import os
import json
import logging
import math
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from mlops.registry import list_versions, load_latest_model

LOGGER = logging.getLogger(__name__)


# ----------------------------
# Prometheus metrics
# ----------------------------
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    from prometheus_client.exposition import CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover
    Counter = Gauge = Histogram = None  # type: ignore
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore


def _metrics_enabled() -> bool:
    return Counter is not None and generate_latest is not None


if _metrics_enabled():
    HTTP_REQUESTS_TOTAL = Counter(
        "api_requests_total",
        "Total number of API requests",
        labelnames=["endpoint", "method", "status", "objective"],
    )
    HTTP_REQUEST_DURATION_SECONDS = Histogram(
        "api_request_duration_seconds",
        "API request duration in seconds",
        labelnames=["endpoint", "method", "objective"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
    )
    HTTP_INFLIGHT = Gauge("api_inflight_requests", "In-flight API requests")
    PREDICTION_ERRORS_TOTAL = Counter(
        "api_prediction_errors_total",
        "Total number of prediction errors",
        labelnames=["objective", "error_type"],
    )

    MODEL_CONFIDENCE = Gauge(
        "model_confidence",
        "Rolling/last confidence proxy (0..1 when applicable)",
        labelnames=["objective"],
    )
    MODEL_ACCURACY = Gauge(

        "model_accuracy",
        "Observed accuracy from feedback (0..1, when available)",
        labelnames=["objective"],
    )
    BASELINE_ACCURACY = Gauge(
        "model_baseline_accuracy",
        "Configured baseline accuracy (0..1)",
        labelnames=["objective"],
    )
    DRIFT_DETECTED = Gauge(
        "data_drift_detected",
        "Drift detected by simple threshold rules (0/1)",
        labelnames=["objective", "type"],
    )
    ACCURACY_DEGRADED = Gauge(
        "accuracy_degraded",
        "Accuracy drop detected vs baseline (0/1)",
        labelnames=["objective"],
    )
    CONFIDENCE_DEGRADED = Gauge(
        "confidence_degraded",
        "Confidence drop detected vs baseline (0/1)",
        labelnames=["objective"],
    )

    DATA_FRESHNESS_SECONDS = Gauge(
        "data_freshness_seconds",
        "Age of latest available data in seconds (DB-backed)",
    )
    MISSING_VALUE_RATIO = Gauge(
        "request_missing_value_ratio",
        "Ratio of missing/None values in input payload",
        labelnames=["objective"],
    )

    ALERTS_RECEIVED_TOTAL = Counter(
        "alerts_received_total",
        "Alerts received via Alertmanager webhook",
        labelnames=["alertname", "severity"],
    )


# Optional DB defaults (used for Objective2 city dropdown + aggregates)
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "urbain_dw")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "admin")
os.environ.setdefault("DB_SCHEMA", "public")


app = FastAPI(title="Urban MLOps API", version="1.0")


# ----------------------------
# Simulation knobs (production-incident scenarios)
# ----------------------------
_SIM_STATE: dict[str, Any] = {
    "latency_multiplier": 1.0,
    "error_rate": 0.0,
    "drift_on": False,
    "until": 0.0,
}


def _sim_active() -> bool:
    until = float(_SIM_STATE.get("until") or 0.0)
    return until <= 0.0 or time.time() < until


def _sim_apply_latency() -> None:
    if not _sim_active():
        return
    mult = float(_SIM_STATE.get("latency_multiplier") or 1.0)
    if mult <= 1.0:
        return
    # Add a small bounded delay to simulate overload.
    extra = min(2.0, 0.03 * (mult - 1.0))
    time.sleep(float(extra))


def _sim_maybe_error() -> None:
    if not _sim_active():
        return
    rate = float(_SIM_STATE.get("error_rate") or 0.0)
    if rate <= 0.0:
        return
    if random.random() < rate:
        raise HTTPException(status_code=500, detail="Simulated API error")


def _sim_drift_on() -> bool:
    return bool(_SIM_STATE.get("drift_on")) and _sim_active()


# ----------------------------
# Drift & degradation state
# ----------------------------
_OBJ3_FEEDBACK_WINDOW = int(os.getenv("OBJ3_FEEDBACK_WINDOW", "200"))
_obj3_feedback: deque[int] = deque(maxlen=max(20, _OBJ3_FEEDBACK_WINDOW))


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _get_baseline_accuracy(objective: str) -> float | None:
    if objective == "3":
        return float(os.getenv("OBJ3_BASELINE_ACCURACY", "0.85"))
    if objective == "2":
        return float(os.getenv("OBJ2_BASELINE_ACCURACY", "0.75"))
    return None


def _set_baselines() -> None:
    if not _metrics_enabled():
        return
    for obj in ("2", "3"):
        b = _get_baseline_accuracy(obj)
        if b is not None:
            BASELINE_ACCURACY.labels(objective=obj).set(float(b))


_freshness_cache: dict[str, Any] = {"ts": 0.0, "value": None}


def _db_freshness_seconds() -> float | None:
    """Compute data freshness (seconds since latest dim_time timestamp).

    Uses dim_time.annee/mois/jour/heure because dim_time.date may be out of range.
    Cached for ~30s to avoid hammering DB.
    """

    if not _db_enabled():
        return None

    now = time.time()
    if (now - float(_freshness_cache.get("ts") or 0.0)) < 30.0:
        return _freshness_cache.get("value")

    try:
        from sqlalchemy import text  # type: ignore
    except Exception:
        return None

    schema = os.getenv("DB_SCHEMA", "public")
    sql = text(
        f"""
        SELECT d.annee, d.mois, d.jour, d.heure
        FROM {schema}.dim_time d
        ORDER BY d.annee DESC, d.mois DESC, d.jour DESC
        LIMIT 500
        """
    )

    eng = _get_engine()
    with eng.connect() as c:
        rows = c.execute(sql).fetchall()
    if not rows:
        return None

    # Pick the first row that parses cleanly.
    latest = None
    for (annee, mois, jour, heure) in rows:
        try:
            y = int(float(annee))
            m = int(float(mois))
            d = int(float(jour))
            d = max(1, min(d, 28))
            base = pd.Timestamp(year=y, month=m, day=d)
            td = pd.to_timedelta(str(heure), errors="coerce")
            ts = base + (td if pd.notna(td) else pd.Timedelta(0))
            latest = ts
            break
        except Exception:
            continue
    if latest is None:
        return None

    age = float((pd.Timestamp.utcnow() - latest.tz_localize(None)).total_seconds())
    age = max(0.0, age)
    _freshness_cache["ts"] = now
    _freshness_cache["value"] = age
    return age


def _detect_drift(objective: str, input_data: dict[str, Any], out: dict[str, Any]) -> tuple[bool, str]:
    """Simple drift rules.

    - Obj2: vitesse deviates strongly from baseline mean/std
    - Simulation mode: always drift
    """
    if _sim_drift_on():
        return True, "simulation"

    if objective == "2":
        v = input_data.get("vitesse")
        try:
            v = float(v)
        except Exception:
            return False, ""
        mean = float(os.getenv("OBJ2_BASELINE_VITESSE_MEAN", "25"))
        std = float(os.getenv("OBJ2_BASELINE_VITESSE_STD", "10"))
        std = max(1e-6, std)
        z = abs((v - mean) / std)
        if z >= float(os.getenv("OBJ2_DRIFT_Z", "3")):
            return True, f"vitesse_z={z:.2f}"

    return False, ""


def _confidence_proxy(objective: str, out: dict[str, Any]) -> float | None:
    if objective == "3":
        rs = out.get("risk_score")
        try:
            return float(rs)
        except Exception:
            return None
    if objective == "2":
        score = out.get("quality_score")
        try:
            # quality_score is unbounded; map to 0..1
            return float(_sigmoid(float(score)))
        except Exception:
            return None
    return None


def _update_health_metrics(objective: str, input_data: dict[str, Any], out: dict[str, Any]) -> None:
    if not _metrics_enabled():
        return

    # Missing ratio (very simple data health proxy)
    expected = 0
    if objective == "1":
        expected = 1
    elif objective == "2":
        expected = 2
    elif objective == "3":
        expected = 2
    elif objective == "4":
        expected = 5
    if expected > 0:
        missing = sum(1 for _, v in (input_data or {}).items() if v is None)
        MISSING_VALUE_RATIO.labels(objective=objective).set(float(missing) / float(max(1, expected)))

    # Data freshness (DB)
    age = _db_freshness_seconds()
    if age is not None:
        DATA_FRESHNESS_SECONDS.set(float(age))

    # Drift
    drift, reason = _detect_drift(objective, input_data, out)
    DRIFT_DETECTED.labels(objective=objective, type="rule").set(1.0 if drift else 0.0)
    if drift:
        LOGGER.warning("drift_detected objective=%s reason=%s", objective, reason)

    # Confidence
    conf = _confidence_proxy(objective, out)
    if conf is not None:
        MODEL_CONFIDENCE.labels(objective=objective).set(float(np.clip(conf, 0.0, 1.0)))
        baseline_conf = float(os.getenv(f"OBJ{objective}_BASELINE_CONFIDENCE", "0.55"))
        thresh = float(os.getenv("CONFIDENCE_DROP_REL", "0.10"))
        degraded = conf < baseline_conf * (1.0 - thresh)
        CONFIDENCE_DEGRADED.labels(objective=objective).set(1.0 if degraded else 0.0)
        if degraded:
            LOGGER.warning(
                "confidence_degraded objective=%s conf=%.3f baseline=%.3f", objective, float(conf), baseline_conf
            )

    # Accuracy degradation (Obj3 only, based on feedback)
    if objective == "3" and len(_obj3_feedback) >= 20:
        acc = float(sum(_obj3_feedback) / float(len(_obj3_feedback)))
        MODEL_ACCURACY.labels(objective=objective).set(acc)
        b = _get_baseline_accuracy(objective)
        if b is not None:
            degraded = acc < (b * 0.95)
            ACCURACY_DEGRADED.labels(objective=objective).set(1.0 if degraded else 0.0)
            if degraded:
                LOGGER.error("accuracy_degraded objective=3 acc=%.3f baseline=%.3f", acc, float(b))


class PredictRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    y_true: int
    prediction: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    if not _metrics_enabled():
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
def models() -> dict[str, Any]:
    return {f"objective{k}": {"versions": list_versions(f"objective{k}")} for k in ["1", "2", "3", "4"]}


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.info("API startup complete")
    _set_baselines()


@app.post("/alert-webhook")
async def alert_webhook(req: Request) -> dict[str, Any]:
    """Receives Alertmanager webhooks and logs them."""
    payload = await req.json()
    try:
        alerts = payload.get("alerts") or []
        for a in alerts:
            labels = a.get("labels") or {}
            alertname = str(labels.get("alertname") or "unknown")
            severity = str(labels.get("severity") or "unknown")
            if _metrics_enabled():
                ALERTS_RECEIVED_TOTAL.labels(alertname=alertname, severity=severity).inc()
            LOGGER.error(
                "ALERT alertname=%s severity=%s status=%s summary=%s",
                alertname,
                severity,
                str(a.get("status")),
                str((a.get("annotations") or {}).get("summary") or ""),
            )
    except Exception as e:
        LOGGER.error("Failed to process alert webhook: %s", e)
    return {"status": "ok"}


@app.post("/feedback/3")
def feedback_obj3(req: FeedbackRequest) -> dict[str, Any]:
    """Accepts feedback for Objective 3 so we can compute observed accuracy."""
    y_true = int(req.y_true)
    pred = int(req.prediction)
    ok = 1 if pred == y_true else 0
    _obj3_feedback.append(ok)
    acc = float(sum(_obj3_feedback) / float(len(_obj3_feedback)))
    if _metrics_enabled():
        MODEL_ACCURACY.labels(objective="3").set(acc)
    # Log retraining trigger suggestion
    b = _get_baseline_accuracy("3")
    if b is not None and acc < (b * 0.95) and len(_obj3_feedback) >= 50:
        LOGGER.error("retrain_trigger objective=3 reason=accuracy_drop acc=%.3f baseline=%.3f", acc, float(b))
    return {"status": "ok", "window": len(_obj3_feedback), "accuracy": acc}


@app.post("/simulate/high-traffic")
def simulate_high_traffic(multiplier: float = 3.0, seconds: int = 120) -> dict[str, Any]:
    _SIM_STATE["latency_multiplier"] = float(max(1.0, multiplier))
    _SIM_STATE["until"] = time.time() + int(max(0, seconds))
    return {"status": "ok", "latency_multiplier": _SIM_STATE["latency_multiplier"], "until": _SIM_STATE["until"]}


@app.post("/simulate/errors")
def simulate_errors(rate: float = 0.2, seconds: int = 120) -> dict[str, Any]:
    _SIM_STATE["error_rate"] = float(min(max(rate, 0.0), 1.0))
    _SIM_STATE["until"] = time.time() + int(max(0, seconds))
    return {"status": "ok", "error_rate": _SIM_STATE["error_rate"], "until": _SIM_STATE["until"]}


@app.post("/simulate/drift")
def simulate_drift(on: bool = True, seconds: int = 300) -> dict[str, Any]:
    _SIM_STATE["drift_on"] = bool(on)
    _SIM_STATE["until"] = time.time() + int(max(0, seconds))
    return {"status": "ok", "drift_on": _SIM_STATE["drift_on"], "until": _SIM_STATE["until"]}


@app.post("/predict/{objective}")
def predict(objective: str, req: PredictRequest) -> dict[str, Any]:
    objective = str(objective).replace("objective", "").replace("objectif", "").strip()
    if objective not in {"1", "2", "3", "4"}:
        raise HTTPException(status_code=404, detail="Unknown objective")

    endpoint = "/predict"
    method = "POST"
    if _metrics_enabled():
        HTTP_INFLIGHT.inc()

    start = time.perf_counter()
    status_code = "200"

    # Simulation scenarios (mandatory): high traffic, errors, drift
    _sim_apply_latency()
    _sim_maybe_error()

    try:
        model, meta = load_latest_model(f"objective{objective}")
    except FileNotFoundError:
        status_code = "404"
        if _metrics_enabled():
            HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status_code, objective=objective).inc()
            HTTP_INFLIGHT.dec()
        raise HTTPException(status_code=404, detail=f"No model registered for objective{objective}. Train it first.")

    try:
        out = _predict_with_loaded(objective, model, meta, req.input)
    except Exception as e:
        status_code = "400"
        if _metrics_enabled():
            PREDICTION_ERRORS_TOTAL.labels(objective=objective, error_type=type(e).__name__).inc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    finally:
        if _metrics_enabled():
            HTTP_INFLIGHT.dec()

    dur_ms = (time.perf_counter() - start) * 1000.0
    if _metrics_enabled():
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status_code, objective=objective).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint, method=method, objective=objective).observe(
            max(0.0, dur_ms / 1000.0)
        )
        _update_health_metrics(objective, req.input, out)

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


def _predict_with_loaded(objective: str, model: Any, meta: dict[str, Any], input_data: dict[str, Any]) -> dict[str, Any]:
    if objective == "3":
        # For Obj3 we saved the sklearn pipeline directly, with its metadata
        feature_cols = meta.get("feature_cols") or []
        if not feature_cols:
            raise ValueError("metadata missing feature_cols")
        row = {c: input_data.get(c, np.nan) for c in feature_cols}
        X = pd.DataFrame([row], columns=feature_cols)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        return {"prediction": pred, "risk_score": proba, "risk_level": "HIGH" if pred == 1 else "LOW"}

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
        score = float(reg.predict(x_scaled)[0])

        out: dict[str, Any] = {
            "quality_score": score,
            "recommendation": "GOOD" if score >= 0 else "BAD",
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
        row = {c: input_data.get(c, np.nan) for c in feature_cols}
        X = pd.DataFrame([row], columns=feature_cols)
        pred = {t: float(models[t].predict(X)[0]) for t in targets if t in models}
        return {"prediction": pred}

    if objective == "1":
        kind = meta.get("model_kind")
        if kind == "NaiveLast":
            last = float(meta.get("last_value"))
            steps = int(input_data.get("steps", 1) or 1)
            return {"forecast": [last] * steps}

        if kind == "MovingAvg_3":
            steps = int(input_data.get("steps", 12) or 12)
            hist_vals = [float(x) for x in (meta.get("history_values") or [])]
            if len(hist_vals) < 3:
                last = float(meta.get("last_value"))
                return {"forecast": [last] * steps}
            avg = float(np.mean(hist_vals[-3:]))
            return {"forecast": [avg] * steps}

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
            first = float(meta.get("first_value"))
            last = float(meta.get("last_value"))
            n = int(meta.get("series_length") or 2)
            denom = max(1, n - 1)
            drift = (last - first) / denom
            out = [float(last + drift * (i + 1)) for i in range(steps)]
            return {"forecast": out}

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
        last = float(meta.get("last_value"))
        steps = int(input_data.get("steps", 12) or 12)
        return {"forecast": [last] * steps}
