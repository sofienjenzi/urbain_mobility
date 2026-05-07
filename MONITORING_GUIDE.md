# Production-like Monitoring - Prometheus & Grafana

This project now exposes production-like observability for the FastAPI serving layer.

## Services

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- API metrics: http://localhost:8000/metrics
- PostgreSQL exporter metrics: http://localhost:9187/metrics

Grafana login:

- user: `admin`
- password: `admin`

## Start

```powershell
docker compose up --build
```

Prometheus scrapes the API and PostgreSQL exporter every 10 seconds from `/metrics`.

## Metrics

Traffic:

- `api_requests_total`
- `model_predictions_total`

Performance:

- `api_request_latency_seconds`

Stability:

- `api_requests_total{status=~"5.."}`
- `model_prediction_errors_total`

Model health:

- `model_accuracy_score`
- `model_baseline_accuracy_score`
- `model_accuracy_drop_ratio`
- `model_confidence_score`
- `model_baseline_confidence_score`

Data health:

- `data_missing_ratio`
- `data_freshness_seconds`
- `data_drift_score`

Observability events:

- `monitoring_alert_events_total`

PostgreSQL database health:

- `pg_up`
- `pg_database_size_bytes`
- `pg_stat_database_numbackends`
- `pg_stat_database_xact_commit`
- `pg_stat_database_xact_rollback`
- `pg_stat_database_blks_read`
- `pg_stat_database_blks_hit`

## Dashboard

The dashboard is automatically provisioned from:

```text
monitoring/grafana/dashboards/urban-mlops-dashboard.json
```

It contains:

- Traffic evolution
- p95 latency
- Error rate
- Accuracy drop vs baseline
- Drift score
- Accuracy and confidence against baselines
- Missing values and data freshness
- Logged alert events

## Alerts

Prometheus rules are configured in:

```text
monitoring/alert_rules.yml
```

Configured alerts:

- `HighApiLatency`: p95 latency > 1 second
- `HighApiErrorRate`: 5xx errors > 5%
- `AccuracyDegradation`: accuracy drop > 5 points
- `DataDriftDetected`: drift score > 0.30
- `ConfidenceDecrease`: confidence drop > 10 points
- `MissingValuesSpike`: missing ratio > 20%

The API also writes anomaly logs for high latency, errors, drift, confidence decrease, accuracy degradation, missing values, and simulated retraining triggers.

## Simulation Scenarios

Run these commands while `docker compose up --build` is running.

Baseline traffic:

```powershell
python scripts/simulate_monitoring.py baseline --requests 30
```

High traffic:

```powershell
python scripts/simulate_monitoring.py high_traffic --requests 200 --concurrency 25
```

API errors:

```powershell
python scripts/simulate_monitoring.py api_errors --requests 30 --duration-seconds 90
```

High latency:

```powershell
python scripts/simulate_monitoring.py high_latency --requests 30 --duration-seconds 90
```

Model drift and degradation:

```powershell
python scripts/simulate_monitoring.py model_drift --requests 30 --duration-seconds 120
```

Expected observations:

- High traffic increases `api_requests_total` and may increase p95 latency.
- API errors increase 5xx error rate and `model_prediction_errors_total`.
- High latency pushes `api_request_latency_seconds` above alert thresholds.
- Model drift increases `data_drift_score`, lowers `model_accuracy_score`, lowers `model_confidence_score`, and increases `model_accuracy_drop_ratio`.

## Baselines

Baselines are defined in `serving/api.py`:

- Objective 1: accuracy `0.92`, confidence `0.86`
- Objective 2: accuracy `0.9989`, confidence `0.90`
- Objective 3: accuracy `0.667`, confidence `0.72`
- Objective 4: accuracy `0.91`, confidence `0.84`

These values are intentionally explicit so deviations are visible and easy to explain during the demonstration.
