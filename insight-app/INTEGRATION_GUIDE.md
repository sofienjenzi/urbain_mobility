# 🚀 Urban Mobility Intelligence Platform - Technical Integration

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (React)                    │
│              (http://localhost:3001)                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │  API Service Layer (src/services/api.js)            │   │
│  │  - predictObjective()                               │   │
│  │  - getPrometheusMetrics()                           │   │
│  │  - getHealthStatus()                                │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──────────────────────────┬──────────────────────┐
             │                          │                      │
    ┌────────▼────────┐    ┌───────────▼────────┐  ┌──────────▼───────┐
    │  FastAPI (8000) │    │ Prometheus (9090)  │  │ Grafana (3000)   │
    │  ML Predictions │    │ Metrics Scraping   │  │ Dashboard Views  │
    │  Real-time      │    │ Time-series Data   │  │ Visualizations   │
    └────────┬────────┘    └────────────────────┘  └──────────────────┘
             │
    ┌────────▼────────┐
    │   MLOps Models  │
    │  (Objective 1-4)│
    │  Model Registry │
    └─────────────────┘
```

## Running the Complete System

### 1. Start FastAPI Backend & Monitoring Stack

```powershell
# From project root
cd c:\Users\user\Downloads\colab_notebooks

# Start all services (API, Prometheus, Grafana) with Docker
docker compose up --build

# Services will be available at:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### 2. Start React Frontend

```powershell
# From insight-app directory
cd insight-app

# Install dependencies (if needed)
npm install

# Start development server on port 3001
npm run dev
```

### 3. Access the Platform

- **Home Page**: http://localhost:3001
- **Login**: http://localhost:3001/login
- **Admin Dashboard**: http://localhost:3001/dashboard/admin
- **Transport Dashboard (Objective 1)**: http://localhost:3001/dashboard/transport
- **Citizen Dashboard (Objective 2)**: http://localhost:3001/dashboard/citizen
- **Minister Dashboard (Objective 3)**: http://localhost:3001/dashboard/minister
- **Air Parif Dashboard (Objective 4)**: http://localhost:3001/dashboard/air_parif

## Component Architecture

### Services Layer

**`src/services/api.js`** - Core API communication module

```javascript
// Make predictions
const result = await predictObjective('1', { steps: 12 });

// Get real-time metrics
const metrics = await getPrometheusMetrics('model_accuracy_score{objective="1"}');

// Get time-series data
const timeSeries = await getPrometheusTimeSeries(query, startTime, endTime);

// Get API health
const status = await getHealthStatus();
```

### Components

**`src/components/MetricsWidget.jsx`** - Real-time metrics display

- Fetches data from Prometheus every 10 seconds
- Supports multiple formats (percentage, decimal, integer)
- Bi-lingual support (FR/EN)
- Error handling & loading states

### Dashboards

Each objective has a specialized dashboard:

#### Objective 1: Traffic Forecast (Transport)
- **Input**: Time horizon (steps) in months
- **Output**: Traffic predictions and trend
- **Metrics**: Predictions count, accuracy, confidence

#### Objective 2: Route Recommendations (Citizen)
- **Input**: Speed, travel time, congestion index
- **Output**: Recommended route with quality score
- **Metrics**: Predictions count, accuracy, confidence

#### Objective 3: Risk Classification (Minister)
- **Input**: Traffic volume, rate, zone, date
- **Output**: Risk class and probability
- **Metrics**: Predictions count, accuracy, confidence

#### Objective 4: CO2 Estimation (Air Parif)
- **Input**: Activity, emission factor, waste, energy, date
- **Output**: CO2 emissions and energy consumption
- **Metrics**: Predictions count, accuracy, confidence

## API Endpoints

### Prediction Endpoints

```bash
# Objective 1 - Traffic Forecast
POST http://localhost:8000/predict/1
Content-Type: application/json
{
  "input": {
    "steps": 12
  }
}

# Objective 2 - Route Recommendations
POST http://localhost:8000/predict/2
{
  "input": {
    "vitesse": 50,
    "temps_trajet_min": 30,
    "congestion_index": 3
  }
}

# Objective 3 - Risk Classification
POST http://localhost:8000/predict/3
{
  "input": {
    "volume": 50000,
    "taux_1000": 500,
    "year": 2024,
    "month": 6,
    "zone_id": 50000
  }
}

# Objective 4 - CO2 Estimation
POST http://localhost:8000/predict/4
{
  "input": {
    "activity_value": 50000,
    "emission_factor": 10,
    "dechets_kg": 25000,
    "station_kwh": 50000,
    "year": 2024,
    "month": 6
  }
}
```

### Health & Metrics

```bash
# API health check
GET http://localhost:8000/health

# Prometheus metrics (for scraping)
GET http://localhost:8000/metrics

# Available models registry
GET http://localhost:8000/models
```

## Prometheus Queries

All available metrics from the API:

```promql
# Traffic & Predictions
api_requests_total
model_predictions_total{objective="X"}
model_prediction_errors_total{objective="X"}

# Model Health
model_accuracy_score{objective="X"}
model_baseline_accuracy_score{objective="X"}
model_accuracy_drop_ratio{objective="X"}
model_confidence_score{objective="X"}

# Data Health
data_missing_ratio{objective="X"}
data_freshness_seconds{objective="X"}
data_drift_score{objective="X"}

# Latency & Performance
api_request_latency_seconds
histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))

# Alerts
monitoring_alert_events_total{objective="X", alert_type="X"}
```

## Environment Configuration

The React app uses these environment variables:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_PROMETHEUS_URL=http://localhost:9090
REACT_APP_GRAFANA_URL=http://localhost:3000
```

These can be overridden by creating a `.env` file in `insight-app/` directory.

## Multi-Language Support

All components support FR/EN through the `LanguageContext`:

```javascript
import { useLanguage } from '../context/LanguageContext';

// In component
const { language, switchLanguage, t } = useLanguage();

// Switch language
switchLanguage('fr'); // French
switchLanguage('en'); // English

// Translate text
t('navbar.login') // Returns translated text
```

## Testing the Integration

### 1. Test API Connectivity

```bash
curl http://localhost:8000/health
# Should return: {"status": "ok"}
```

### 2. Test a Prediction

```bash
curl -X POST http://localhost:8000/predict/1 \
  -H "Content-Type: application/json" \
  -d '{"input": {"steps": 12}}'
```

### 3. Test Metrics

Open browser to: http://localhost:9090

Query: `model_predictions_total`

### 4. Test Grafana Dashboard

Open browser to: http://localhost:3000

- Login: admin / admin
- Dashboard: Urban MLOps Dashboard

## Simulation Scenarios

Test monitoring with simulated scenarios:

```powershell
# High latency
python scripts/simulate_monitoring.py high_latency --requests 30 --duration-seconds 90

# API errors
python scripts/simulate_monitoring.py api_errors --requests 30 --duration-seconds 90

# Model drift
python scripts/simulate_monitoring.py model_drift --requests 30 --duration-seconds 120

# Baseline traffic
python scripts/simulate_monitoring.py baseline --requests 30
```

## Troubleshooting

### React can't connect to API

```
Error: Failed to fetch from http://localhost:8000/health
```

**Solution**: 
- Check if FastAPI is running: `curl http://localhost:8000/health`
- Verify `.env` file has correct `REACT_APP_API_URL`
- Check CORS settings in FastAPI (should be enabled)

### Prometheus has no data

**Solution**:
- Make some predictions to generate metrics
- Wait 10 seconds for Prometheus to scrape
- Check Prometheus targets: http://localhost:9090/targets

### Grafana dashboard not loading

**Solution**:
- Ensure Prometheus is connected as data source
- Dashboard UID should be: `urban-mlops-dashboard`
- Manual import available at: `monitoring/grafana/dashboards/urban-mlops-dashboard.json`

## Performance Notes

- React app runs on **port 3001** (not 3000 to avoid Grafana conflict)
- API refreshes metrics from Prometheus **every 10 seconds**
- Prometheus scrapes API metrics **every 10 seconds**
- Grafana dashboard updates in **real-time** via Prometheus

## Future Enhancements

- [ ] WebSocket support for real-time metrics
- [ ] Advanced data visualization (D3.js, Echarts)
- [ ] Historical prediction tracking
- [ ] Custom alert rules UI
- [ ] Export predictions to CSV/Excel
- [ ] Mobile responsive improvements
