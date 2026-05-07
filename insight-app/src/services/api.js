/**
 * Urban MLOps API Service
 * Communicates with FastAPI backend at http://localhost:8000
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const PROMETHEUS_URL = import.meta.env.VITE_PROMETHEUS_URL || 'http://localhost:9090';
const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3000';

export const API_CONFIG = {
  baseURL: API_BASE_URL,
  prometheusURL: PROMETHEUS_URL,
  grafanaURL: GRAFANA_URL,
};

/**
 * Make a prediction request to the FastAPI backend
 * @param {string} objective - Objective number (1, 2, 3, or 4)
 * @param {object} input - Input data for prediction
 * @returns {Promise<object>} Prediction result
 */
export async function predictObjective(objective, input) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/${objective}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Prediction failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Prediction error for objective ${objective}:`, error);
    throw error;
  }
}

/**
 * Get available models for all objectives
 * @returns {Promise<object>} Models registry
 */
export async function getModels() {
  try {
    const response = await fetch(`${API_BASE_URL}/models`);
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching models:', error);
    throw error;
  }
}

/**
 * Get API health status
 * @returns {Promise<object>} Health status
 */
export async function getHealthStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`API health check failed: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
}

/**
 * Fetch metrics from Prometheus
 * @param {string} query - Prometheus query
 * @returns {Promise<object>} Query result
 */
export async function getPrometheusMetrics(query) {
  try {
    const encodedQuery = encodeURIComponent(query);
    const response = await fetch(
      `${PROMETHEUS_URL}/api/v1/query?query=${encodedQuery}`
    );

    if (!response.ok) {
      throw new Error(`Prometheus query failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Prometheus query error:', error);
    throw error;
  }
}

/**
 * Fetch historical metrics from Prometheus
 * @param {string} query - Prometheus query
 * @param {number} startTime - Start timestamp
 * @param {number} endTime - End timestamp
 * @param {string} step - Step interval (e.g., '1m', '5m')
 * @returns {Promise<object>} Time series data
 */
export async function getPrometheusTimeSeries(query, startTime, endTime, step = '1m') {
  try {
    const encodedQuery = encodeURIComponent(query);
    const response = await fetch(
      `${PROMETHEUS_URL}/api/v1/query_range?query=${encodedQuery}&start=${startTime}&end=${endTime}&step=${step}`
    );

    if (!response.ok) {
      throw new Error(`Prometheus range query failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Prometheus range query error:', error);
    throw error;
  }
}

/**
 * Get prediction metrics for a specific objective
 * @param {string} objective - Objective number
 * @returns {Promise<object>} Metrics data
 */
export async function getObjectiveMetrics(objective) {
  try {
    const queries = {
      predictions: `model_predictions_total{objective="${objective}"}`,
      accuracy: `model_accuracy_score{objective="${objective}"}`,
      confidence: `model_confidence_score{objective="${objective}"}`,
      errorRate: `increase(model_prediction_errors_total{objective="${objective}"}[5m])`,
      dataFreshness: `data_freshness_seconds{objective="${objective}"}`,
      dataDrift: `data_drift_score{objective="${objective}"}`,
    };

    const results = {};
    for (const [key, query] of Object.entries(queries)) {
      const response = await getPrometheusMetrics(query);
      results[key] = response.data?.result || [];
    }

    return results;
  } catch (error) {
    console.error(`Error fetching metrics for objective ${objective}:`, error);
    throw error;
  }
}

/**
 * Simulate a scenario (high latency, api errors, model drift)
 * @param {string} scenario - Scenario name
 * @param {number} durationSeconds - Duration in seconds
 * @returns {Promise<object>} Simulation result
 */
export async function simulateScenario(scenario, durationSeconds = 120) {
  try {
    const response = await fetch(`${API_BASE_URL}/simulate/${scenario}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ duration_seconds: durationSeconds }),
    });

    if (!response.ok) {
      throw new Error(`Simulation failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Simulation error for ${scenario}:`, error);
    throw error;
  }
}

/**
 * Objectives metadata
 */
export const OBJECTIVES = {
  1: {
    id: 1,
    name: 'Traffic Forecast',
    name_fr: 'Forecast du Trafic',
    description: 'Predict urban traffic volumes',
    description_fr: 'Prédire les volumes de trafic urbain',
    color: 'from-red-500 to-pink-500',
    icon: '🚗',
  },
  2: {
    id: 2,
    name: 'Route Recommendations',
    name_fr: 'Recommandation de Trajets',
    description: 'Recommend optimal routes',
    description_fr: 'Recommander les trajets optimaux',
    color: 'from-yellow-500 to-orange-500',
    icon: '🗺️',
  },
  3: {
    id: 3,
    name: 'Risk Classification',
    name_fr: 'Classification des Risques',
    description: 'Classify security risks',
    description_fr: 'Classifier les risques de sécurité',
    color: 'from-blue-500 to-cyan-500',
    icon: '🏛️',
  },
  4: {
    id: 4,
    name: 'CO2 Estimation',
    name_fr: 'Estimation CO2 & Énergie',
    description: 'Estimate carbon emissions',
    description_fr: 'Estimer les émissions CO2',
    color: 'from-green-500 to-emerald-500',
    icon: '🌍',
  },
};

/**
 * Get objective by ID
 * @param {number|string} id - Objective ID
 * @param {string} language - Language code ('en' or 'fr')
 * @returns {object} Objective metadata
 */
export function getObjectiveMetadata(id, language = 'en') {
  const objective = OBJECTIVES[id];
  if (!objective) return null;

  return {
    ...objective,
    displayName: language === 'fr' ? objective.name_fr : objective.name,
    displayDescription: language === 'fr' ? objective.description_fr : objective.description,
  };
}

export default {
  API_CONFIG,
  predictObjective,
  getModels,
  getHealthStatus,
  getPrometheusMetrics,
  getPrometheusTimeSeries,
  getObjectiveMetrics,
  simulateScenario,
  OBJECTIVES,
  getObjectiveMetadata,
};
