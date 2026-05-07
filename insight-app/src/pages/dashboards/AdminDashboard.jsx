import { useState, useEffect } from 'react';
import { getHealthStatus, API_CONFIG } from '../../services/api';
import { useLanguage } from '../../context/LanguageContext';
import MetricsWidget from '../../components/MetricsWidget';

export default function AdminDashboard({ user }) {
  const { language, t } = useLanguage();
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [grafanaEmbedded, setGrafanaEmbedded] = useState(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        setLoading(true);
        const status = await getHealthStatus();
        setApiStatus(status);
      } catch (error) {
        console.error('API status check failed:', error);
        setApiStatus({ status: 'error', message: error.message });
      } finally {
        setLoading(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <h1 className="text-4xl font-bold text-gray-900 mb-2">
        👋 {language === 'fr' ? 'Bienvenue' : 'Welcome'} {user.name}, {language === 'fr' ? 'Administrateur' : 'Administrator'}
      </h1>
      <p className="text-gray-600 mb-8">
        {language === 'fr' 
          ? 'Gérez les modèles, les métriques et la santé du système Urban Mobility MLOps'
          : 'Manage models, metrics and system health for Urban Mobility MLOps'}
      </p>

      {/* System Health Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-teal-600">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{language === 'fr' ? 'API' : 'API'}</p>
              <p className="text-2xl font-bold text-gray-900">
                {loading ? '...' : apiStatus?.status === 'ok' ? '✓' : '✗'}
              </p>
            </div>
            <div className="text-3xl">🔌</div>
          </div>
          <p className="text-xs text-gray-500 mt-2">{API_CONFIG.baseURL}</p>
        </div>

        <MetricsWidget
          title={language === 'fr' ? 'Prédictions Totales' : 'Total Predictions'}
          query="increase(model_predictions_total[1h])"
          format="integer"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Taux d\'Erreur' : 'Error Rate'}
          query="increase(model_prediction_errors_total[5m])"
          format="integer"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Latence p95' : 'p95 Latency'}
          query="histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))"
          format="decimal2"
          language={language}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Real-time Metrics */}
        <div className="bg-white rounded-xl shadow-lg p-8 border-l-4 border-blue-600">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">📊 {language === 'fr' ? 'Métriques en Temps Réel' : 'Real-time Metrics'}</h2>
          
          <div className="space-y-4">
            <MetricsWidget
              title={language === 'fr' ? 'Requêtes API (5m)' : 'API Requests (5m)'}
              query="increase(api_requests_total[5m])"
              format="integer"
              language={language}
            />
            
            <MetricsWidget
              title={language === 'fr' ? 'Latence Moyenne' : 'Avg Latency'}
              query="avg(rate(api_request_latency_seconds_sum[5m])) / avg(rate(api_request_latency_seconds_count[5m]))"
              format="decimal3"
              language={language}
            />

            <MetricsWidget
              title={language === 'fr' ? 'Taux de Succès' : 'Success Rate'}
              query='(1 - (increase(api_requests_total{status=~"5.."}[5m]) / increase(api_requests_total[5m]))) * 100'
              format="percentage"
              language={language}
            />
          </div>
        </div>

        {/* Grafana Monitoring */}
        <div className="bg-white rounded-xl shadow-lg p-8 border-l-4 border-orange-600">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">📈 Grafana MLOps Monitoring</h2>
            <button
              onClick={() => setGrafanaEmbedded(!grafanaEmbedded)}
              className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition text-sm"
            >
              {grafanaEmbedded ? (language === 'fr' ? 'Réduire' : 'Minimize') : (language === 'fr' ? 'Agrandir' : 'Expand')}
            </button>
          </div>

          <p className="text-gray-600 mb-6">
            {language === 'fr'
              ? 'Suivez en temps réel les performances des modèles, l\'accuracité, et la dérive des données.'
              : 'Monitor model performance, accuracy, and data drift in real-time.'}
          </p>

          {grafanaEmbedded ? (
            <iframe
              src={`${API_CONFIG.grafanaURL}/d/urban-mlops-dashboard?kiosk=tv`}
              width="100%"
              height="600"
              frameBorder="0"
              className="rounded-lg"
              title="Grafana Dashboard"
            />
          ) : (
            <a
              href={API_CONFIG.grafanaURL}
              target="_blank"
              rel="noopener noreferrer"
              className="block w-full p-8 bg-gradient-to-r from-orange-50 to-red-50 rounded-lg border-2 border-orange-200 text-center hover:shadow-lg transition"
            >
              <p className="text-orange-600 font-semibold mb-2">🔗 {language === 'fr' ? 'Ouvrir Grafana' : 'Open Grafana'}</p>
              <p className="text-sm text-gray-600">{API_CONFIG.grafanaURL}</p>
            </a>
          )}
        </div>
      </div>

      {/* Model Objectives Overview */}
      <div className="bg-white rounded-xl shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">🎯 {language === 'fr' ? 'État des Objectifs' : 'Objectives Status'}</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { id: 1, name: language === 'fr' ? 'Forecast Trafic' : 'Traffic Forecast', icon: '🚗', color: 'red' },
            { id: 2, name: language === 'fr' ? 'Recommandation Trajets' : 'Route Recommendations', icon: '🗺️', color: 'yellow' },
            { id: 3, name: language === 'fr' ? 'Classification Risques' : 'Risk Classification', icon: '🏛️', color: 'blue' },
            { id: 4, name: language === 'fr' ? 'Estimation CO2' : 'CO2 Estimation', icon: '🌍', color: 'green' },
          ].map((obj) => (
            <a
              key={obj.id}
              href={`#objective${obj.id}`}
              className={`p-6 border-l-4 border-${obj.color}-600 bg-${obj.color}-50 rounded-lg hover:shadow-lg transition cursor-pointer`}
            >
              <div className="text-3xl mb-2">{obj.icon}</div>
              <p className="font-semibold text-gray-900">{obj.name}</p>
              <p className="text-xs text-gray-600 mt-2">{language === 'fr' ? 'Objectif' : 'Objective'} {obj.id}</p>
            </a>
          ))}
        </div>
      </div>

      {/* Quick Links */}
      <div className="mt-8 bg-gradient-to-r from-teal-50 to-cyan-50 rounded-xl p-8">
        <h3 className="text-lg font-bold text-gray-900 mb-4">🔗 {language === 'fr' ? 'Accès Rapide' : 'Quick Links'}</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href={`${API_CONFIG.prometheusURL}`}
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 bg-white rounded-lg hover:shadow-lg transition"
          >
            <p className="font-semibold text-gray-900">📊 Prometheus</p>
            <p className="text-xs text-gray-600">{API_CONFIG.prometheusURL}</p>
          </a>

          <a
            href={`${API_CONFIG.grafanaURL}`}
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 bg-white rounded-lg hover:shadow-lg transition"
          >
            <p className="font-semibold text-gray-900">📈 Grafana</p>
            <p className="text-xs text-gray-600">{API_CONFIG.grafanaURL}</p>
          </a>

          <a
            href={`${API_CONFIG.baseURL}/docs`}
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 bg-white rounded-lg hover:shadow-lg transition"
          >
            <p className="font-semibold text-gray-900">📚 API Docs</p>
            <p className="text-xs text-gray-600">{API_CONFIG.baseURL}/docs</p>
          </a>
        </div>
      </div>
    </div>
  );
}
