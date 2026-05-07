import { useState } from 'react';
import { predictObjective, OBJECTIVES } from '../../services/api';
import { useLanguage } from '../../context/LanguageContext';
import MetricsWidget from '../../components/MetricsWidget';

export default function MinisterDashboard({ user }) {
  const { language, t } = useLanguage();
  const [volume, setVolume] = useState(50000);
  const [taux1000, setTaux1000] = useState(500);
  const [year, setYear] = useState(2024);
  const [month, setMonth] = useState(6);
  const [zoneId, setZoneId] = useState(50000);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleClassify = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);

      const result = await predictObjective('3', {
        volume,
        taux_1000: taux1000,
        year,
        month,
        zone_id: zoneId,
      });
      setPrediction(result);
    } catch (err) {
      setError(language === 'fr' ? `Erreur: ${err.message}` : `Error: ${err.message}`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const obj = OBJECTIVES[3];

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          🏛️ {language === 'fr' ? 'Ministre de l\'Intérieur' : 'Interior Minister'}
        </h1>
        <p className="text-lg text-gray-600">
          {language === 'fr' ? 'Classification et Analyse des Risques' : 'Risk Classification & Analysis'}
        </p>
      </div>

      {/* Real-time Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <MetricsWidget
          title={language === 'fr' ? 'Prédictions' : 'Predictions'}
          query='increase(model_predictions_total{objective="3"}[1h])'
          format="integer"
          objective="3"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Précision' : 'Accuracy'}
          query='model_accuracy_score{objective="3"}'
          format="percentage"
          objective="3"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Confiance' : 'Confidence'}
          query='model_confidence_score{objective="3"}'
          format="percentage"
          objective="3"
          language={language}
        />
      </div>

      {/* Objective 3 - Risk Classification */}
      <div className="bg-white rounded-xl shadow-lg p-8 border-l-4 border-blue-600 mb-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {obj.icon} {language === 'fr' ? obj.name_fr : obj.name}
            </h2>
            <p className="text-gray-600">
              {language === 'fr' ? obj.description_fr : obj.description}
            </p>
          </div>
        </div>

        {/* Classification Form */}
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg p-6 border border-blue-200 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            {language === 'fr' ? '🔍 Analyser les Risques' : '🔍 Analyze Risk'}
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Volume de Trafic' : 'Traffic Volume'}
              </label>
              <input
                type="range"
                min="0"
                max="100000"
                value={volume}
                onChange={(e) => setVolume(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{volume.toLocaleString()}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Taux (par 1000)' : 'Rate (per 1000)'}
              </label>
              <input
                type="range"
                min="0"
                max="1000"
                value={taux1000}
                onChange={(e) => setTaux1000(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{taux1000}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Zone ID' : 'Zone ID'}
              </label>
              <input
                type="range"
                min="0"
                max="100000"
                value={zoneId}
                onChange={(e) => setZoneId(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{zoneId}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Année' : 'Year'}
              </label>
              <input
                type="range"
                min="2020"
                max="2035"
                value={year}
                onChange={(e) => setYear(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{year}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Mois' : 'Month'}
              </label>
              <input
                type="range"
                min="1"
                max="12"
                value={month}
                onChange={(e) => setMonth(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{month}</p>
            </div>
          </div>

          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          <button
            onClick={handleClassify}
            disabled={loading}
            className={`w-full px-6 py-3 rounded-lg font-semibold transition ${
              loading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:shadow-lg'
            }`}
          >
            {loading ? (
              <>
                <span className="inline-block animate-spin mr-2">⏳</span>
                {language === 'fr' ? 'Analyse en cours...' : 'Analyzing...'}
              </>
            ) : (
              language === 'fr' ? '🚀 Classifier les Risques' : '🚀 Classify Risk'
            )}
          </button>
        </div>

        {/* Classification Results */}
        {prediction && (
          <div className="bg-green-50 rounded-lg p-6 border border-green-200">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              {language === 'fr' ? '✅ Résultats de la Classification' : '✅ Classification Results'}
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">{language === 'fr' ? 'Modèle' : 'Model'}</p>
                <p className="text-lg font-bold text-gray-900">v{prediction.model_version || 'Unknown'}</p>
              </div>

              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <p className="text-sm text-gray-600 mb-1">{language === 'fr' ? 'Latence' : 'Latency'}</p>
                <p className="text-lg font-bold text-gray-900">{prediction.duration_ms?.toFixed(2)}ms</p>
              </div>

              {prediction.risk_class && (
                <div className="col-span-full bg-white rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-2">{language === 'fr' ? 'Classe de Risque' : 'Risk Class'}</p>
                  <p className="text-2xl font-bold text-blue-600">{prediction.risk_class}</p>
                </div>
              )}

              {prediction.probability !== undefined && (
                <div className="col-span-full bg-white rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-2">{language === 'fr' ? 'Probabilité' : 'Probability'}</p>
                  <p className="text-2xl font-bold text-blue-600">{(prediction.probability * 100).toFixed(1)}%</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Data Health */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-600">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            📊 {language === 'fr' ? 'Santé des Données' : 'Data Health'}
          </h3>
          <div className="space-y-3">
            <MetricsWidget
              title={language === 'fr' ? 'Fraîcheur des Données' : 'Data Freshness'}
              query='data_freshness_seconds{objective="3"}'
              format="integer"
              language={language}
            />
            <MetricsWidget
              title={language === 'fr' ? 'Dérive Détectée' : 'Data Drift'}
              query='data_drift_score{objective="3"}'
              format="decimal2"
              language={language}
            />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-purple-600">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            ℹ️ {language === 'fr' ? 'À Propos' : 'About'}
          </h3>
          <p className="text-gray-600 text-sm mb-4">
            {language === 'fr'
              ? 'Cet objectif classifie les risques de sécurité routière basés sur les paramètres de trafic et les conditions de zone.'
              : 'This objective classifies road safety risks based on traffic parameters and zone conditions.'}
          </p>
          <div className="bg-purple-50 rounded p-3">
            <p className="text-xs font-semibold text-purple-900 mb-1">{language === 'fr' ? 'Version du Modèle' : 'Model Version'}</p>
            <p className="text-sm text-purple-700 font-mono">objective3-v1.0</p>
          </div>
        </div>
      </div>
    </div>
  );
}
