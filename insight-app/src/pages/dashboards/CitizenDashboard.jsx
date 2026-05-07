import { useState } from 'react';
import { predictObjective, OBJECTIVES } from '../../services/api';
import { useLanguage } from '../../context/LanguageContext';
import MetricsWidget from '../../components/MetricsWidget';

export default function CitizenDashboard({ user }) {
  const { language, t } = useLanguage();
  const [speed, setSpeed] = useState(50);
  const [trajetTime, setTrajetTime] = useState(30);
  const [congestion, setCongestion] = useState(3);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleRecommend = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);

      const result = await predictObjective('2', {
        vitesse: speed,
        temps_trajet_min: trajetTime,
        congestion_index: congestion,
      });
      setPrediction(result);
    } catch (err) {
      setError(language === 'fr' ? `Erreur: ${err.message}` : `Error: ${err.message}`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const obj = OBJECTIVES[2];

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          👤 {language === 'fr' ? 'Citoyen' : 'Citizen'}
        </h1>
        <p className="text-lg text-gray-600">
          {language === 'fr' ? 'Recommandation de Trajets Intelligents' : 'Smart Route Recommendations'}
        </p>
      </div>

      {/* Real-time Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <MetricsWidget
          title={language === 'fr' ? 'Prédictions' : 'Predictions'}
          query='increase(model_predictions_total{objective="2"}[1h])'
          format="integer"
          objective="2"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Précision' : 'Accuracy'}
          query='model_accuracy_score{objective="2"}'
          format="percentage"
          objective="2"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Confiance' : 'Confidence'}
          query='model_confidence_score{objective="2"}'
          format="percentage"
          objective="2"
          language={language}
        />
      </div>

      {/* Objective 2 - Route Recommendations */}
      <div className="bg-white rounded-xl shadow-lg p-8 border-l-4 border-yellow-600 mb-8">
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

        {/* Recommendation Form */}
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border border-yellow-200 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            {language === 'fr' ? '🗺️ Recommandation de Trajet' : '🗺️ Route Recommendation'}
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Vitesse (km/h)' : 'Speed (km/h)'}
              </label>
              <input
                type="range"
                min="1"
                max="140"
                value={speed}
                onChange={(e) => setSpeed(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{speed} km/h</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Temps de Trajet (min)' : 'Travel Time (min)'}
              </label>
              <input
                type="range"
                min="1"
                max="300"
                value={trajetTime}
                onChange={(e) => setTrajetTime(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{trajetTime} min</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Index de Congestion' : 'Congestion Index'}
              </label>
              <input
                type="range"
                min="0"
                max="9"
                value={congestion}
                onChange={(e) => setCongestion(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{congestion}/9</p>
            </div>
          </div>

          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          <button
            onClick={handleRecommend}
            disabled={loading}
            className={`w-full px-6 py-3 rounded-lg font-semibold transition ${
              loading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-yellow-600 to-orange-600 text-white hover:shadow-lg'
            }`}
          >
            {loading ? (
              <>
                <span className="inline-block animate-spin mr-2">⏳</span>
                {language === 'fr' ? 'Calcul en cours...' : 'Computing...'}
              </>
            ) : (
              language === 'fr' ? '🚀 Obtenir une Recommandation' : '🚀 Get Recommendation'
            )}
          </button>
        </div>

        {/* Recommendation Results */}
        {prediction && (
          <div className="bg-green-50 rounded-lg p-6 border border-green-200">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              {language === 'fr' ? '✅ Résultats de la Recommandation' : '✅ Recommendation Results'}
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

              {prediction.recommended_route && (
                <div className="col-span-full bg-white rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-2">{language === 'fr' ? 'Trajet Recommandé' : 'Recommended Route'}</p>
                  <div className="bg-gray-50 rounded p-3 font-mono text-sm max-h-40 overflow-y-auto">
                    {typeof prediction.recommended_route === 'object'
                      ? JSON.stringify(prediction.recommended_route, null, 2)
                      : prediction.recommended_route}
                  </div>
                </div>
              )}

              {prediction.quality_score !== undefined && (
                <div className="col-span-full bg-white rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-2">{language === 'fr' ? 'Score de Qualité' : 'Quality Score'}</p>
                  <p className="text-2xl font-bold text-yellow-600">{(prediction.quality_score * 10).toFixed(1)}/10</p>
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
              query='data_freshness_seconds{objective="2"}'
              format="integer"
              language={language}
            />
            <MetricsWidget
              title={language === 'fr' ? 'Dérive Détectée' : 'Data Drift'}
              query='data_drift_score{objective="2"}'
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
              ? 'Cet objectif fournit des recommandations de trajets optimisés basées sur les conditions de trafic actuelles.'
              : 'This objective provides optimized route recommendations based on current traffic conditions.'}
          </p>
          <div className="bg-purple-50 rounded p-3">
            <p className="text-xs font-semibold text-purple-900 mb-1">{language === 'fr' ? 'Version du Modèle' : 'Model Version'}</p>
            <p className="text-sm text-purple-700 font-mono">objective2-v1.0</p>
          </div>
        </div>
      </div>
    </div>
  );
}
