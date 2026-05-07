import { useState } from 'react';
import { predictObjective, OBJECTIVES } from '../../services/api';
import { useLanguage } from '../../context/LanguageContext';
import MetricsWidget from '../../components/MetricsWidget';

export default function AirParifDashboard({ user }) {
  const { language, t } = useLanguage();
  const [activity, setActivity] = useState(50000);
  const [emissionFactor, setEmissionFactor] = useState(10);
  const [dechetKg, setDechetKg] = useState(25000);
  const [stationKwh, setStationKwh] = useState(50000);
  const [year, setYear] = useState(2024);
  const [month, setMonth] = useState(6);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleEstimate = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);

      const result = await predictObjective('4', {
        activity_value: activity,
        emission_factor: emissionFactor,
        dechets_kg: dechetKg,
        station_kwh: stationKwh,
        year,
        month,
      });
      setPrediction(result);
    } catch (err) {
      setError(language === 'fr' ? `Erreur: ${err.message}` : `Error: ${err.message}`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const obj = OBJECTIVES[4];

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          🌍 Air Parif
        </h1>
        <p className="text-lg text-gray-600">
          {language === 'fr' ? 'Estimation CO2 & Énergie' : 'CO2 & Energy Estimation'}
        </p>
      </div>

      {/* Real-time Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <MetricsWidget
          title={language === 'fr' ? 'Prédictions' : 'Predictions'}
          query='increase(model_predictions_total{objective="4"}[1h])'
          format="integer"
          objective="4"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Précision' : 'Accuracy'}
          query='model_accuracy_score{objective="4"}'
          format="percentage"
          objective="4"
          language={language}
        />

        <MetricsWidget
          title={language === 'fr' ? 'Confiance' : 'Confidence'}
          query='model_confidence_score{objective="4"}'
          format="percentage"
          objective="4"
          language={language}
        />
      </div>

      {/* Objective 4 - CO2 & Energy Estimation */}
      <div className="bg-white rounded-xl shadow-lg p-8 border-l-4 border-green-600 mb-8">
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

        {/* Estimation Form */}
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-6 border border-green-200 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            {language === 'fr' ? '🌱 Estimer l\'Impact Environnemental' : '🌱 Estimate Environmental Impact'}
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Activité' : 'Activity Value'}
              </label>
              <input
                type="range"
                min="0"
                max="100000"
                value={activity}
                onChange={(e) => setActivity(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{activity.toLocaleString()}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Facteur d\'Émission' : 'Emission Factor'}
              </label>
              <input
                type="range"
                min="0"
                max="20"
                step="0.1"
                value={emissionFactor}
                onChange={(e) => setEmissionFactor(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{emissionFactor.toFixed(1)}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Déchets (kg)' : 'Waste (kg)'}
              </label>
              <input
                type="range"
                min="0"
                max="50000"
                value={dechetKg}
                onChange={(e) => setDechetKg(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{dechetKg.toLocaleString()} kg</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {language === 'fr' ? 'Énergie Gare (kWh)' : 'Station Energy (kWh)'}
              </label>
              <input
                type="range"
                min="0"
                max="100000"
                value={stationKwh}
                onChange={(e) => setStationKwh(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-600 mt-2 font-semibold">{stationKwh.toLocaleString()} kWh</p>
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
            onClick={handleEstimate}
            disabled={loading}
            className={`w-full px-6 py-3 rounded-lg font-semibold transition ${
              loading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:shadow-lg'
            }`}
          >
            {loading ? (
              <>
                <span className="inline-block animate-spin mr-2">⏳</span>
                {language === 'fr' ? 'Calcul en cours...' : 'Computing...'}
              </>
            ) : (
              language === 'fr' ? '🚀 Estimer l\'Impact' : '🚀 Estimate Impact'
            )}
          </button>
        </div>

        {/* Estimation Results */}
        {prediction && (
          <div className="bg-green-50 rounded-lg p-6 border border-green-200">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">
              {language === 'fr' ? '✅ Résultats de l\'Estimation' : '✅ Estimation Results'}
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

              {prediction.co2_kg !== undefined && (
                <div className="col-span-full bg-white rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-2">🌍 {language === 'fr' ? 'Émissions CO2' : 'CO2 Emissions'}</p>
                  <p className="text-2xl font-bold text-green-600">{prediction.co2_kg.toFixed(2)} kg CO₂</p>
                </div>
              )}

              {prediction.energy_kwh !== undefined && (
                <div className="col-span-full bg-white rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-2">⚡ {language === 'fr' ? 'Consommation Énergétique' : 'Energy Consumption'}</p>
                  <p className="text-2xl font-bold text-green-600">{prediction.energy_kwh.toFixed(2)} kWh</p>
                </div>
              )}

              {prediction.environmental_impact && (
                <div className="col-span-full bg-white rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-2">{language === 'fr' ? 'Impact Environnemental' : 'Environmental Impact'}</p>
                  <div className="bg-gray-50 rounded p-3 font-mono text-sm max-h-40 overflow-y-auto">
                    {typeof prediction.environmental_impact === 'object'
                      ? JSON.stringify(prediction.environmental_impact, null, 2)
                      : prediction.environmental_impact}
                  </div>
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
              query='data_freshness_seconds{objective="4"}'
              format="integer"
              language={language}
            />
            <MetricsWidget
              title={language === 'fr' ? 'Dérive Détectée' : 'Data Drift'}
              query='data_drift_score{objective="4"}'
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
              ? 'Cet objectif estime l\'impact environnemental des transports urbains en calculant les émissions CO2 et la consommation énergétique.'
              : 'This objective estimates the environmental impact of urban transportation by calculating CO2 emissions and energy consumption.'}
          </p>
          <div className="bg-purple-50 rounded p-3">
            <p className="text-xs font-semibold text-purple-900 mb-1">{language === 'fr' ? 'Version du Modèle' : 'Model Version'}</p>
            <p className="text-sm text-purple-700 font-mono">objective4-v1.0</p>
          </div>
        </div>
      </div>
    </div>
  );
}
