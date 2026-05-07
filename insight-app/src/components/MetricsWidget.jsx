import { useState, useEffect } from 'react';
import { getPrometheusMetrics } from '../services/api';

export default function MetricsWidget({ title, query, objective, format = 'number', language = 'en' }) {
  const [value, setValue] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await getPrometheusMetrics(query);
        
        if (data.data?.result?.length > 0) {
          const result = data.data.result[0];
          const rawValue = parseFloat(result.value[1]);
          setValue(rawValue);
          setLastUpdate(new Date());
        } else {
          setValue(null);
          setError(language === 'fr' ? 'Pas de données' : 'No data');
        }
      } catch (err) {
        setError(language === 'fr' ? 'Erreur de chargement' : 'Load error');
        console.error('Metrics fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000); // Refresh every 10 seconds

    return () => clearInterval(interval);
  }, [query, language]);

  const formatValue = (val) => {
    if (val === null) return '—';
    switch (format) {
      case 'percentage':
        return `${(val * 100).toFixed(2)}%`;
      case 'decimal2':
        return val.toFixed(2);
      case 'decimal1':
        return val.toFixed(1);
      case 'integer':
        return Math.round(val).toString();
      default:
        return val.toFixed(2);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
        {lastUpdate && (
          <span className="text-xs text-gray-500">
            {lastUpdate.toLocaleTimeString(language === 'fr' ? 'fr-FR' : 'en-US')}
          </span>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-12">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-teal-600"></div>
        </div>
      ) : error ? (
        <div className="text-red-500 text-sm">{error}</div>
      ) : (
        <div className="text-3xl font-bold text-teal-600">{formatValue(value)}</div>
      )}

      <div className="mt-4 text-xs text-gray-500">
        {objective && `Objective ${objective}`}
      </div>
    </div>
  );
}
