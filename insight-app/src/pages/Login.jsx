import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useLanguage } from '../context/LanguageContext';
import Logo from '../components/Logo';

const ROLES = [
  {
    id: 'admin',
    name: 'Administrateur',
    name_en: 'Administrator',
    description: 'Power BI & Grafana MLOps',
    description_en: 'Power BI & Grafana MLOps',
    icon: '⚙️',
    color: 'from-purple-500 to-pink-500',
  },
  {
    id: 'minister',
    name: 'Ministre de l\'Intérieur',
    name_en: 'Interior Minister',
    description: 'Objectif 3 - Classification des Risques',
    description_en: 'Objective 3 - Risk Classification',
    icon: '🏛️',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    id: 'air_parif',
    name: 'Air Parif',
    name_en: 'Air Parif',
    description: 'Objectif 4 - CO2 & Énergie',
    description_en: 'Objective 4 - CO2 & Energy',
    icon: '🌍',
    color: 'from-green-500 to-emerald-500',
  },
  {
    id: 'citizen',
    name: 'Citoyen',
    name_en: 'Citizen',
    description: 'Recommandation de Trajets',
    description_en: 'Route Recommendations',
    icon: '👤',
    color: 'from-yellow-500 to-orange-500',
  },
  {
    id: 'transport',
    name: 'Société de Transport',
    name_en: 'Transport Company',
    description: 'Objectif 1 - Forecast du Trafic',
    description_en: 'Objective 1 - Traffic Forecast',
    icon: '🚌',
    color: 'from-red-500 to-pink-500',
  },
];

export default function Login({ onLogin }) {
  const { language, switchLanguage, t } = useLanguage();
  const [selectedRole, setSelectedRole] = useState(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    if (!selectedRole || !email || !password) {
      alert('Veuillez remplir tous les champs');
      return;
    }

    const role = ROLES.find(r => r.id === selectedRole);
    const userData = {
      email,
      role: selectedRole,
      roleName: role.name,
      name: email.split('@')[0],
    };

    onLogin(userData);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 via-blue-50 to-cyan-50 flex items-center justify-center py-12 px-4 relative">
      {/* Language Selector - Modern Style */}
      <div className="absolute top-6 right-6">
        <div className="inline-flex items-center bg-white/95 backdrop-blur-md rounded-full p-1 border border-gray-200 shadow-lg hover:shadow-xl transition-all duration-300">
          <button
            onClick={() => switchLanguage('fr')}
            className={`px-5 py-2 rounded-full font-semibold text-sm transition-all duration-300 whitespace-nowrap ${
              language === 'fr'
                ? 'bg-gradient-to-r from-teal-600 to-cyan-600 text-white shadow-lg scale-105'
                : 'text-gray-700 hover:text-teal-600 hover:bg-gray-50'
            }`}
          >
            <span className="mr-2">🇫🇷</span>Français
          </button>
          <button
            onClick={() => switchLanguage('en')}
            className={`px-5 py-2 rounded-full font-semibold text-sm transition-all duration-300 whitespace-nowrap ${
              language === 'en'
                ? 'bg-gradient-to-r from-teal-600 to-cyan-600 text-white shadow-lg scale-105'
                : 'text-gray-700 hover:text-teal-600 hover:bg-gray-50'
            }`}
          >
            <span className="mr-2">🇬🇧</span>English
          </button>
        </div>
      </div>

      <div className="w-full max-w-4xl">
        {/* Logo & Header */}
        <div className="text-center mb-12">
          <Link to="/" className="inline-block mb-4 hover:opacity-80 transition">
            <div className="transform hover:scale-110 transition-transform duration-300">
              <Logo size="xl" />
            </div>
          </Link>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Urban Mobility</h1>
          <p className="text-lg text-gray-600">Intelligence Platform</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Role Selection */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">{t('login.title')}</h2>
            <div className="space-y-3">
              {ROLES.map((role) => (
                <button
                  key={role.id}
                  onClick={() => setSelectedRole(role.id)}
                  className={`w-full p-4 rounded-xl text-left transition border-2 ${
                    selectedRole === role.id
                      ? `border-teal-600 bg-teal-50 shadow-md`
                      : 'border-gray-200 hover:border-teal-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center space-x-4">
                    <div className="text-3xl">{role.icon}</div>
                    <div>
                      <div className="font-bold text-gray-900">
                        {language === 'fr' ? role.name : role.name_en}
                      </div>
                      <div className="text-sm text-gray-600">
                        {language === 'fr' ? role.description : role.description_en}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Right: Login Form */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              {language === 'fr' ? 'Connexion' : 'Sign In'}
            </h2>

            {selectedRole && (
              <div className={`mb-6 p-4 rounded-lg bg-gradient-to-r ${ROLES.find(r => r.id === selectedRole).color} text-white`}>
                <p className="font-semibold">
                  {language === 'fr' ? 'Rôle sélectionné :' : 'Selected role:'} <span className="font-bold">{language === 'fr' ? ROLES.find(r => r.id === selectedRole).name : ROLES.find(r => r.id === selectedRole).name_en}</span>
                </p>
              </div>
            )}

            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('login.email')}
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder={language === 'fr' ? 'votre.email@example.com' : 'your.email@example.com'}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-600"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('login.password')}
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-600"
                />
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
                <p className="font-semibold mb-2">🔐 {language === 'fr' ? 'Mode Démo' : 'Demo Mode'}</p>
                <p>{language === 'fr' ? 'Utilisez n\'importe quel email et mot de passe pour tester la plateforme.' : 'Use any email and password to test the platform.'}</p>
              </div>

              <button
                type="submit"
                disabled={!selectedRole}
                className={`w-full py-3 rounded-lg font-semibold text-white transition ${
                  selectedRole
                    ? 'bg-teal-600 hover:bg-teal-700 cursor-pointer'
                    : 'bg-gray-400 cursor-not-allowed'
                }`}
              >
                {t('login.signin')}
              </button>

              <p className="text-center text-sm text-gray-600">
                {language === 'fr' ? 'Aucune authentification réelle requise en mode démo' : 'No real authentication required in demo mode'}
              </p>
            </form>
          </div>
        </div>

        {/* Back to Home Link */}
        <div className="text-center mt-8">
          <Link to="/" className="text-teal-600 hover:text-teal-700 font-semibold">
            ← {t('login.back')}
          </Link>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-gray-600">
          <p>© 2026 Urban Mobility Intelligence - Intelligence Platform</p>
        </div>
      </div>
    </div>
  );
}
