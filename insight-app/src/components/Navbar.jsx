import { Link, useNavigate } from 'react-router-dom';
import Logo from './Logo';
import { useLanguage } from '../context/LanguageContext';

export default function Navbar({ user, onLogout }) {
  const navigate = useNavigate();
  const { language, switchLanguage, t } = useLanguage();

  const handleLogout = () => {
    onLogout();
    navigate('/');
  };

  return (
    <nav className="bg-gradient-to-r from-teal-600 to-cyan-600 shadow-2xl sticky top-0 z-50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo & Brand */}
          <Link 
            to="/dashboard" 
            className="flex items-center space-x-3 hover:opacity-90 transition-all duration-300 group"
          >
            <div className="transform group-hover:scale-110 transition-transform duration-300">
              <Logo size="sm" />
            </div>
            <div className="flex flex-col hidden sm:block">
              <span className="text-lg font-bold text-white">Urban Mobility</span>
              <span className="text-xs text-teal-100">{t('navbar.intelligence')}</span>
            </div>
          </Link>

          {/* User Info, Language & Logout */}
          <div className="flex items-center space-x-4">
            <div className="hidden sm:block text-right">
              <p className="text-sm font-semibold text-white">{user.name}</p>
              <p className="text-xs text-teal-100 capitalize">{user.roleName}</p>
            </div>
            
            {/* Language Selector - Modern Style */}
            <div className="inline-flex items-center bg-white/15 backdrop-blur-md rounded-full p-1 border border-white/20 hover:bg-white/20 transition-all duration-300">
              <button
                onClick={() => switchLanguage('fr')}
                className={`px-4 py-2 rounded-full font-semibold text-sm transition-all duration-300 whitespace-nowrap ${
                  language === 'fr'
                    ? 'bg-white text-teal-600 shadow-lg scale-105'
                    : 'text-white/80 hover:text-white'
                }`}
              >
                <span className="mr-1.5">🇫🇷</span>Français
              </button>
              <button
                onClick={() => switchLanguage('en')}
                className={`px-4 py-2 rounded-full font-semibold text-sm transition-all duration-300 whitespace-nowrap ${
                  language === 'en'
                    ? 'bg-white text-teal-600 shadow-lg scale-105'
                    : 'text-white/80 hover:text-white'
                }`}
              >
                <span className="mr-1.5">🇬🇧</span>English
              </button>
            </div>
            
            {/* Profile Avatar */}
            <div className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center border border-white/30 hover:bg-white/30 transition-all duration-300">
              <span className="text-white font-bold text-sm">{user.name[0].toUpperCase()}</span>
            </div>

            {/* Logout Button */}
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-all duration-300 font-semibold hover:shadow-lg hover:scale-105 active:scale-95 text-sm"
            >
              {t('navbar.logout')}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
