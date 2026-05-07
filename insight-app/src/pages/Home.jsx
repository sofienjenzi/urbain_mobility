import { Link } from 'react-router-dom';
import Logo from '../components/Logo';
import { useLanguage } from '../context/LanguageContext';

export default function Home({ user }) {
  const { language, switchLanguage, t } = useLanguage();
  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 via-blue-50 to-cyan-50">
      {/* Navigation Bar */}
      <nav className="bg-gradient-to-r from-teal-600 to-cyan-600 shadow-2xl sticky top-0 z-40 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center space-x-3 group hover:opacity-90 transition-opacity duration-300">
              <div className="transform group-hover:scale-110 transition-transform duration-300">
                <Logo size="sm" />
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-bold text-white">Urban Mobility</span>
                <span className="text-xs text-teal-100">{t('navbar.intelligence')}</span>
              </div>
            </Link>
            <div className="flex items-center space-x-4">
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
              
              {user ? (
                <Link
                  to="/dashboard"
                  className="px-6 py-2 bg-white text-teal-600 rounded-lg hover:bg-teal-50 transition-all duration-300 font-semibold hover:shadow-lg hover:scale-105 active:scale-95"
                >
                  {t('navbar.dashboard')}
                </Link>
              ) : (
                <Link
                  to="/login"
                  className="px-6 py-2 bg-white text-teal-600 rounded-lg hover:bg-teal-50 transition-all duration-300 font-semibold hover:shadow-lg hover:scale-105 active:scale-95"
                >
                  {t('navbar.login')}
                </Link>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute top-10 left-10 w-72 h-72 bg-teal-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-cyan-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-10 left-1/2 w-72 h-72 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>

        <div className="max-w-6xl mx-auto relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            {/* Left: Text with animations */}
            <div className="animate-fadeInLeft">
              <div className="mb-6 flex items-center space-x-3">
                <div className="transform hover:scale-110 transition-transform duration-300">
                  <Logo size="2xl" />
                </div>
              </div>
              <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6 leading-tight">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-cyan-600">
                  Urban Mobility
                </span>
              </h1>
              <p className="text-xl text-gray-700 mb-8 leading-relaxed animate-fadeInUp" style={{ animationDelay: '0.2s' }}>
                {t('home.subtitle')}
              </p>
              <p className="text-lg text-gray-600 mb-8 animate-fadeInUp" style={{ animationDelay: '0.4s' }}>
                {t('home.description')}
              </p>
              <div className="flex space-x-4 animate-fadeInUp" style={{ animationDelay: '0.6s' }}>
                <Link
                  to="/login"
                  className="px-8 py-4 bg-gradient-to-r from-teal-600 to-cyan-600 text-white rounded-xl hover:shadow-2xl transition-all duration-300 font-bold hover:scale-105 active:scale-95 hover:-translate-y-1"
                >
                  {t('home.cta_platform')}
                </Link>
                <a
                  href="#about"
                  className="px-8 py-4 border-2 border-teal-600 text-teal-600 rounded-xl hover:bg-teal-50 transition-all duration-300 font-bold hover:shadow-lg hover:scale-105 active:scale-95"
                >
                  {t('home.cta_learn')}
                </a>
              </div>
            </div>

            {/* Right: Visual with animation */}
            <div className="hidden md:block animate-fadeInRight">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-teal-400 to-cyan-500 rounded-3xl blur-3xl opacity-30 animate-pulse"></div>
                <div className="relative bg-gradient-to-br from-teal-400 via-cyan-400 to-blue-500 rounded-3xl p-12 shadow-2xl transform hover:scale-105 transition-transform duration-300">
                  <div className="aspect-square flex items-center justify-center">
                    <div className="text-center">
                      <div className="mb-6 inline-block transform animate-bounce">
                        <Logo size="2xl" />
                      </div>
                      <p className="text-white text-2xl font-bold">Urban Mobility Intelligence</p>
                      <p className="text-white/80 mt-4 text-sm">Powered by ML & Data Analytics</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 px-4 sm:px-6 lg:px-8 bg-white/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center text-gray-900 mb-12 animate-fadeInUp">
            💡 {t('about.title')}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="group bg-gradient-to-br from-blue-50 to-cyan-50 p-8 rounded-2xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 animate-fadeInUp" style={{ animationDelay: '0.1s' }}>
              <div className="text-5xl mb-4 transform group-hover:scale-110 transition-transform duration-300">👥</div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">{t('about.team')}</h3>
              <p className="text-gray-700 mb-4">
                {t('about.team_text')}
              </p>
              <p className="text-gray-700">
                {t('about.team_project')}
              </p>
            </div>

            <div className="group bg-gradient-to-br from-teal-50 to-green-50 p-8 rounded-2xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 animate-fadeInUp" style={{ animationDelay: '0.2s' }}>
              <div className="text-5xl mb-4 transform group-hover:scale-110 transition-transform duration-300">🎯</div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">{t('about.approach')}</h3>
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-center space-x-3 hover:translate-x-2 transition-transform duration-300">
                  <span className="text-teal-600 font-bold">✓</span>
                  <span>{t('about.approach1')}</span>
                </li>
                <li className="flex items-center space-x-3 hover:translate-x-2 transition-transform duration-300">
                  <span className="text-teal-600 font-bold">✓</span>
                  <span>{t('about.approach2')}</span>
                </li>
                <li className="flex items-center space-x-3 hover:translate-x-2 transition-transform duration-300">
                  <span className="text-teal-600 font-bold">✓</span>
                  <span>{t('about.approach3')}</span>
                </li>
                <li className="flex items-center space-x-3 hover:translate-x-2 transition-transform duration-300">
                  <span className="text-teal-600 font-bold">✓</span>
                  <span>{t('about.approach4')}</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-teal-50 to-cyan-50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center text-gray-900 mb-12 animate-fadeInUp">
            🎯 {t('mission.title')}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              { icon: '📊', title: t('mission.manage'), desc: t('mission.manage_desc') },
              { icon: '🚦', title: t('mission.anticipate'), desc: t('mission.anticipate_desc') },
              { icon: '🌍', title: t('mission.sustainability'), desc: t('mission.sustainability_desc') },
              { icon: '⚙️', title: t('mission.optimize'), desc: t('mission.optimize_desc') },
            ].map((item, idx) => (
              <div 
                key={idx} 
                className="group flex items-start space-x-4 p-6 bg-white rounded-2xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 animate-fadeInUp"
                style={{ animationDelay: `${idx * 0.1}s` }}
              >
                <div className="text-4xl transform group-hover:scale-125 transition-transform duration-300 flex-shrink-0">{item.icon}</div>
                <div>
                  <h4 className="font-bold text-lg mb-2 text-gray-900">{item.title}</h4>
                  <p className="text-gray-700">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center text-gray-900 mb-12 animate-fadeInUp">
            🚀 {t('features.title')}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { icon: '📊', title: t('features.realtime'), desc: t('features.realtime_desc') },
              { icon: '🚦', title: t('features.ml'), desc: t('features.ml_desc') },
              { icon: '🌍', title: t('features.environment'), desc: t('features.environment_desc') },
              { icon: '🤖', title: t('features.chatbot'), desc: t('features.chatbot_desc') },
              { icon: '⚙️', title: t('features.etl'), desc: t('features.etl_desc') },
              { icon: '📈', title: t('features.dashboard'), desc: t('features.dashboard_desc') },
            ].map((feature, idx) => (
              <div 
                key={idx} 
                className="group bg-gradient-to-br from-gray-50 to-gray-100 p-8 rounded-2xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-4 hover:from-teal-50 hover:to-cyan-50 animate-fadeInUp"
                style={{ animationDelay: `${idx * 0.05}s` }}
              >
                <div className="text-5xl mb-4 transform group-hover:scale-125 group-hover:rotate-12 transition-transform duration-300">{feature.icon}</div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-700">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Vision Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-teal-600 to-cyan-600 text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 right-0 w-96 h-96 bg-white rounded-full mix-blend-multiply filter blur-3xl"></div>
        </div>
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h2 className="text-4xl md:text-5xl font-bold mb-8 animate-fadeInUp">🌟 {t('vision.title')}</h2>
          <p className="text-xl text-white/90 mb-8 leading-relaxed animate-fadeInUp" style={{ animationDelay: '0.2s' }}>
            {t('vision.text')}
          </p>
          <p className="text-lg text-white/80 animate-fadeInUp" style={{ animationDelay: '0.4s' }}>
            {t('vision.text2')}
          </p>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-2xl mx-auto text-center animate-fadeInUp">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">{t('cta.ready')}</h2>
          <p className="text-gray-600 text-lg mb-8">
            {t('cta.join')}
          </p>
          <Link
            to="/login"
            className="inline-block px-10 py-4 bg-gradient-to-r from-teal-600 to-cyan-600 text-white rounded-xl font-bold text-lg hover:shadow-2xl transition-all duration-300 hover:scale-110 active:scale-95 hover:-translate-y-1"
          >
            🚀 {t('cta_button')}
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-gray-400">© 2026 Urban Mobility Intelligence Platform. Tous droits réservés.</p>
        </div>
      </footer>

      {/* Custom animations */}
      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes fadeInLeft {
          from {
            opacity: 0;
            transform: translateX(-30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes fadeInRight {
          from {
            opacity: 0;
            transform: translateX(30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes blob {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }

        .animate-fadeInUp {
          animation: fadeInUp 0.8s ease-out forwards;
          opacity: 0;
        }

        .animate-fadeInLeft {
          animation: fadeInLeft 0.8s ease-out forwards;
        }

        .animate-fadeInRight {
          animation: fadeInRight 0.8s ease-out forwards;
        }

        .animate-blob {
          animation: blob 7s infinite;
        }

        .animation-delay-2000 {
          animation-delay: 2s;
        }

        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </div>
  );
}
