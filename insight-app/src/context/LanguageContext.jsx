import { createContext, useContext, useState } from 'react';

// Traductions
const translations = {
  fr: {
    // Navbar
    'navbar.dashboard': 'Tableau de Bord',
    'navbar.login': 'Se Connecter',
    'navbar.logout': 'Déconnexion',
    'navbar.intelligence': 'Plateforme Intelligence',

    // Home
    'home.welcome': 'Bienvenue sur',
    'home.title': 'Urban Mobility',
    'home.subtitle': 'Une plateforme intelligente dédiée à l\'analyse et à l\'optimisation de la mobilité urbaine.',
    'home.description': 'Transformez les données complexes de la ville en informations simples, utiles et exploitables pour améliorer la vie des citoyens et la performance des systèmes de transport.',
    'home.cta_platform': 'Accéder à la Plateforme',
    'home.cta_learn': 'En Savoir Plus',

    // About
    'about.title': 'Qui sommes-nous ?',
    'about.team': 'Notre Équipe',
    'about.team_text': 'Nous sommes une équipe d\'étudiants passionnés par les data, l\'intelligence artificielle et les systèmes intelligents.',
    'about.team_project': 'Dans le cadre de notre projet intégré, nous avons conçu une solution complète de data-driven decision support system appliquée à la mobilité urbaine.',
    'about.approach': 'Notre Approche',
    'about.approach1': 'Collecte de données multi-sources',
    'about.approach2': 'Data Warehouse structuré',
    'about.approach3': 'Processus ETL avancé',
    'about.approach4': 'Analyse prédictive ML',

    // Mission
    'mission.title': 'Notre Mission',
    'mission.manage': 'Améliorer la Gestion',
    'mission.manage_desc': 'Gestion de la mobilité urbaine grâce aux données intelligentes',
    'mission.anticipate': 'Anticiper les Problèmes',
    'mission.anticipate_desc': 'Prédire les embouteillages via machine learning',
    'mission.sustainability': 'Durabilité',
    'mission.sustainability_desc': 'Suivre l\'impact environnemental et réduire la pollution',
    'mission.optimize': 'Optimisation',
    'mission.optimize_desc': 'Automatiser le traitement et l\'intégration des données',

    // Features
    'features.title': 'Nos Fonctionnalités',
    'features.realtime': 'Analyse Temps Réel',
    'features.realtime_desc': 'Tendances de mobilité analysées instantanément',
    'features.ml': 'Prédiction ML',
    'features.ml_desc': 'Anticipez les embouteillages avant qu\'ils ne se produisent',
    'features.environment': 'Impact Environnemental',
    'features.environment_desc': 'Suivi de la qualité de l\'air et des émissions',
    'features.chatbot': 'Chatbot Intelligent',
    'features.chatbot_desc': 'Recommandations personnalisées en temps réel',
    'features.etl': 'Automatisation ETL',
    'features.etl_desc': 'Traitement et intégration automatisés des données',
    'features.dashboard': 'Dashboards Avancés',
    'features.dashboard_desc': 'Visualisations professionnelles et interactives',

    // Vision
    'vision.title': 'Notre Vision',
    'vision.text': 'Créer un écosystème de mobilité urbaine intelligent, durable et inclusif où les données guident chaque décision, optimisent chaque trajet et contribuent à une ville plus verte, plus fluide et plus humaine.',
    'vision.text2': 'À travers Urban Mobility, nous transformons la façon dont les villes gèrent leurs transports pour un avenir meilleur.',

    // CTA
    'cta.ready': 'Prêt à Commencer ?',
    'cta.join': 'Rejoignez-nous pour explorer les capacités de la plateforme Urban Mobility Intelligence.',
    'cta_button': 'Accéder Maintenant',

    // Login
    'login.title': 'Sélectionnez votre Rôle',
    'login.email': 'Email',
    'login.password': 'Mot de passe',
    'login.signin': 'Se Connecter',
    'login.back': 'Retour à l\'accueil',
  },
  en: {
    // Navbar
    'navbar.dashboard': 'Dashboard',
    'navbar.login': 'Sign In',
    'navbar.logout': 'Logout',
    'navbar.intelligence': 'Intelligence Platform',

    // Home
    'home.welcome': 'Welcome to',
    'home.title': 'Urban Mobility',
    'home.subtitle': 'An intelligent platform dedicated to the analysis and optimization of urban mobility.',
    'home.description': 'Transform complex city data into simple, useful, and actionable information to improve citizens\' lives and transportation system performance.',
    'home.cta_platform': 'Access Platform',
    'home.cta_learn': 'Learn More',

    // About
    'about.title': 'Who are we?',
    'about.team': 'Our Team',
    'about.team_text': 'We are a team of students passionate about data, artificial intelligence, and intelligent systems.',
    'about.team_project': 'As part of our integrated project, we designed a complete data-driven decision support system applied to urban mobility.',
    'about.approach': 'Our Approach',
    'about.approach1': 'Multi-source data collection',
    'about.approach2': 'Structured Data Warehouse',
    'about.approach3': 'Advanced ETL process',
    'about.approach4': 'ML predictive analysis',

    // Mission
    'mission.title': 'Our Mission',
    'mission.manage': 'Improve Management',
    'mission.manage_desc': 'Urban mobility management through intelligent data',
    'mission.anticipate': 'Anticipate Problems',
    'mission.anticipate_desc': 'Predict traffic jams via machine learning',
    'mission.sustainability': 'Sustainability',
    'mission.sustainability_desc': 'Track environmental impact and reduce pollution',
    'mission.optimize': 'Optimization',
    'mission.optimize_desc': 'Automate data processing and integration',

    // Features
    'features.title': 'Our Features',
    'features.realtime': 'Real-time Analysis',
    'features.realtime_desc': 'Mobility trends analyzed instantly',
    'features.ml': 'ML Prediction',
    'features.ml_desc': 'Anticipate traffic before it happens',
    'features.environment': 'Environmental Impact',
    'features.environment_desc': 'Track air quality and emissions',
    'features.chatbot': 'Smart Chatbot',
    'features.chatbot_desc': 'Personalized recommendations in real-time',
    'features.etl': 'ETL Automation',
    'features.etl_desc': 'Automated data processing and integration',
    'features.dashboard': 'Advanced Dashboards',
    'features.dashboard_desc': 'Professional and interactive visualizations',

    // Vision
    'vision.title': 'Our Vision',
    'vision.text': 'Create an intelligent, sustainable, and inclusive urban mobility ecosystem where data guides every decision, optimizes every journey, and contributes to a greener, smoother, and more human city.',
    'vision.text2': 'Through Urban Mobility, we transform how cities manage their transportation for a better future.',

    // CTA
    'cta.ready': 'Ready to Get Started?',
    'cta.join': 'Join us to explore the capabilities of the Urban Mobility Intelligence platform.',
    'cta_button': 'Access Now',

    // Login
    'login.title': 'Select Your Role',
    'login.email': 'Email',
    'login.password': 'Password',
    'login.signin': 'Sign In',
    'login.back': 'Back to home',
  },
};

// Créer le contexte
const LanguageContext = createContext();

// Provider
export function LanguageProvider({ children }) {
  const [language, setLanguage] = useState(
    localStorage.getItem('language') || 'fr'
  );

  const switchLanguage = (lang) => {
    setLanguage(lang);
    localStorage.setItem('language', lang);
  };

  const t = (key) => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, switchLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

// Hook pour utiliser le contexte
export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within LanguageProvider');
  }
  return context;
}
