# 🏙️ Insight - Urban Mobility Intelligence Platform

Application React/Vite pour "Insight", une plateforme d'analyse de la mobilité urbaine.

## 📁 Structure du Projet

```
insight-app/
├── src/
│   ├── components/         # Composants réutilisables
│   │   └── Navbar.jsx
│   ├── pages/              # Pages principales
│   │   ├── Home.jsx        # Page d'accueil
│   │   ├── Login.jsx       # Authentification & sélection rôle
│   │   ├── Dashboard.jsx   # Routeur de dashboard
│   │   └── dashboards/     # Dashboards spécifiques par rôle
│   │       ├── AdminDashboard.jsx
│   │       ├── MinisterDashboard.jsx
│   │       ├── AirParifDashboard.jsx
│   │       ├── CitizenDashboard.jsx
│   │       └── TransportDashboard.jsx
│   ├── App.jsx             # Composant principal avec routage
│   ├── main.jsx            # Point d'entrée React
│   └── index.css           # Styles globaux
├── index.html              # HTML racine
├── package.json
├── vite.config.js          # Configuration Vite
├── tailwind.config.js      # Configuration Tailwind
└── postcss.config.js       # Configuration PostCSS

```

## 🎯 Fonctionnalités

### Page d'Accueil
- Présentation professionnelle de la plateforme Insight
- Sections: "Qui sommes-nous", "Mission", "Fonctionnalités", "Vision"
- Design moderne avec gradients et animations
- Appels à l'action clairs

### Authentification
- 5 rôles disponibles :
  - **Admin**: Power BI + Grafana MLOps Monitoring
  - **Ministre de l'Intérieur**: Objectif 3 (Classification des Risques)
  - **Air Parif**: Objectif 4 (CO2 & Énergie)
  - **Citoyen**: Objectif 2 (Recommandation de Trajets)
  - **Société de Transport**: Objectif 1 (Forecast du Trafic)
- Interface de sélection visuelle des rôles
- Mode démo (sans authentification réelle)

### Dashboards Spécifiques par Rôle
Chaque rôle a un dashboard personnalisé avec :
- Métriques KPI clés
- Interfaces d'interaction avec les APIs
- Visualisations adaptées

## 🚀 Installation et Démarrage

### Prérequis
- Node.js >= 16
- npm ou yarn

### 1. Installation des dépendances

```bash
cd insight-app
npm install
```

### 2. Démarrage en développement

```bash
npm run dev
```

L'app démarre sur **http://localhost:3000**

### 3. Build pour production

```bash
npm run build
npm run preview
```

## 🎨 Design & Styling

- **Tailwind CSS** : Framework CSS utility-first
- **Couleurs principales** : Teal (#0d9488) et gradients
- **Responsive** : Mobile-first, adapté à tous les écrans
- **Composants modernes** : Cartes, formulaires, boutons

## 🔑 Variables d'Environnement

(Optionnel) Créer `.env` pour les URLs d'API :

```env
VITE_API_URL=http://localhost:8000
VITE_GRAFANA_URL=http://localhost:3000
VITE_POWER_BI_URL=http://localhost:5000
```

## 📱 Pages Principales

### Home (`/`)
- Page d'accueil publique
- Présentation de la mission et valeurs

### Login (`/login`)
- Sélection du rôle utilisateur
- Formulaire de connexion (mode démo)

### Dashboard (`/dashboard`)
- Page protégée (redirection vers login si non authentifié)
- Contenu adapté au rôle de l'utilisateur
- Navbar avec déconnexion

## 🔐 Authentification

**Mode Démo** : L'authentification stocke l'utilisateur dans `localStorage`
- Email et mot de passe acceptent n'importe quelle valeur
- Rôle détermine le contenu du dashboard
- La déconnexion efface les données de session

**Pour production** : Intégrer un vrai système d'auth (JWT, OAuth, etc.)

## 🔗 Intégration APIs

Chaque dashboard peut appeler les APIs de prédiction :

### Exemples d'intégration

#### Objectif 1 (Transport)
```javascript
fetch('http://localhost:8000/predict/1', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ input: { steps: 12 } })
})
```

#### Objectif 2 (Citoyen)
```javascript
fetch('http://localhost:8000/predict/2', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    input: { ville: 'Paris', vitesse: 25 } 
  })
})
```

#### Objectif 3 (Ministre)
```javascript
fetch('http://localhost:8000/predict/3', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    input: { volume: 1500, taux_1000: 0.8 } 
  })
})
```

#### Objectif 4 (Air Parif)
```javascript
fetch('http://localhost:8000/predict/4', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    input: { 
      zone_id: 5, year: 2026, month: 4,
      mode: 'bus', activity_value: 120, emission_factor: 0.25
    } 
  })
})
```

## 📊 Prochaines Étapes

1. Intégrer les appels API réels aux dashboards
2. Ajouter des graphiques interactifs (Chart.js, D3.js)
3. Implémenter l'authentification complète
4. Ajouter les pages Grafana et Power BI en iframes
5. Créer les pages de détail par objectif

## 📝 Notes

- Design fully responsive
- Dark mode optionnel possible
- Prêt pour intégration avec APIs FastAPI existantes
- Prêt pour Streamlit/Grafana embeddé

## 👨‍💻 Auteur

Insight Team - 2026
