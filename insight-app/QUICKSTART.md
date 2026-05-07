## 🚀 Guide de Démarrage Rapide - Insight Web App

### Prérequis
- Windows, macOS, ou Linux
- Node.js 16+ ([télécharger](https://nodejs.org))
- Terminal/PowerShell

### ⚡ Installation en 3 Étapes

#### 1. Ouvrir le terminal dans le dossier `insight-app`

```bash
cd insight-app
```

#### 2. Installer les dépendances

```bash
npm install
```

Cela va installer :
- React 18
- React Router 6
- Tailwind CSS
- Vite (bundler)

#### 3. Démarrer le serveur de développement

```bash
npm run dev
```

Vous devriez voir :
```
VITE v4.3.9  ready in 245 ms

➜  Local:   http://localhost:3000
```

### 🌐 Accès à l'Application

Ouvrez votre navigateur et allez à : **http://localhost:3000**

### 📋 Flux d'Utilisation

1. **Page d'Accueil** : Découvrez Insight
2. **Se Connecter** : Cliquez sur "Se Connecter" ou "Accéder à la Plateforme"
3. **Sélectionner Rôle** : Choisissez parmi 5 rôles (Admin, Ministre, Air Parif, Citoyen, Transport)
4. **Remplir Email/Mot de Passe** : Entrez n'importe quel email/mot de passe (mode démo)
5. **Accès Dashboard** : Dashboard personnalisé selon votre rôle

### 🔑 Rôles & Dashboards

| Rôle | Icône | Dashboard |
|------|-------|-----------|
| Admin | ⚙️ | Power BI + Grafana MLOps |
| Ministre | 🏛️ | Objectif 3 - Risques |
| Air Parif | 🌍 | Objectif 4 - CO2 |
| Citoyen | 👤 | Objectif 2 - Trajets |
| Transport | 🚌 | Objectif 1 - Trafic |

### 🔧 Commandes Utiles

```bash
# Démarrer le serveur (développement)
npm run dev

# Build pour production
npm run build

# Prévisualiser le build
npm run preview

# Installer une nouvelle dépendance
npm install nom-du-package

# Supprimer node_modules et réinstaller
npm ci
```

### 🌐 URLs Principales

- **App Insight** : http://localhost:3000
- **API Backend** : http://localhost:8000 (si disponible)
- **Prometheus** : http://localhost:9090
- **Grafana** : http://localhost:3000 (conflit possible avec Insight)

### 📝 Variables d'Environnement (Optionnel)

Créer un fichier `.env.local` à la racine du projet :

```env
VITE_API_URL=http://localhost:8000
VITE_GRAFANA_URL=http://localhost:3000
```

Puis utiliser dans le code :
```javascript
const apiUrl = import.meta.env.VITE_API_URL
```

### 🐛 Dépannage

**Port 3000 déjà utilisé ?**
```bash
# Changer le port dans vite.config.js
# ou

npm run dev -- --port 3001
```

**Erreur : "Cannot find module" ?**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Changements ne s'affichent pas ?**
- Rafraîchir la page (F5 ou Ctrl+R)
- Vider le cache du navigateur (Ctrl+Shift+Delete)
- Redémarrer le serveur (Ctrl+C puis npm run dev)

### 📖 Documentation

- [React Docs](https://react.dev)
- [Vite Docs](https://vitejs.dev)
- [Tailwind CSS Docs](https://tailwindcss.com)
- [React Router Docs](https://reactrouter.com)

### ✅ Checklist de Vérification

- [ ] Node.js installé (`node --version`)
- [ ] npm installé (`npm --version`)
- [ ] Dépendances installées (`npm install`)
- [ ] Serveur démarre sans erreurs (`npm run dev`)
- [ ] App accessible sur http://localhost:3000
- [ ] Peut se connecter avec un rôle
- [ ] Dashboard s'affiche correctement

### 🎯 Prochaines Étapes

1. Intégrer les appels API aux dashboards
2. Ajouter des graphiques (Chart.js, D3.js)
3. Implémenter l'authentification réelle
4. Déployer sur Vercel, Netlify, ou autre

### 💬 Support

En cas de problème, consultez :
- [Forum Vite](https://github.com/vitejs/vite/discussions)
- [Issues React Router](https://github.com/remix-run/react-router/issues)
- Logs du navigateur (F12 > Console)

Bon développement ! 🚀
