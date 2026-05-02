# ⚡ Quick Start - Correction de l'API pour n8n

## 🔴 Problème Identifié
```
NameError: name 'run_Classification' is not defined
```

## ✅ Solutions Appliquées

### 1. **Correction de `api2.py`**
- ✅ Ajout de l'import: `from Objectif3_Classification_Risques_V2 import run_Classification`
- ✅ Gestion d'erreur d'import avec fallback
- ✅ Support des méthodes GET, POST, PUT
- ✅ Ajout du traitement des données JSON

### 2. **Restructuration de `Objectif3_Classification_Risques_V2.py`**
- ✅ Modèle ne s'entraîne plus automatiquement à l'import
- ✅ Sauvegarde/chargement du modèle en `.pkl` pour performance
- ✅ Fonction `run_Classification(input_data)` accepte les paramètres
- ✅ Prédictions individuelles basées sur les données d'entrée

### 3. **Fichiers Créés**
- ✅ `N8N_API_GUIDE.md` - Guide complet d'intégration n8n
- ✅ `test_api.py` - Script de test de l'API

---

## 🚀 Démarrage Rapide

### Étape 1: Démarrer l'API
```bash
cd c:\Users\user\Downloads\colab_notebooks
python api2.py
```

**Résultat attendu:**
```
🚀 Starting Urban Traffic Classification API...
📍 Available endpoints:
   - GET/POST/PUT http://127.0.0.1:5000/classification
   - GET http://127.0.0.1:5000/status
   - GET http://127.0.0.1:5000/
```

### Étape 2: Tester l'API (Terminal 2)
```bash
cd c:\Users\user\Downloads\colab_notebooks
python test_api.py
```

**Résultat attendu:**
```
✅ All tests passed!
🎉 Ready to integrate with n8n!
```

### Étape 3: Test Manuel Rapide
```bash
# Health check
curl http://127.0.0.1:5000/

# Classification par défaut
curl http://127.0.0.1:5000/classification

# Classification avec données
curl -X POST http://127.0.0.1:5000/classification \
  -H "Content-Type: application/json" \
  -d '{"volume": 1500, "taux_1000": 25}'
```

---

## 🔌 Configuration n8n

### URL d'Appel:
```
POST http://127.0.0.1:5000/classification
```

### Body JSON:
```json
{
  "volume": {{ $json.body.volume }},
  "taux_1000": {{ $json.body.taux_1000 }}
}
```

### Réponse Exemple (SUCCESS):
```json
{
  "status": "success",
  "model": "Pipeline",
  "f1_score": 0.85,
  "input": {
    "volume": 1500,
    "taux_1000": 25
  },
  "prediction": 1,
  "risk_score": 0.78,
  "risk_level": "HIGH",
  "timestamp": "2026-04-23T14:35:22.123456"
}
```

### Réponse Exemple (ERROR Prevention):
```json
{
  "status": "error",
  "message": "Could not load or train model",
  "http_method": "POST"
}
```

---

## 🎯 Prochaines Étapes n8n

### 1. HTTP Request Node
```
Method: POST
URL: http://127.0.0.1:5000/classification
Body: { "volume": ..., "taux_1000": ... }
```

### 2. Parse Response (JavaScript)
```javascript
const response = $input.all()[0].json;
return [{
  risk_level: response.risk_level,
  risk_score: response.risk_score,
  timestamp: response.timestamp
}];
```

### 3. IF Condition (Alert)
```
$json.risk_level === "HIGH"
```

### 4. Yes Path: Send Email Alert
- À: team@example.com
- Sujet: 🚨 ALERTE RISQUE: {{$json.risk_level}}
- Message: Volume={{$json.volume}}, Score={{$json.risk_score}}

### 5. Both Paths: Insert in PostgreSQL
```sql
INSERT INTO classifications(volume, taux, risk_level, score, timestamp)
VALUES({{...}})
```

---

## 📊 Endpoints Disponibles

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/status` | GET | Statut API |
| `/classification` | GET | Classification par défaut |
| `/classification` | POST | Classification avec données |
| `/classification` | PUT | Classification (PUT) |

---

## 🐛 Troubleshooting

### Erreur: "Connection refused"
```
❌ Cannot reach API at http://127.0.0.1:5000
```
**Solution:** L'API n'est pas démarrée. Exécutez:
```bash
python api2.py
```

### Erreur: "NameError: name 'run_Classification' is not defined"
```
✅ CORRECTION APPLIQUÉE - Cela ne devrait plus se produire
```
Mais si c'est pas le cas:
1. Vérifier que `api2.py` importe bien depuis `Objectif3_Classification_Risques_V2.py`
2. Vérifier que les deux fichiers sont dans le même répertoire

### Erreur PostgreSQL (connexion DB)
```json
{"status": "error", "message": "could not connect to server"}
```
**Solution:**
1. Vérifier que PostgreSQL est actif
2. Vérifier les credentials en variables d'environnement:
   ```bash
   $env:DB_HOST = "localhost"
   $env:DB_PORT = "5432"
   $env:DB_NAME = "urbain_dw"
   $env:DB_USER = "postgres"
   $env:DB_PASSWORD = "admin"
   ```
3. Ou modifier directement dans `Objectif3_Classification_Risques_V2.py`

---

## 🎓 Format de Données Attendu

### Input (POST/PUT Body):
```json
{
  "volume": 1500,           // Traffic volume (0-5000)
  "taux_1000": 25          // Risk rate per 1000 (0-100)
}
```

### Output:
```json
{
  "status": "success",
  "prediction": 0,          // 0 = LOW RISK, 1 = HIGH RISK
  "risk_level": "LOW",      // "LOW" or "HIGH"
  "risk_score": 0.23,       // Probability (0.0-1.0)
  "f1_score": 0.85,         // Model accuracy
  "timestamp": "ISO8601",   // When prediction was made
  "http_method": "POST"
}
```

---

## 📝 Fichiers Modifiés

```
✅ api2.py
   - Import run_Classification
   - Support POST/PUT
   - Gestion erreurs

✅ Objectif3_Classification_Risques_V2.py
   - Model lazy loading
   - Save/Load en pickle
   - run_Classification(input_data)

✨ Nouveaux fichiers:
   - N8N_API_GUIDE.md
   - test_api.py
   - QUICK_START.md (ce fichier)
```

---

## ✅ Checklist

- [ ] API démarrée avec `python api2.py`
- [ ] Tests passants: `python test_api.py`
- [ ] URL API accessible: http://127.0.0.1:5000
- [ ] n8n HTTP Node configuré
- [ ] Body JSON correctement formaté
- [ ] PostgreSQL credentials valides
- [ ] Workflow n8n testé de bout en bout

---

## 💡 Conseil Pro Pour n8n

**Utiliser une variable d'environnement pour l'URL API:**

Dans n8n, créer une variable:
```
API_URL = http://127.0.0.1:5000
```

Puis dans le HTTP Node:
```
URL: {{$vars.API_URL}}/classification
```

Cela facilite le changement d'environnement (dev/prod).

---

**Dernière mise à jour:** 2026-04-23
**Status:** ✅ Prêt pour intégration n8n
