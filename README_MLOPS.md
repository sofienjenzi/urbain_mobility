# MLOps (Objectifs 1→4)

Objectif: appliquer MLOps **sur vos scripts existants**.

## 1) Experiment tracking (MLflow)
- Chaque entraînement est lancé via `pipelines/train_objective.py`.
- Les sorties `stdout/stderr` sont loggées comme artifacts.
- Les versions de modèles créées dans `models/objectiveX/` sont loggées.

## 2) Training pipeline automatisé
- Exécution de bout en bout: data load → preprocessing → training → eval → export modèle.
- Reproductible: `python pipelines/train_objective.py --objective X`.

## 3) Model management (versioning)
- Registry local:
  - `models/objective1/<version>/...` + `models/objective1/latest/...`
  - idem pour objective2/3/4

## 4) Model serving
- FastAPI: `serving/api.py`
- Endpoint: `POST /predict/{objective}`

## 5) Docker
- `docker-compose.yml` démarre:
  - MLflow UI (port 5001)
  - API (port 8000)
  - Webapp (port 8501)

## 6) Monitoring minimal
- Les requêtes de prédiction sont loggées dans `data/predictions.jsonl` (latence + input/output)

## 7) Ré-entraînement automatique (simple)
- Script time-based: ré-entraine si aucun modèle ou si le modèle a plus de N jours.
- Commande: `python pipelines/auto_retrain.py --objective X --max-age-days 7`

## Commandes

### Entraîner un objectif
```powershell
python pipelines/train_objective.py --objective 3
```

### Ré-entraîner automatiquement (si modèle trop ancien)
```powershell
python pipelines/auto_retrain.py --objective 3 --max-age-days 7
```

### Lancer l'API
```powershell
uvicorn serving.api:app --reload --port 8000
```

### Tester un endpoint
```powershell
python -c "import requests; print(requests.post('http://localhost:8000/predict/3', json={'input': {'volume': 1500, 'taux_1000': 0.8}}).json())"
```

### Web app
```powershell
$env:PREDICTION_API_URL='http://localhost:8000'
streamlit run webapp/app.py
```

### Docker compose
```powershell
docker compose up --build
```

- MLflow UI: http://localhost:5001
- API docs: http://localhost:8000/docs
- Webapp: http://localhost:8501
