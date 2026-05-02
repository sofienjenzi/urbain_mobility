# Configuration n8n pour l'intégration avec Flask API

Cette configuration permet à n8n d'appeler les endpoints de l'API Flask pour obtenir des prédictions de risque.

## Nœuds recommandés

### 1. HTTP Request - Prédiction unique

```json
{
  "node": "HTTP Request",
  "config": {
    "method": "POST",
    "url": "http://localhost:5000/api/v1/predict/single",
    "headers": {
      "Content-Type": "application/json"
    },
    "body": {
      "zone_id": "{{ $node.Trigger.json.zone_id }}",
      "year": "{{ $node.Trigger.json.year }}",
      "month": "{{ $node.Trigger.json.month }}"
    }
  }
}
```

### 2. HTTP Request - Prédictions futures

```json
{
  "node": "HTTP Request",
  "config": {
    "method": "GET",
    "url": "http://localhost:5000/api/v1/predict/future?months=36"
  }
}
```

### 3. HTTP Request - Statistiques par zone

```json
{
  "node": "HTTP Request",
  "config": {
    "method": "GET",
    "url": "http://localhost:5000/api/v1/stats/by-zone"
  }
}
```

## Workflow exemple

```
[Trigger] 
    ↓
[HTTP Request - Prédiction unique]
    ↓
[Set - Formater le résultat]
    ↓
[Google Sheets / Database - Stocker le résultat]
```

## Variables d'environnement recommandées

Ajouter à votre configuration n8n:
```env
FLASK_API_URL=http://localhost:5000
FLASK_API_VERSION=v1
```

## Gestion des erreurs

Utiliser un nœud "Try/Catch" ou "Error Handler" pour gérer:
- Connexion refusée (API non disponible)
- Réponses 404 (zone inexistante)
- Erreurs 500 (erreur serveur)

## Sécurité

Pour la production:
1. Configurer un token d'authentification dans l'API Flask
2. Utiliser HTTPS au lieu de HTTP
3. Ajouter l'authentification Bearer au header
4. Utiliser un reverse proxy (nginx) avec rate limiting
