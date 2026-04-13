"""
OBJECTIF 1: PRÉDIRE LE TRAFIC URBAIN PAR ZONE, HEURE, MOIS, ANNÉE
Sorties: prévisions sur 36 mois + visualisations + comparaison de modèles.
"""

# ============ DÉPENDANCES / COMPAT COLAB vs PYTHON ============
from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")


def _is_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except Exception:
        return False


def _data_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _read_csv(filename: str) -> pd.DataFrame:
    p = _data_dir() / filename
    if p.exists():
        return pd.read_csv(p)
    p2 = Path.cwd() / filename
    if p2.exists():
        return pd.read_csv(p2)
    raise FileNotFoundError(
        f"Fichier introuvable: {filename}. Place-le dans {_data_dir()} ou dans le répertoire courant."
    )


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# ============ CHARGEMENT DES DONNÉES ============
print("🚗 OBJECTIF 1: PRÉDIRE LE TRAFIC URBAIN\n")
print("📊 CHARGEMENT DES DONNÉES...")

if _is_colab():
    try:
        from google.colab import files  # type: ignore

        _ = files.upload()
    except Exception:
        pass

# Charger les données
fact_circulation = _read_csv('fact_circulation.csv')
dim_trafic = _read_csv('dim_trafic.csv')
dim_zone = _read_csv('dim_zone.csv')
dim_time = _read_csv('dim_time.csv')

print(f"✓ fact_circulation chargé: {fact_circulation.shape}")
print(f"✓ dim_trafic chargé: {dim_trafic.shape}")
print(f"✓ dim_zone chargé: {dim_zone.shape}")
print(f"✓ dim_time chargé: {dim_time.shape}")

# ============ EXPLORATION DES DONNÉES ============
print("\n📈 EXPLORATION DES DONNÉES...")
print("\nColonnes fact_circulation:")
print(f"{list(fact_circulation.columns)}")
print(fact_circulation.head())

print("\nColonnes dim_zone:")
print(f"{list(dim_zone.columns)}")
print(dim_zone.head())

# ============ PRÉPARATION DES DONNÉES ============
print("\n🔧 PRÉPARATION DES DONNÉES...")

# Fusionner les données
df = fact_circulation.copy()

# Normaliser les IDs (évite KeyError si certaines dimensions ne sont pas jointes)
if 'zone_id' not in df.columns and 'fk_zone' in df.columns:
    df['zone_id'] = df['fk_zone']
if 'trafic_id' not in df.columns and 'fk_trafic' in df.columns:
    df['trafic_id'] = df['fk_trafic']

# Joindre les dimensions
if 'zone_id' in df.columns and 'zone_id' in dim_zone.columns:
    zone_cols = ['zone_id']
    if 'zone_type' in dim_zone.columns:
        zone_cols.append('zone_type')
    # Ne merge que si on apporte des infos supplémentaires (sinon inutile)
    if len(zone_cols) > 1:
        df = df.merge(
            dim_zone[zone_cols].drop_duplicates('zone_id'),
            on='zone_id',
            how='left',
        )

if 'trafic_id' in df.columns and 'trafic_id' in dim_trafic.columns:
    trafic_cols = ['trafic_id']
    if 'source_sensor' in dim_trafic.columns:
        trafic_cols.append('source_sensor')
    if len(trafic_cols) > 1:
        df = df.merge(
            dim_trafic[trafic_cols].drop_duplicates('trafic_id'),
            on='trafic_id',
            how='left',
        )

# Colonne date
if 'date' in df.columns:
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        print("⚠️ Colonne date illisible, création d'une série temporelle synthétique")
        df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='H')
else:
    print("⚠️ Colonne date non trouvée, création d'une série temporelle synthétique")
    df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='H')

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['hour'] = df['date'].dt.hour

# Sélectionner les colonnes numériques
numeric_cols_possible = ['congestion_index', 'vitesse_moyenne', 'volume_trafic', 
                         'densite_vehicules', 'temps_trajet_moyen']
numeric_cols = [col for col in numeric_cols_possible if col in df.columns]

print(f"\nColonnes numériques trouvées: {numeric_cols}")

if not numeric_cols:
    print("❌ ERREUR: Aucune colonne numérique trouvée!")
    print(f"Colonnes disponibles: {list(df.columns)}")
    raise ValueError("Pas de colonne cible pour la prédiction")

# Choisir la target
if 'congestion_index' in df.columns:
    target_col = 'congestion_index'
elif 'vitesse_moyenne' in df.columns:
    target_col = 'vitesse_moyenne'
else:
    target_col = numeric_cols[0]

# Créer le dataset
if 'zone_id' not in df.columns:
    df['zone_id'] = 1

desired_cols = list(dict.fromkeys(numeric_cols + ['zone_id', 'year', 'month', 'hour', 'date']))
base_cols = [c for c in desired_cols if c in df.columns]

missing_required = [c for c in [target_col, 'date', 'zone_id'] if c not in base_cols]
if missing_required:
    raise KeyError(f"Colonnes requises manquantes pour le modèle: {missing_required}. Colonnes dispo: {list(df.columns)}")

df_model = df[base_cols].dropna(subset=[target_col, 'date'])
df_model = df_model[(df_model[numeric_cols] >= 0).all(axis=1)]

# A — Outliers: clipping quantiles (1%–99%) sur numériques
for col in numeric_cols:
    if col in df_model.columns:
        lo, hi = df_model[col].quantile([0.01, 0.99]).values
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            df_model[col] = df_model[col].clip(lo, hi)

print(f"\n✓ Dataset: {df_model.shape}")
print(f"Statistiques descriptives:")
print(df_model[numeric_cols].describe())


# ============================
# B — MODEL UNDERSTANDING (obligatoire)
# ============================
print("\n" + "=" * 70)
print("🧠 COMPRÉHENSION DU MODÈLE (B)")
print("=" * 70)
print(
    "Objectif: prévoir une série temporelle (trafic/congestion). On compare deux approches:\n"
    "- SARIMAX (ARIMA saisonnier): modèle statistique pour tendances + saisonnalité (hypothèses de stationnarité après différenciation).\n"
    "- Modèle ML supervisé sur lags: transforme la série en problème de régression avec retards (capte non-linéarités).\n"
    "Choix final via MAE/RMSE/MAPE sur une coupe temporelle." 
)

# ============================
# F — TIME SERIES ANALYSIS & FORECASTING
# ============================
print("\n" + "=" * 70)
print("⏱️ TIME SERIES / FORECASTING (F) — ANALYSE + 2 MODÈLES")
print("=" * 70)

df_model = df_model.sort_values('date').copy()
ts = df_model.set_index('date')[target_col].resample('MS').mean().dropna()

if len(ts) < 18:
    print(f"⚠️ Série mensuelle courte ({len(ts)} points). Les résultats seront moins stables.")

# Stationarity tests + decomposition (si statsmodels dispo)
statsmodels_ok = True
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as e:
    statsmodels_ok = False
    print(f"⚠️ statsmodels indisponible: {e}")

if statsmodels_ok and len(ts) >= 24:
    try:
        adf_p = adfuller(ts.dropna(), autolag='AIC')[1]
        kpss_p = kpss(ts.dropna(), regression='c', nlags='auto')[1]
        print(f"ADF p-value: {adf_p:.4f} | KPSS p-value: {kpss_p:.4f}")
    except Exception as e:
        print(f"⚠️ Tests ADF/KPSS échoués: {e}")

    try:
        decomp = seasonal_decompose(ts, model='additive', period=12)
        decomp.plot()
        plt.suptitle('Décomposition saisonnière (mensuelle)', y=1.02)
        plt.tight_layout()
        plt.savefig('Objectif1_Decomposition.png', dpi=200, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"⚠️ Décomposition échouée: {e}")


def _train_test_split_ts(series: pd.Series, test_months: int = 6):
    if len(series) <= test_months + 3:
        test_months = max(1, len(series) // 5)
    train = series.iloc[:-test_months]
    test = series.iloc[-test_months:]
    return train, test


train_ts, test_ts = _train_test_split_ts(ts, test_months=6)
print(f"✓ Train months: {len(train_ts)} | Test months: {len(test_ts)}")


def _eval_forecast(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true_arr = y_true.values.astype(float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mape = _mape(y_true_arr, y_pred_arr)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# -------- Baseline: Naive (dernier point observé)
def _naive_last_forecast(train: pd.Series, steps: int) -> np.ndarray:
    if len(train) == 0:
        return np.zeros(steps, dtype=float)
    return np.full(shape=(steps,), fill_value=float(train.iloc[-1]), dtype=float)


# -------- Baseline: Seasonal Naive (répète le même mois de l'année précédente)
def _seasonal_naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    preds = []
    full = train.copy()
    last_date = full.index[-1]
    for _ in range(steps):
        next_date = (last_date + pd.offsets.MonthBegin(1)).normalize()
        ref_date = next_date - pd.DateOffset(years=1)
        if ref_date in full.index:
            preds.append(float(full.loc[ref_date]))
        else:
            preds.append(float(full.iloc[-1]))
        # extend for multi-step
        full.loc[next_date] = preds[-1]
        last_date = next_date
    return np.asarray(preds, dtype=float)


naive_pred = _seasonal_naive_forecast(train_ts, steps=len(test_ts))
naive_metrics = _eval_forecast(test_ts, naive_pred)
print("\n📌 Seasonal Naive metrics:")
print(pd.Series(naive_metrics).to_string())

last_pred = _naive_last_forecast(train_ts, steps=len(test_ts))
last_metrics = _eval_forecast(test_ts, last_pred)
print("\n📌 NaiveLast metrics:")
print(pd.Series(last_metrics).to_string())


# -------- Modèle 1: SARIMAX (si dispo)
sarimax_pred = None
sarimax_metrics = None
sarimax_model = None
if statsmodels_ok and len(train_ts) >= 18:
    try:
        sarimax_model = SARIMAX(
            train_ts,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        sarimax_pred = sarimax_model.get_forecast(steps=len(test_ts)).predicted_mean.values
        sarimax_metrics = _eval_forecast(test_ts, sarimax_pred)
        print("\n📌 SARIMAX metrics:")
        print(pd.Series(sarimax_metrics).to_string())
    except Exception as e:
        print(f"⚠️ SARIMAX échoué: {e}")


# -------- Modèle 2: ML sur lags (GradientBoosting)
def _choose_lags(n_points: int, preferred=(1, 2, 3, 6, 12)) -> tuple:
    max_lag = n_points - 2
    if max_lag < 1:
        return tuple()
    lags = tuple(l for l in preferred if l <= max_lag)
    return lags if lags else (1,)


def _make_supervised(series: pd.Series, lags=(1, 2, 3, 6, 12)) -> pd.DataFrame:
    df_s = pd.DataFrame({'y': series})
    for lag in lags:
        df_s[f'lag_{lag}'] = df_s['y'].shift(lag)
    idx = df_s.index
    df_s['month'] = idx.month
    df_s['month_sin'] = np.sin(2 * np.pi * df_s['month'] / 12)
    df_s['month_cos'] = np.cos(2 * np.pi * df_s['month'] / 12)
    return df_s.dropna()

ml_best = None
ml_metrics = None

lags_used = _choose_lags(len(ts))
sup = _make_supervised(ts, lags=lags_used) if len(lags_used) > 0 else pd.DataFrame()

if sup.empty or len(sup) < max(4, len(test_ts) + 2):
    print(
        f"\n⚠️ ML (lags) ignoré: série trop courte pour construire des retards (points={len(ts)}, lags={lags_used})."
    )
else:
    test_start = test_ts.index.min() if len(test_ts) > 0 else None
    if test_start is not None and test_start in sup.index:
        sup_train = sup.loc[sup.index < test_start]
        sup_test = sup.loc[sup.index >= test_start]
    else:
        # fallback: coupe en fin de série supervisée
        k = int(len(test_ts)) if len(test_ts) > 0 else max(1, int(len(sup) * 0.2))
        k = max(1, min(k, len(sup) - 1))
        sup_train = sup.iloc[:-k]
        sup_test = sup.iloc[-k:]

    X_tr = sup_train.drop(columns=['y'])
    y_tr = sup_train['y'].values
    X_te = sup_test.drop(columns=['y'])
    y_te = sup_test['y'].values

    gbr = GradientBoostingRegressor(random_state=42)

    if len(sup_train) >= 20:
        n_splits = min(4, max(2, len(sup_train) // 5))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        param_dist = {
            'n_estimators': [200, 400, 600],
            'learning_rate': [0.03, 0.06, 0.1],
            'max_depth': [2, 3, 4],
            'subsample': [0.7, 0.9, 1.0],
        }

        ml_search = RandomizedSearchCV(
            gbr,
            param_distributions=param_dist,
            n_iter=12,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            random_state=42,
            n_jobs=-1,
        )

        print("\n🔎 RandomizedSearchCV (ML lag-based) ...")
        ml_search.fit(X_tr, y_tr)
        ml_best = ml_search.best_estimator_
    else:
        print("\n⚠️ Peu de points: entraînement ML sans tuning/CV.")
        ml_best = gbr.fit(X_tr, y_tr)

    ml_pred = ml_best.predict(X_te)
    ml_metrics = _eval_forecast(pd.Series(y_te, index=sup_test.index), ml_pred)
    print("\n📌 ML (lags) metrics:")
    print(pd.Series(ml_metrics).to_string())


def _forecast_ml_iterative(model, history: pd.Series, steps: int = 36, lags=(1, 2, 3, 6, 12)) -> pd.Series:
    hist = history.copy()
    preds = []
    current_index = hist.index
    last_date = current_index[-1]
    for i in range(steps):
        next_date = (last_date + pd.offsets.MonthBegin(1)).normalize()
        # construire features
        feats = {}
        for lag in lags:
            feats[f'lag_{lag}'] = float(hist.iloc[-lag])
        m = int(next_date.month)
        feats['month'] = m
        feats['month_sin'] = float(np.sin(2 * np.pi * m / 12))
        feats['month_cos'] = float(np.cos(2 * np.pi * m / 12))
        x = pd.DataFrame([feats])
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        hist.loc[next_date] = yhat
        last_date = next_date
    return pd.Series(preds, index=pd.date_range(start=history.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS'))


# Comparaison & sélection du meilleur modèle
model_rows = []
model_rows.append({'model': 'SeasonalNaive', **naive_metrics})
model_rows.append({'model': 'NaiveLast', **last_metrics})
if sarimax_metrics is not None:
    model_rows.append({'model': 'SARIMAX', **sarimax_metrics})
if ml_metrics is not None:
    model_rows.append({'model': 'ML_Lags_GBR', **ml_metrics})
model_cmp = pd.DataFrame(model_rows).sort_values('RMSE')
print("\n✅ Comparaison modèles:")
print(model_cmp.to_string(index=False))

best_model_name = model_cmp.iloc[0]['model']
print(f"\n🏆 Modèle retenu: {best_model_name}")

# Forecast 36 mois (mensuel global)
if best_model_name == 'SARIMAX' and sarimax_model is not None:
    monthly_forecast = sarimax_model.get_forecast(steps=36).predicted_mean
elif best_model_name == 'SeasonalNaive':
    monthly_forecast = pd.Series(
        _seasonal_naive_forecast(ts, steps=36),
        index=pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=36, freq='MS'),
    )
elif best_model_name == 'NaiveLast':
    monthly_forecast = pd.Series(
        _naive_last_forecast(ts, steps=36),
        index=pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=36, freq='MS'),
    )
else:
    if ml_best is None:
        monthly_forecast = pd.Series(
            _naive_last_forecast(ts, steps=36),
            index=pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=36, freq='MS'),
        )
    else:
        monthly_forecast = _forecast_ml_iterative(ml_best, ts, steps=36)

# Visualisation: actual vs forecast
plt.figure(figsize=(12, 4.5))
plt.plot(ts.index, ts.values, label='Historique', linewidth=2)
plt.plot(monthly_forecast.index, monthly_forecast.values, label='Forecast (36 mois)', linewidth=2)
plt.title('Forecast mensuel — Trafic / Congestion')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.tight_layout()
plt.savefig('Objectif1_Forecast_Monthly.png', dpi=200, bbox_inches='tight')
plt.show()

# ============================
# ✅ VISUALISATIONS PROFESSIONNELLES - COMPARAISON DES MODÈLES
# ============================
print("\n" + "=" * 80)
print("📊 COMPARAISON PROFESSIONNELLE DES MODÈLES DE PRÉDICTION")
print("=" * 80 + "\n")

# Tableau de synthèse des modèles
print("📈 RÉSUMÉ DES PERFORMANCES:")
print(model_cmp.to_string(index=False))

# Visualisation 1: Comparaison des métriques en barres
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('white')

metrics_names = ['MAE', 'RMSE', 'MAPE']
colors_models = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_cmp)]

for idx, metric in enumerate(metrics_names):
    ax = axes[idx]
    bars = ax.bar(model_cmp['model'], model_cmp[metric], color=colors_models, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel(metric, fontweight='bold', fontsize=11)
    ax.set_xlabel('Modèle', fontweight='bold', fontsize=11)
    ax.set_title(f'Comparaison - {metric}', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('Objectif1_Model_Comparison_Metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Métriques sauvegardées: Objectif1_Model_Comparison_Metrics.png")
plt.show()

# Visualisation 2: Heatmap de corrélation des métriques
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('white')

# Créer une matrice pour la heatmap
metric_matrix = model_cmp[['model', 'MAE', 'RMSE', 'MAPE']].set_index('model')
metric_matrix_norm = (metric_matrix - metric_matrix.min()) / (metric_matrix.max() - metric_matrix.min())

sns.heatmap(metric_matrix_norm.T, annot=metric_matrix.T.values, fmt='.4f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Métrique (normalisée)'}, ax=ax, linewidths=1, linecolor='gray')
ax.set_title('Heatmap Comparative - Tous les Modèles', fontweight='bold', fontsize=13)
ax.set_xlabel('Modèle', fontweight='bold', fontsize=11)
ax.set_ylabel('Métrique', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('Objectif1_Model_Heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Heatmap sauvegardée: Objectif1_Model_Heatmap.png")
plt.show()

# Résumé final
print("\n" + "=" * 80)
print("🏆 ANALYSE COMPLÉTÉE")
print("=" * 80)
print(f"\n✅ Meilleur modèle sélectionné: {best_model_name}")
print(f"   • MAE: {model_cmp.iloc[0]['MAE']:.6f}")
print(f"   • RMSE: {model_cmp.iloc[0]['RMSE']:.6f}")
print(f"   • MAPE: {model_cmp.iloc[0]['MAPE']:.6f}%")

print(f"\n📊 Comparaison des performances:")
for idx, row in model_cmp.iterrows():
    print(f"   {idx+1}. {row['model']}: RMSE={row['RMSE']:.6f}, MAE={row['MAE']:.6f}, MAPE={row['MAPE']:.2f}%")

print(f"\n📁 Visualisations générées:")
print(f"   • Objectif1_Forecast_Monthly.png")
print(f"   • Objectif1_Model_Comparison_Metrics.png")
print(f"   • Objectif1_Model_Heatmap.png")

print("\n✅ OBJECTIF 1 - ANALYSE COMPLÉTÉE AVEC SUCCÈS!")
print("=" * 80)
