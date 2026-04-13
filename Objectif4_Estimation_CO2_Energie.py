"""OBJECTIF 4: PRÉDIRE LES ÉMISSIONS CO2 ET LA CONSOMMATION D'ÉNERGIE.

Ce script compare plusieurs modèles de régression (avec tuning/CV), puis génère
des prédictions sur 36 mois et des visualisations professionnelles.
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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import LinearSVR

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


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_arr, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("QuantileClipper not fitted")
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

# ============ CHARGEMENT DES DONNÉES ============
print("♻️ OBJECTIF 4: ESTIMER CO2 ET CONSOMMATION ÉNERGÉTIQUE\n")
print("📊 CHARGEMENT DES DONNÉES...")

if _is_colab():
    try:
        from google.colab import files  # type: ignore

        _ = files.upload()
    except Exception:
        pass

# Charger les données
fact_energieconsomation = _read_csv('fact_energieconsomation.csv')
dim_emission_co2 = _read_csv('dim_emission_co2.csv')
dim_energietransport = _read_csv('dim_energietransport.csv')

# CHARGER DIM_ZONE POUR LES NOMS DE ZONES
try:
    dim_zone = _read_csv('dim_zone.csv')
    zone_name_mapping = dict(zip(dim_zone['zone_id'], dim_zone['zone_name'])) if 'zone_name' in dim_zone.columns else {}
    print(f"✓ dim_zone chargé: {dim_zone.shape}")
    print(f"  Colonnes: {list(dim_zone.columns)}")
except:
    dim_zone = None
    zone_name_mapping = {}
    print("⚠️ dim_zone non disponible - utilisation des IDs")

print(f"✓ fact_energieconsomation chargé: {fact_energieconsomation.shape}")
print(f"✓ dim_emission_co2 chargé: {dim_emission_co2.shape}")
print(f"✓ dim_energietransport chargé: {dim_energietransport.shape}")

try:
    fact_pollution = _read_csv('fact_pollution.csv')
    print(f"✓ fact_pollution chargé: {fact_pollution.shape}")
except:
    fact_pollution = None
    print("⚠️ fact_pollution non disponible")

# ============ EXPLORATION DES DONNÉES ============
print("\n📈 EXPLORATION DES DONNÉES...")
print("\nStructure fact_energieconsomation:")
print(f"Colonnes: {list(fact_energieconsomation.columns)}")
print(fact_energieconsomation.head(10))
print(fact_energieconsomation.describe())

print("\nStructure dim_emission_co2:")
print(f"Colonnes: {list(dim_emission_co2.columns)}")
print(dim_emission_co2.head())

print("\nStructure dim_energietransport:")
print(f"Colonnes: {list(dim_energietransport.columns)}")
print(dim_energietransport.head())

# ============ PRÉPARATION DES DONNÉES ============
print("\n🔧 PRÉPARATION DES DONNÉES...")

# Fusion des données
df = fact_energieconsomation.copy()

# Joindre dim_emission_co2 - utiliser seulement les colonnes disponibles
if 'fk_emco2' in df.columns:
    emco2_cols = ['emission_id']
    if 'mode' in dim_emission_co2.columns:
        emco2_cols.append('mode')
    if 'activity_type' in dim_emission_co2.columns:
        emco2_cols.append('activity_type')
    if len(emco2_cols) > 1:
        df = df.merge(dim_emission_co2[emco2_cols], 
                      left_on='fk_emco2', right_on='emission_id', how='left')

# Joindre dim_energietransport - utiliser seulement les colonnes disponibles
if 'fk_energie' in df.columns:
    energie_cols = ['energie_id']
    if 'type_energie' in dim_energietransport.columns:
        energie_cols.append('type_energie')
    if 'source_energie' in dim_energietransport.columns:
        energie_cols.append('source_energie')
    if len(energie_cols) > 1:
        try:
            df = df.merge(dim_energietransport[energie_cols], 
                          left_on='fk_energie', right_on='energie_id', how='left')
        except:
            print("⚠️ Merge dim_energietransport impossible")

# Traiter les données manquantes
print(f"\nValeurs manquantes avant nettoyage:")
print(df.isnull().sum())

# Sélectionner les colonnes numériques
numeric_features_possible = ['activity_value', 'emission_factor', 'co2_kg', 
                             'dechets_kg', 'station_kwh', 'energie_kwh']
numeric_features = [col for col in numeric_features_possible if col in df.columns]

print(f"\nColonnes numériques trouvées: {numeric_features}")

if not numeric_features:
    print("❌ ERREUR: Pas de colonnes numériques trouvées!")
    print(f"Colonnes disponibles: {list(df.columns)}")
    raise ValueError("Aucune colonne numérique pour la modélisation")

df_numeric = df[numeric_features].copy()

# Supprimer les valeurs manquantes
df_numeric = df_numeric.dropna()

# Garder seulement les valeurs positives (CO2 ne doit pas être négatif)
for col in df_numeric.columns:
    df_numeric = df_numeric[df_numeric[col] >= 0]

print(f"\n✓ Dataset après nettoyage: {df_numeric.shape}")
print(f"Statistiques descriptives:")
print(df_numeric.describe())

# ============================
# B — MODEL UNDERSTANDING (obligatoire)
# ============================
print("\n" + "=" * 70)
print("🧠 COMPRÉHENSION DU MODÈLE (B)")
print("=" * 70)
print(
    "Problème: prédire des variables continues (CO2, énergie). "
    "On compare 3 modèles de régression:\n"
    "- Ridge (linéaire régularisé): interprétable, rapide, hypothèse de relation ~linéaire.\n"
    "- Random Forest: non-linéaire, capte interactions, robuste, mais moins interprétable.\n"
    "- LinearSVR: modèle à marge (support vector) adapté aux données hautement dimensionnelles.\n"
    "Choix basé sur RMSE/MAE/R² sur un jeu de test + validation K-Fold." 
)

# ============================
# A — DATA PREPARATION & FEATURE ENGINEERING (Pipeline)
# ============================
print("\n" + "=" * 70)
print("🧹 DATA PREPARATION & FEATURE ENGINEERING (A)")
print("=" * 70)

# Créer des features temporelles si possible
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].notna().any():
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
else:
    df['date'] = pd.NaT

if 'year' not in df.columns:
    df['year'] = pd.Timestamp.today().year
if 'month' not in df.columns:
    df['month'] = 1

target_co2 = 'co2_kg'
target_energy = 'energie_kwh' if 'energie_kwh' in df.columns else None
if target_co2 not in df.columns:
    raise ValueError("Colonne cible manquante: co2_kg")

# Définir features candidates (on exclut les targets des features)
numeric_predictors_possible = ['activity_value', 'emission_factor', 'dechets_kg', 'station_kwh']
numeric_predictors = [c for c in numeric_predictors_possible if c in df.columns]
numeric_predictors += [c for c in ['year', 'month'] if c in df.columns]

categorical_candidates = []
for c in ['zone_id', 'mode', 'type_energie', 'source_energie', 'activity_type']:
    if c in df.columns:
        categorical_candidates.append(c)

feature_cols = list(dict.fromkeys(numeric_predictors + categorical_candidates))
print(f"✓ Features utilisées: {feature_cols}")

numeric_cols = [c for c in feature_cols if c in numeric_predictors]
categorical_cols = [c for c in feature_cols if c in categorical_candidates]

numeric_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('clip', QuantileClipper(lower_q=0.01, upper_q=0.99)),
        ('scaler', RobustScaler()),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ]
)
preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ],
    remainder='drop'
)


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def _train_compare_for_target(target_col: str):
    df_train = df[feature_cols + [target_col]].copy()
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.dropna(subset=[target_col])
    # cibles négatives non plausibles
    df_train = df_train[df_train[target_col] >= 0]
    if len(df_train) < 50:
        print(f"⚠️ Peu de données ({len(df_train)}) pour {target_col}: résultats à interpréter avec prudence")

    X_all = df_train[feature_cols]
    y_all = df_train[target_col].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Modèle 1: Ridge (GridSearch)
    ridge_pipe = Pipeline(steps=[('prep', preprocess), ('reg', Ridge())])
    ridge_grid = {'reg__alpha': [0.1, 1.0, 10.0, 50.0]}
    ridge_search = GridSearchCV(ridge_pipe, ridge_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)

    # Modèle 2: RandomForest (RandomizedSearch)
    rf_pipe = Pipeline(steps=[('prep', preprocess), ('reg', RandomForestRegressor(random_state=42))])
    rf_dist = {
        'reg__n_estimators': [300, 600],
        'reg__max_depth': [None, 8, 14, 20],
        'reg__min_samples_split': [2, 5, 10],
        'reg__min_samples_leaf': [1, 2, 4],
        'reg__max_features': ['sqrt', 0.7, 1.0],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe,
        rf_dist,
        n_iter=12,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
    )

    # Modèle 3: LinearSVR (RandomizedSearch)
    svr_pipe = Pipeline(steps=[('prep', preprocess), ('reg', LinearSVR(max_iter=8000, random_state=42))])
    svr_dist = {
        'reg__C': [0.1, 1.0, 5.0, 10.0],
        'reg__epsilon': [0.0, 0.05, 0.1, 0.2],
    }
    svr_search = RandomizedSearchCV(
        svr_pipe,
        svr_dist,
        n_iter=10,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
    )

    searches = [('Ridge', ridge_search), ('RandomForest', rf_search), ('LinearSVR', svr_search)]
    best_name = None
    best_est = None
    best_rmse = float('inf')
    rows = []

    print("\n" + "-" * 70)
    print(f"🎯 RÉGRESSION (D) — cible: {target_col}")
    print("-" * 70)
    for name, search in searches:
        print(f"🔎 Tuning: {name} ...")
        search.fit(X_train, y_train)
        est = search.best_estimator_
        pred = est.predict(X_test)
        met = _evaluate_regression(y_test, pred)
        rows.append({'model': name, 'best_params': search.best_params_, **met})
        print(pd.Series(rows[-1]).to_string())
        if met['RMSE'] < best_rmse:
            best_rmse = met['RMSE']
            best_name = name
            best_est = est

    results_df = pd.DataFrame(rows).sort_values('RMSE')
    print("\n📌 Comparaison (test set):")
    print(results_df[['model', 'RMSE', 'MAE', 'R2']].to_string(index=False))

    # Visualisation — comparaison des modèles (barres RMSE/MAE/R2)
    try:
        cmp_plot = results_df[['model', 'RMSE', 'MAE', 'R2']].copy()
        cmp_plot = cmp_plot.melt(id_vars='model', var_name='metric', value_name='value')
        plt.figure(figsize=(9.5, 4.8))
        sns.barplot(data=cmp_plot, x='metric', y='value', hue='model')
        plt.title(f"Comparaison des modèles — {target_col} (test set)")
        plt.xlabel('Métrique')
        plt.ylabel('Valeur')
        plt.tight_layout()
        safe_target = target_col.replace('/', '_')
        plt.savefig(f"Objectif4_{safe_target}_Model_Comparison.png", dpi=220, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"⚠️ Plot comparaison modèles indisponible pour {target_col}: {e}")

    if best_est is None or best_name is None:
        raise RuntimeError("Aucun modèle entraîné")
    print(f"\n✅ Modèle retenu pour {target_col}: {best_name} (RMSE={best_rmse:.3f})")

    # Visualisations: résidus + actual vs predicted
    best_pred = best_est.predict(X_test)
    residuals = y_test - best_pred

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(best_pred, residuals, alpha=0.35)
    ax[0].axhline(0, color='red', linestyle='--', linewidth=1)
    ax[0].set_title(f"Residuals plot — {target_col} ({best_name})")
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Residuals')
    ax[1].scatter(y_test, best_pred, alpha=0.35)
    lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    ax[1].plot(lims, lims, 'r--', linewidth=1)
    ax[1].set_title(f"Actual vs Predicted — {target_col} ({best_name})")
    ax[1].set_xlabel('Actual')
    ax[1].set_ylabel('Predicted')
    plt.tight_layout()
    safe_target = target_col.replace('/', '_')
    plt.savefig(f"Objectif4_{safe_target}_Residuals_ActualVsPred.png", dpi=200, bbox_inches='tight')
    plt.show()

    # Explainability: coefficients or feature importance
    try:
        prep = best_est.named_steps['prep']
        reg = best_est.named_steps['reg']
        feature_names = prep.get_feature_names_out()
        if hasattr(reg, 'feature_importances_'):
            imp = pd.DataFrame({'feature': feature_names, 'importance': reg.feature_importances_}).sort_values('importance', ascending=False).head(20)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=imp, y='feature', x='importance', color='#4c72b0')
            plt.title(f"Top 20 Feature Importance — {target_col} ({best_name})")
            plt.tight_layout()
            plt.savefig(f"Objectif4_{safe_target}_FeatureImportance.png", dpi=200, bbox_inches='tight')
            plt.show()
        elif hasattr(reg, 'coef_'):
            coef = np.asarray(reg.coef_).ravel()
            coef_df = pd.DataFrame({'feature': feature_names, 'coef': coef})
            coef_df['abs_coef'] = coef_df['coef'].abs()
            top = coef_df.sort_values('abs_coef', ascending=False).head(20)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top, y='feature', x='coef', palette='vlag')
            plt.title(f"Top 20 Coefficients — {target_col} ({best_name})")
            plt.tight_layout()
            plt.savefig(f"Objectif4_{safe_target}_Coefficients.png", dpi=200, bbox_inches='tight')
            plt.show()
    except Exception as e:
        print(f"⚠️ Explainability indisponible pour {target_col}: {e}")

    return best_est


best_model_co2 = _train_compare_for_target('co2_kg')
best_model_energy = _train_compare_for_target('energie_kwh') if target_energy else None

# ============================
# Génération des prédictions futures (36 mois)
# ============================
print("\n" + "=" * 70)
print("📅 GÉNÉRATION DES PRÉDICTIONS FUTURES (36 MOIS) — MODÈLES")
print("=" * 70)

# Date max depuis l'historique
if df['date'].notna().any():
    max_date = df['date'].max()
else:
    max_date = pd.Timestamp.today().normalize()

future_dates = pd.date_range(start=max_date + timedelta(days=1), periods=36, freq='MS')

zones_uniques = sorted(df['zone_id'].dropna().unique()) if 'zone_id' in df.columns else [1, 2, 3, 4, 5]
modes_uniques = sorted(df['mode'].dropna().unique()) if 'mode' in df.columns else ['Bus', 'Tram', 'Métro']
print(f"✓ Zones: {len(zones_uniques)} | Modes: {len(modes_uniques)}")

# Baselines de features par (zone, mode)
group_keys = [c for c in ['zone_id', 'mode'] if c in df.columns]
numeric_base = [c for c in numeric_predictors_possible if c in df.columns]
baseline = None
if group_keys:
    baseline = df.groupby(group_keys)[numeric_base].median(numeric_only=True).reset_index()

predictions_list = []
for date in future_dates:
    year = int(date.year)
    month = int(date.month)
    for zone in zones_uniques:
        for mode in modes_uniques:
            row = {}
            # Numerics: baseline médiane par groupe, sinon médiane globale
            for col in numeric_base:
                if baseline is not None and group_keys == ['zone_id', 'mode']:
                    m = baseline[(baseline['zone_id'] == zone) & (baseline['mode'] == mode)]
                elif baseline is not None and group_keys == ['zone_id']:
                    m = baseline[baseline['zone_id'] == zone]
                else:
                    m = None

                if m is not None and len(m) > 0 and col in m.columns:
                    row[col] = float(m[col].iloc[0])
                else:
                    row[col] = float(df[col].median()) if col in df.columns else np.nan

            # Time
            if 'year' in df.columns:
                row['year'] = year
            if 'month' in df.columns:
                row['month'] = month

            # Categoricals
            if 'zone_id' in df.columns:
                row['zone_id'] = int(zone)
            if 'mode' in df.columns:
                row['mode'] = str(mode)
            if 'type_energie' in df.columns:
                row['type_energie'] = df['type_energie'].mode().iloc[0] if df['type_energie'].notna().any() else 'N/A'
            if 'source_energie' in df.columns:
                row['source_energie'] = df['source_energie'].mode().iloc[0] if df['source_energie'].notna().any() else 'N/A'
            if 'activity_type' in df.columns:
                row['activity_type'] = df['activity_type'].mode().iloc[0] if df['activity_type'].notna().any() else 'N/A'

            feature_row = pd.DataFrame([row], columns=feature_cols)
            co2_pred = float(best_model_co2.predict(feature_row)[0])
            if best_model_energy is not None:
                energy_pred = float(best_model_energy.predict(feature_row)[0])
            else:
                # fallback si pas de modèle énergie: ratio simple basé sur station_kwh si dispo
                energy_pred = float(row.get('station_kwh', 0.0))

            zone_name = zone_name_mapping.get(zone, f"Zone {int(zone)}")
            predictions_list.append(
                {
                    'zone_id': int(zone),
                    'zone_name': zone_name,
                    'mode_transport': str(mode),
                    'année': year,
                    'mois': month,
                    'date': date,
                    'co2_kg_predite': round(max(0.0, co2_pred), 2),
                    'energie_kwh_predite': round(max(0.0, energy_pred), 2),
                }
            )

predictions_df = pd.DataFrame(predictions_list)

print(f"\n✓ {len(predictions_df):,} prédictions générées")
print(f"✓ Période: {predictions_df['date'].min().strftime('%Y-%m-%d')} à {predictions_df['date'].max().strftime('%Y-%m-%d')}")

# ============ 📊 STATISTIQUES DES PRÉDICTIONS (AMÉLIORÉES) ============
print("\n" + "="*70)
print("📊 STATISTIQUES DES PRÉDICTIONS - DÉTAILLÉES")
print("="*70)

print(f"\n♻️ CO2 - ÉMISSIONS (kg):")
print(f"  Total 36 mois: {predictions_df['co2_kg_predite'].sum()/1000:,.1f} Tonnes")
print(f"  Moyenne/mois: {predictions_df.groupby('date')['co2_kg_predite'].sum().mean()/1000:,.2f} T")
print(f"  Min/Max: {predictions_df['co2_kg_predite'].min():.1f} - {predictions_df['co2_kg_predite'].max():.1f} kg")
print(f"  Écart-type: {predictions_df['co2_kg_predite'].std()/1000:,.2f} T")

print(f"\n⚡ ÉNERGIE - CONSOMMATION (kWh):")
print(f"  Total 36 mois: {predictions_df['energie_kwh_predite'].sum()/1e6:,.2f} MWh")
print(f"  Moyenne/mois: {predictions_df.groupby('date')['energie_kwh_predite'].sum().mean()/1e6:,.2f} MWh")
print(f"  Min/Max: {predictions_df['energie_kwh_predite'].min()/1e3:,.2f} - {predictions_df['energie_kwh_predite'].max()/1e3:,.2f} MWh")
print(f"  Écart-type: {predictions_df['energie_kwh_predite'].std()/1e6:,.2f} MWh")

print(f"\n🗺️ ÉNERGIE par Zone (Top 5):")
energie_by_zone = predictions_df.groupby('zone_name')['energie_kwh_predite'].sum().sort_values(ascending=False).head()
for zone, total in energie_by_zone.items():
    pct = (total / predictions_df['energie_kwh_predite'].sum()) * 100
    print(f"  {zone}: {total/1e6:,.2f} MWh ({pct:.1f}%)")

print(f"\n🚚 ÉNERGIE par Mode (Ranking):")
energie_by_mode_stats = predictions_df.groupby('mode_transport')['energie_kwh_predite'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
for mode in energie_by_mode_stats.index:
    total = energie_by_mode_stats.loc[mode, 'sum']
    avg = energie_by_mode_stats.loc[mode, 'mean']
    pct = (total / predictions_df['energie_kwh_predite'].sum()) * 100
    print(f"  {mode}: Total {total/1e6:,.2f}MWh (Moy: {avg/1e3:,.1f}MWh) - {pct:.1f}%")

print(f"\nAperçu des prédictions (premiers 15 enregistrements):")
print(predictions_df.head(15).to_string(index=False))

# ============ 📈 VISUALISATIONS PROFESSIONNELLES FINALES ============
print("\n\n📈 GÉNÉRATION DES VISUALISATIONS PROFESSIONNELLES...")

# ========== FIGURE 1: ANALYSE CO2 COMPLÈTE ==========
fig_co2 = plt.figure(figsize=(20, 12))
gs_co2 = fig_co2.add_gridspec(3, 2, hspace=0.35, wspace=0.28)

# Plot 1.1: Évolution mensuelle CO2 (Top 3 zones + Total)
ax1 = fig_co2.add_subplot(gs_co2[0, :])
co2_total_by_date = predictions_df.groupby('date')['co2_kg_predite'].sum() / 1000  # en tonnes

ax1.plot(pd.to_datetime(co2_total_by_date.index), co2_total_by_date.values, 
         linewidth=4, color='#c0392b', marker='o', markersize=8, label='Total CO2', zorder=3)
ax1.fill_between(pd.to_datetime(co2_total_by_date.index), co2_total_by_date.values, alpha=0.2, color='#c0392b')

# Ajouter les 3 top zones
co2_by_zone_date = predictions_df.groupby(['date', 'zone_name'])['co2_kg_predite'].sum().unstack(fill_value=0) / 1000
top_3_zones = predictions_df.groupby('zone_name')['co2_kg_predite'].sum().nlargest(3).index
colors_zones = ['#e74c3c', '#e67e22', '#f39c12']

for idx, zone_name in enumerate(top_3_zones):
    if zone_name in co2_by_zone_date.columns:
        ax1.plot(pd.to_datetime(co2_by_zone_date.index), co2_by_zone_date[zone_name].values, 
                linewidth=2.5, alpha=0.7, linestyle='--', color=colors_zones[idx], 
                label=zone_name, marker='s', markersize=5)

ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
ax1.set_ylabel('Émissions CO2 (Tonnes)', fontsize=13, fontweight='bold')
ax1.set_title('📊 ÉVOLUTION MENSUELLE DES ÉMISSIONS CO2 - 36 MOIS', fontsize=15, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.25, linestyle='--', linewidth=1)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.97, edgecolor='black')
ax1.set_facecolor('#f8f9fa')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

# Ajouter stats box
total_co2 = predictions_df['co2_kg_predite'].sum() / 1000
avg_monthly = co2_total_by_date.mean()
peak_co2 = co2_total_by_date.max()
stats_txt = f"Total: {total_co2:,.0f}T | Moyenne/mois: {avg_monthly:,.1f}T | Pic: {peak_co2:,.1f}T"
ax1.text(0.5, -0.18, stats_txt, transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', alpha=0.95, edgecolor='#34495e', linewidth=2))

# Plot 1.2: Top 10 Zones (Horizontal bars) - AVEC NOMS
ax2 = fig_co2.add_subplot(gs_co2[1, 0])
co2_zones_total = predictions_df.groupby('zone_name')['co2_kg_predite'].sum().sort_values(ascending=True).tail(10) / 1000
colors_bar = plt.cm.Reds(np.linspace(0.35, 0.95, len(co2_zones_total)))

bars = ax2.barh(range(len(co2_zones_total)), co2_zones_total.values, color=colors_bar, edgecolor='#2c3e50', linewidth=2.5)
ax2.set_yticks(range(len(co2_zones_total)))
ax2.set_yticklabels(co2_zones_total.index, fontsize=11, fontweight='bold')
ax2.set_xlabel('CO2 Total (Tonnes)', fontsize=12, fontweight='bold')
ax2.set_title('🏆 TOP 10 ZONES - CO2 CUMULÉ (36 mois)', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_facecolor('#f8f9fa')

# Valeurs sur barres
for i, (zone_name, val) in enumerate(zip(co2_zones_total.index, co2_zones_total.values)):
    pct = (val / co2_zones_total.sum()) * 100
    ax2.text(val + 2, i, f'{val:,.0f}T\n({pct:.1f}%)', va='center', fontsize=9, fontweight='bold', color='#c0392b')

# Plot 1.3: Saisonnalité mensuelle
ax3 = fig_co2.add_subplot(gs_co2[1, 1])
co2_mois = predictions_df.groupby('mois')['co2_kg_predite'].mean() / 1000
mois_ext = ['Janv', 'Févr', 'Mars', 'Avri', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc']

bars_mois = ax3.bar(range(1, 13), co2_mois.values, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 12)), 
                    edgecolor='#2c3e50', linewidth=2, alpha=0.8)
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(mois_ext, fontsize=10, fontweight='bold')
ax3.set_ylabel('CO2 Moyen (Tonnes)', fontsize=12, fontweight='bold')
ax3.set_title('📅 SAISONNALITÉ - MOYENNE MENSUELLE', fontsize=13, fontweight='bold')
ax3.axhline(y=co2_mois.mean(), color='#2c3e50', linestyle='--', linewidth=2.5, label=f'Moyenne annuelle: {co2_mois.mean():.1f}T', alpha=0.7)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.legend(fontsize=10, loc='upper left')
ax3.set_facecolor('#f8f9fa')

# Valeurs sur barres
for bar in bars_mois:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 1.4: Heatmap Zone x Mois - AVEC NOMS
ax4 = fig_co2.add_subplot(gs_co2[2, :])
# Créer pivot avec zone_name au lieu de zone_id
predictions_with_names = predictions_df[['zone_name', 'mois', 'co2_kg_predite']].copy()
pivot_co2 = predictions_with_names.groupby(['zone_name', 'mois'])['co2_kg_predite'].sum().unstack(fill_value=0) / 1000

# Top 10 zones par nom
top_zones_names = predictions_df.groupby('zone_name')['co2_kg_predite'].sum().nlargest(10).index
pivot_subset = pivot_co2.loc[top_zones_names]

sns.heatmap(pivot_subset, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax4, 
            cbar_kws={'label': 'CO2 (Tonnes)'}, linewidths=1, linecolor='white',
            xticklabels=mois_ext, yticklabels=pivot_subset.index,
            cbar=True, vmin=pivot_subset.values.min(), vmax=pivot_subset.values.max())
ax4.set_title('🔥 MATRICE ZONE × MOIS - CO2 ÉMIS (Top 10 Zones)', fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('Mois', fontsize=12, fontweight='bold')
ax4.set_ylabel('Zone', fontsize=12, fontweight='bold')

fig_co2.suptitle('♻️ RAPPORT COMPLET - ANALYSE DES ÉMISSIONS CO2 - PRÉDICTIONS 36 MOIS', 
                 fontsize=18, fontweight='bold', y=0.995)
fig_co2.patch.set_facecolor('white')

plt.savefig('Visualisations_CO2_Professional.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ CO2 Professional: Visualisations_CO2_Professional.png")
plt.show()

# ========== FIGURE 2: ANALYSE ÉNERGIE COMPLÈTE (AMÉLIORÉE) ==========
fig_energie = plt.figure(figsize=(24, 16))
gs_energie = fig_energie.add_gridspec(3, 3, hspace=0.38, wspace=0.30)

# Plot 2.1: Évolution mensuelle Énergie (Top 3 modes + Total) - AMÉLIORÉ
ax5 = fig_energie.add_subplot(gs_energie[0, :])
energie_total_by_date = predictions_df.groupby('date')['energie_kwh_predite'].sum() / 1e6  # en MWh

# Ligne principale avec remplissage
ax5.plot(pd.to_datetime(energie_total_by_date.index), energie_total_by_date.values, 
         linewidth=4, color='#2980b9', marker='o', markersize=10, label='Total Énergie', zorder=3)
ax5.fill_between(pd.to_datetime(energie_total_by_date.index), energie_total_by_date.values, alpha=0.2, color='#2980b9')

# Ajouter les limites min/max par mois
monthly_stats = predictions_df.groupby(predictions_df['date'].dt.to_period('M'))['energie_kwh_predite'].agg(['min', 'max']) / 1e6
monthly_stats.index = monthly_stats.index.to_timestamp()
ax5.fill_between(pd.to_datetime(monthly_stats.index), monthly_stats['min'].values, monthly_stats['max'].values, 
                 alpha=0.1, color='#3498db', label='Plage Min-Max mensuelle', zorder=1)

# Top 3 modes
energie_by_mode_date = predictions_df.groupby(['date', 'mode_transport'])['energie_kwh_predite'].sum().unstack(fill_value=0) / 1e6
top_3_modes = predictions_df.groupby('mode_transport')['energie_kwh_predite'].sum().nlargest(3).index
colors_modes = ['#e74c3c', '#f39c12', '#27ae60']

for idx, mode in enumerate(top_3_modes):
    if mode in energie_by_mode_date.columns:
        ax5.plot(pd.to_datetime(energie_by_mode_date.index), energie_by_mode_date[mode].values, 
                linewidth=3, alpha=0.75, linestyle='--', color=colors_modes[idx], 
                label=f'{str(mode)[:18]}', marker='^', markersize=7)

ax5.set_xlabel('Date', fontsize=13, fontweight='bold')
ax5.set_ylabel('Consommation Énergétique (MWh)', fontsize=13, fontweight='bold')
ax5.set_title('⚡ ÉVOLUTION MENSUELLE DE LA CONSOMMATION ÉNERGÉTIQUE - 36 MOIS', fontsize=16, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
ax5.legend(loc='upper left', fontsize=11, framealpha=0.98, edgecolor='black', fancybox=True, shadow=True)
ax5.set_facecolor('#f8f9fa')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

# Ajouter stats box enrichies
total_energie = predictions_df['energie_kwh_predite'].sum() / 1e6
avg_monthly_e = energie_total_by_date.mean()
peak_energie = energie_total_by_date.max()
min_energie = energie_total_by_date.min()
std_energie = energie_total_by_date.std()
stats_txt_e = f"Total: {total_energie:,.0f}MWh | Moy: {avg_monthly_e:,.1f} | Pic: {peak_energie:,.1f} | Min: {min_energie:,.1f} | Std: {std_energie:,.1f}"
ax5.text(0.5, -0.20, stats_txt_e, transform=ax5.transAxes, ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=1.2', facecolor='#d6eaf8', alpha=0.97, edgecolor='#154360', linewidth=2.5))

# Plot 2.2: Distribution par Mode de Transport (Pie chart amélioré)
ax6 = fig_energie.add_subplot(gs_energie[1, 0])
energie_by_mode = predictions_df.groupby('mode_transport')['energie_kwh_predite'].sum().sort_values(ascending=False) / 1e6
colors_pie = plt.cm.Blues(np.linspace(0.35, 0.95, len(energie_by_mode)))

wedges, texts, autotexts = ax6.pie(energie_by_mode.values, labels=[str(m)[:12] for m in energie_by_mode.index], 
                                     autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'}, 
                                     wedgeprops={'edgecolor': '#2c3e50', 'linewidth': 2.5})
# Mettre en gras les pourcentages
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
ax6.set_title('📊 RÉPARTITION PAR MODE\nDE TRANSPORT', fontsize=13, fontweight='bold', pad=10)

# Plot 2.3: Énergie par mode (Barres avec valeurs détaillées)
ax7 = fig_energie.add_subplot(gs_energie[1, 1])
bars_mode = ax7.barh(range(len(energie_by_mode)), energie_by_mode.values, 
                     color=colors_pie, edgecolor='#2c3e50', linewidth=2.5, alpha=0.9)
ax7.set_yticks(range(len(energie_by_mode)))
ax7.set_yticklabels([str(m)[:18] for m in energie_by_mode.index], fontsize=11, fontweight='bold')
ax7.set_xlabel('Énergie Totale (MWh)', fontsize=12, fontweight='bold')
ax7.set_title('📌 CONSOMMATION CUMULÉE\nPAR MODE', fontsize=13, fontweight='bold')
ax7.grid(axis='x', alpha=0.4, linestyle='--', linewidth=1.5)
ax7.set_facecolor('#f8f9fa')

# Valeurs supplémentaires sur barres
for i, (mode, val) in enumerate(zip(energie_by_mode.index, energie_by_mode.values)):
    pct = (val / energie_by_mode.sum()) * 100
    mode_data = predictions_df[predictions_df['mode_transport'] == mode]['energie_kwh_predite'] / 1e6
    avg_val = mode_data.mean()
    ax7.text(val + 15, i, f'{val:,.0f}MWh\n({pct:.1f}%)\nMoy: {avg_val:.1f}', 
            va='center', fontsize=9, fontweight='bold', color='#2980b9')

# Plot 2.4: Box plot - Distribution d'énergie par mode
ax9 = fig_energie.add_subplot(gs_energie[1, 2])
modes_list = [str(m)[:15] for m in predictions_df['mode_transport'].unique()]
energie_by_mode_list = [predictions_df[predictions_df['mode_transport'] == m]['energie_kwh_predite'].values / 1e3 
                        for m in predictions_df['mode_transport'].unique()]

bp = ax9.boxplot(energie_by_mode_list, labels=modes_list, patch_artist=True, 
                 notch=False, showmeans=True, vert=False)

# Colorer les boîtes
colors_box = plt.cm.Blues(np.linspace(0.4, 0.9, len(energie_by_mode_list)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='#2c3e50', linewidth=2)

ax9.set_xlabel('Consommation (MWh)', fontsize=12, fontweight='bold')
ax9.set_title('📊 DISTRIBUTION PAR MODE\n(Box Plot)', fontsize=13, fontweight='bold')
ax9.grid(axis='x', alpha=0.3, linestyle='--')
ax9.set_facecolor('#f8f9fa')

# Plot 2.5: Profil saisonnier amélioré
ax8 = fig_energie.add_subplot(gs_energie[2, :2])
energie_mois = predictions_df.groupby('mois')['energie_kwh_predite'].mean() / 1e3  # kWh à MWh
energie_std = predictions_df.groupby('mois')['energie_kwh_predite'].std() / 1e3

# Barres avec dégradé
bars_energie = ax8.bar(range(1, 13), energie_mois.values, 
                       color=plt.cm.Blues(np.linspace(0.4, 0.95, 12)), 
                       edgecolor='#2c3e50', linewidth=2.5, alpha=0.85, width=0.7)
# Ajouter les erreurs à barres
ax8.errorbar(range(1, 13), energie_mois.values, yerr=energie_std.values, 
            fmt='none', ecolor='#c0392b', elinewidth=2.5, capsize=5, capthick=2, alpha=0.7, label='Écart-type')

ax8.set_xticks(range(1, 13))
ax8.set_xticklabels(mois_ext, fontsize=11, fontweight='bold')
ax8.set_ylabel('Consommation Moyenne (MWh)', fontsize=12, fontweight='bold')
ax8.set_title('📅 SAISONNALITÉ - PROFIL MENSUEL AVEC VARIABILITÉ', 
             fontsize=14, fontweight='bold')

# Ligne de moyenne
avg_energie_line = energie_mois.mean()
ax8.axhline(y=avg_energie_line, color='#e74c3c', linestyle='--', linewidth=3, 
           label=f'Moyenne annuelle: {avg_energie_line:.2f}MWh', alpha=0.8, zorder=2)

# Zones de pics en arrière-plan
summer_idx = [6, 7]
winter_idx = [0, 11]
for idx in summer_idx:
    ax8.axvspan(idx+0.5, idx+1.5, alpha=0.08, color='orange', zorder=0)
    ax8.text(idx+1, energie_mois.max()*0.95, 'ÉTÉ', fontsize=10, fontweight='bold', 
            ha='center', color='#d68910', alpha=0.7)
for idx in winter_idx:
    ax8.axvspan(idx+0.5, idx+1.5, alpha=0.08, color='blue', zorder=0)
    ax8.text(idx+1, energie_mois.max()*0.95, 'HIVER', fontsize=10, fontweight='bold', 
            ha='center', color='#154360', alpha=0.7)

ax8.grid(axis='y', alpha=0.4, linestyle='--', linewidth=1.5)
ax8.legend(fontsize=11, loc='upper right', framealpha=0.98, edgecolor='black')
ax8.set_facecolor('#f8f9fa')

# Valeurs sur barres
for bar in bars_energie:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#154360')

# Plot 2.6: EFFICACITÉ ÉCOLOGIQUE - Énergie vs CO2 (NOUVEAU - CRUCIAL!)
ax11 = fig_energie.add_subplot(gs_energie[2, 2])

# Calculer l'efficacité écologique (ratio CO2/Énergie) par mode
efficiency_data = []
for mode in predictions_df['mode_transport'].unique():
    mode_data = predictions_df[predictions_df['mode_transport'] == mode]
    total_energy = mode_data['energie_kwh_predite'].sum() / 1e6  # MWh
    total_co2 = mode_data['co2_kg_predite'].sum() / 1000  # Tonnes
    
    # Ratio CO2 par MWh consommé (plus bas = plus écologique)
    ratio = total_co2 / total_energy if total_energy > 0 else 0
    efficiency_data.append({'mode': str(mode)[:15], 'ratio': ratio, 'energy': total_energy, 'co2': total_co2})

efficiency_df = pd.DataFrame(efficiency_data).sort_values('ratio')

# Bubble chart : x=énergie, y=CO2, size=ratio
colors_eco = []
for ratio in efficiency_df['ratio']:
    if ratio < 0.1:
        colors_eco.append('#27ae60')  # Vert (très éco)
    elif ratio < 0.3:
        colors_eco.append('#f39c12')  # Orange (moyen)
    else:
        colors_eco.append('#c0392b')  # Rouge (pollutant)

scatter = ax11.scatter(efficiency_df['energy'], efficiency_df['co2'], 
                      s=efficiency_df['ratio']*1000 + 100, c=efficiency_df['ratio'], 
                      cmap='RdYlGn_r', alpha=0.7, edgecolors='#2c3e50', linewidth=2)

# Ajouter labels sur points
for idx, row in efficiency_df.iterrows():
    ax11.annotate(row['mode'], (row['energy'], row['co2']), 
                 fontsize=9, fontweight='bold', ha='center', va='center')

ax11.set_xlabel('Énergie Totale (MWh)', fontsize=12, fontweight='bold')
ax11.set_ylabel('CO2 Total (Tonnes)', fontsize=12, fontweight='bold')
ax11.set_title('♻️ EFFICACITÉ ÉCOLOGIQUE\n(CO2/MWh)', fontsize=13, fontweight='bold')
ax11.grid(True, alpha=0.3, linestyle='--')
ax11.set_facecolor('#f8f9fa')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax11)
cbar.set_label('CO2/MWh (kg/MWh)', fontweight='bold')
fig_energie.suptitle('⚡ RAPPORT COMPLET - ANALYSE DE LA CONSOMMATION ÉNERGÉTIQUE - PRÉDICTIONS 36 MOIS (AMÉLIORÉ)', 
                     fontsize=19, fontweight='bold', y=0.998)
fig_energie.patch.set_facecolor('white')

plt.savefig('Visualisations_Energie_Professional.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Énergie Professional (AMÉLIORÉ avec efficacité écologique): Visualisations_Energie_Professional.png")
plt.show()

print("\n✅ Visualisations professionnelles créées avec succès!")

print("\n✅ OBJECTIF 4 TERMINÉ!")
print("=" * 70)
