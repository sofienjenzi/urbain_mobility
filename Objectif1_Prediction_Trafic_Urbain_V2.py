"""
OBJECTIF 1: PRÉDIRE LE TRAFIC URBAIN PAR ZONE, HEURE, MOIS, ANNÉE
Sorties: prévisions sur 36 mois + visualisations + comparaison de modèles.
"""

# ============ DÉPENDANCES / COMPAT COLAB vs PYTHON ============
from __future__ import annotations

import os
import socket
import warnings
from datetime import timedelta

import psycopg2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")


# ============================
# CONFIG BASE DE DONNÉES (PostgreSQL) — SOURCE DU MODÈLE
# ============================
# Variables supportées (recommandé):
#   - PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, PGSCHEMA
# Tables (optionnel):
#   - FACT_CIRCULATION_TABLE, DIM_TRAFIC_TABLE, DIM_ZONE_TABLE, DIM_TIME_TABLE
#
# Sous Windows, psycopg2 peut lever UnicodeDecodeError quand libpq renvoie un message
# d'erreur localisé (ex: CP1252 avec 'é' => 0xE9). On essaie plusieurs encodages.

os.environ.setdefault("PGCLIENTENCODING", os.getenv("PGCLIENTENCODING", "UTF8"))

# Supporte aussi DB_* (même convention que ton script Objectif4)
_PG_CONN_KWARGS = {
    "host": os.getenv("PGHOST") or os.getenv("DB_HOST") or "localhost",
    "database": os.getenv("PGDATABASE") or os.getenv("DB_NAME") or "urbain_dw",
    "user": os.getenv("PGUSER") or os.getenv("DB_USER") or "postgres",
    "password": os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD") or "admin",
    "port": int(os.getenv("PGPORT") or os.getenv("DB_PORT") or "5432"),
    "connect_timeout": int(os.getenv("PGCONNECT_TIMEOUT", "5")),
}


def _try_open_pg_connection(purpose: str = "utiliser la base PostgreSQL"):
    host = _PG_CONN_KWARGS["host"]
    port = _PG_CONN_KWARGS["port"]

    print(
        "🔌 Connexion PostgreSQL: "
        f"host={host} "
        f"db={_PG_CONN_KWARGS['database']} "
        f"user={_PG_CONN_KWARGS['user']} "
        f"port={port}"
    )

    # Pré-check réseau: évite le UnicodeDecodeError de psycopg2/libpq sous Windows
    # quand PostgreSQL n'écoute pas (message OS localisé en CP1252).
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError as e:
        print(
            f"⚠️ Aucun serveur PostgreSQL n'écoute sur {host}:{port} ({e}). "
            f"Impossible de {purpose}."
        )
        return None

    enc_candidates = [
        os.getenv("PGCLIENTENCODING", "UTF8"),
        "WIN1252",
        "LATIN1",
    ]
    seen: set[str] = set()
    enc_try = [e for e in enc_candidates if not (e in seen or seen.add(e))]

    for enc in enc_try:
        os.environ["PGCLIENTENCODING"] = enc
        try:
            return psycopg2.connect(**_PG_CONN_KWARGS)
        except UnicodeDecodeError:
            # psycopg2/libpq peut planter ici en décodant un message d'erreur localisé
            # (souvent quand l'auth/DB est incorrecte). On tentera un driver alternatif.
            break
        except Exception as e:
            print(
                f"⚠️ Connexion PostgreSQL impossible ({type(e).__name__}: {e}). "
                f"Impossible de {purpose}."
            )
            return None

    print(
        "⚠️ psycopg2 n'arrive pas à se connecter (UnicodeDecodeError). "
        f"Encodages testés: {', '.join(enc_try)}. "
        "Tentative avec un driver alternatif (pg8000) pour obtenir une erreur lisible..."
    )

    try:
        import pg8000  # type: ignore
    except Exception:
        print(
            "❌ Driver alternatif manquant: installe pg8000 puis réessaie:\n"
            "   pip install pg8000\n"
            "(ou corrige la config DB/USER/PASSWORD si tu la connais déjà)"
        )
        return None

    try:
        return pg8000.connect(
            host=host,
            port=int(port),
            database=_PG_CONN_KWARGS["database"],
            user=_PG_CONN_KWARGS["user"],
            password=_PG_CONN_KWARGS["password"],
            timeout=int(_PG_CONN_KWARGS.get("connect_timeout", 5)),
        )
    except Exception as e:
        details = None
        try:
            for a in getattr(e, "args", []) or []:
                if isinstance(a, dict) and "C" in a:
                    details = a
                    break
        except Exception:
            details = None

        # 3D000 = invalid_catalog_name (database does not exist)
        if isinstance(details, dict) and details.get("C") == "3D000":
            db = _PG_CONN_KWARGS["database"]
            msg = details.get("M") or str(e)
            print(f"❌ Base PostgreSQL introuvable: {db} ({msg})")

            # Tente de lister les bases existantes via une base de maintenance
            for maintenance_db in ("postgres", "template1"):
                try:
                    tmp = pg8000.connect(
                        host=host,
                        port=int(port),
                        database=maintenance_db,
                        user=_PG_CONN_KWARGS["user"],
                        password=_PG_CONN_KWARGS["password"],
                        timeout=int(_PG_CONN_KWARGS.get("connect_timeout", 5)),
                    )
                    try:
                        cur = tmp.cursor()
                        cur.execute(
                            "SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname;"
                        )
                        dbs = [r[0] for r in cur.fetchall()]
                    finally:
                        tmp.close()

                    preview = ", ".join(str(x) for x in dbs[:30])
                    more = "" if len(dbs) <= 30 else f" (+{len(dbs) - 30} autres)"
                    print(f"📌 Bases disponibles: {preview}{more}")
                    print(
                        "➡️ Corrige le nom de la base via variable d'environnement, par ex.:\n"
                        "   $env:PGDATABASE=\"<nom_base>\"\n"
                        "   (ou $env:DB_NAME=\"<nom_base>\")"
                    )
                    break
                except Exception:
                    continue

            print(f"Impossible de {purpose}.")
            return None

        print(
            f"⚠️ Connexion PostgreSQL impossible via pg8000 ({type(e).__name__}: {e}). "
            f"Impossible de {purpose}."
        )
        return None


def _qident(name: str) -> str:
    # Quote d'identifiants SQL (schema/table) sans injection.
    return '"' + str(name).replace('"', '""') + '"'


def _read_pg_table(conn, table_name: str) -> pd.DataFrame:
    schema = os.getenv("PGSCHEMA") or os.getenv("DB_SCHEMA") or "public"
    sql = f"SELECT * FROM {_qident(schema)}.{_qident(table_name)};"
    return pd.read_sql_query(sql, conn)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# ============ CHARGEMENT DES DONNÉES ============
print("🚗 OBJECTIF 1: PRÉDIRE LE TRAFIC URBAIN\n")
print("📊 CHARGEMENT DES DONNÉES (PostgreSQL)...")

fact_table = os.getenv("FACT_CIRCULATION_TABLE", "fact_circulation")
dim_trafic_table = os.getenv("DIM_TRAFIC_TABLE", "dim_trafic")
dim_zone_table = os.getenv("DIM_ZONE_TABLE", "dim_zone")
dim_time_table = os.getenv("DIM_TIME_TABLE", "dim_time")

_conn_load = _try_open_pg_connection("charger les données")
if _conn_load is None:
    raise SystemExit(1)

try:
    fact_circulation = _read_pg_table(_conn_load, fact_table)
    dim_trafic = _read_pg_table(_conn_load, dim_trafic_table)
    dim_zone = _read_pg_table(_conn_load, dim_zone_table)
    dim_time = _read_pg_table(_conn_load, dim_time_table)
finally:
    _conn_load.close()

print(f"✓ {fact_table} chargé: {fact_circulation.shape}")
print(f"✓ {dim_trafic_table} chargé: {dim_trafic.shape}")
print(f"✓ {dim_zone_table} chargé: {dim_zone.shape}")
print(f"✓ {dim_time_table} chargé: {dim_time.shape}")

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
# IMPORTANT: la table fact_circulation n'a pas toujours de colonne 'date'.
# On reconstruit alors une datetime en joignant dim_time via fk_time -> time_id.
# NOTE: dans ce dataset, dim_time.date est hors bornes (années 0006..0037) et ne passe pas
# dans pandas (datetime64[ns] supporte ~1677..2262). On privilégie donc annee/mois/jour/heure.
if 'date' in df.columns:
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Si les années sont hors bornes, on force une reconstruction via dim_time.
        years = df['date'].dropna().dt.year
        if len(years) and (years.min() < 1900 or years.max() > 2100):
            df['date'] = pd.NaT
    except Exception:
        df['date'] = pd.NaT

need_rebuild = ('date' not in df.columns) or df['date'].isna().all()
if need_rebuild:
    if 'fk_time' in df.columns and 'time_id' in dim_time.columns:
        dt_cols = ['time_id']
        for c in ('annee', 'mois', 'jour', 'heure', 'date'):
            if c in dim_time.columns:
                dt_cols.append(c)

        dt = dim_time[dt_cols].drop_duplicates('time_id')
        df = df.merge(dt, left_on='fk_time', right_on='time_id', how='left')

        # Rebuild from annee/mois/jour if available.
        if 'annee' in df.columns and 'mois' in df.columns:
            year = pd.to_numeric(df['annee'], errors='coerce').astype('Int64')
            month = pd.to_numeric(df['mois'], errors='coerce').astype('Int64')
            if 'jour' in df.columns:
                day = pd.to_numeric(df['jour'], errors='coerce').astype('Int64')
            else:
                day = pd.Series([1] * len(df), index=df.index, dtype='Int64')
            # Clamp day into 1..28 to avoid invalid dates (e.g., February 30).
            day = day.fillna(1).clip(lower=1, upper=28)
            df['date'] = pd.to_datetime({'year': year, 'month': month, 'day': day}, errors='coerce')

            if 'heure' in df.columns:
                # heure is often 'HH:MM:SS'
                td = pd.to_timedelta(df['heure'].astype(str), errors='coerce')
                df['date'] = df['date'] + td.fillna(pd.Timedelta(0))
        else:
            # Fallback: try using dim_time.date if present and in-bounds.
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

        df = df.drop(columns=['time_id'], errors='ignore')

    if 'date' not in df.columns or df['date'].isna().all():
        print("⚠️ Impossible de reconstruire une date fiable via dim_time, création d'une série temporelle synthétique")
        df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='H')

# Final fallback if 'date' exists but is mostly NaT
if df['date'].isna().all():
    print("⚠️ Dates invalides, création d'une série temporelle synthétique")
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

# NB: on compare toujours 2 baselines (SeasonalNaive, NaiveLast) et, si possible,
# SARIMAX et un modèle ML sur lags. Donc 2 à 4 modèles selon les dépendances et la taille de la série.

df_model = df_model.sort_values('date').copy()
ts = df_model.set_index('date')[target_col].resample('MS').mean().dropna()

# L'utilisateur veut travailler "sur 12 mois": par défaut on garde les 12 derniers mois.
try:
    history_months = int(os.getenv('OBJ1_HISTORY_MONTHS', '12'))
except Exception:
    history_months = 12
if history_months > 0 and len(ts) > history_months:
    ts = ts.tail(history_months)

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


# -------- Baseline: Naive avec dérive (trend) — extrapolation linéaire simple
def _naive_drift_forecast(train: pd.Series, steps: int) -> np.ndarray:
    if len(train) < 2:
        return _naive_last_forecast(train, steps)
    y0 = float(train.iloc[0])
    y1 = float(train.iloc[-1])
    slope = (y1 - y0) / max(1, (len(train) - 1))
    return np.asarray([y1 + (i + 1) * slope for i in range(steps)], dtype=float)


# -------- Baseline: Moving Average (moyenne glissante des derniers mois)
def _moving_average_forecast(train: pd.Series, steps: int, window: int = 3) -> np.ndarray:
    if len(train) == 0:
        return np.zeros(steps, dtype=float)
    w = int(max(1, min(window, len(train))))
    level = float(train.iloc[-w:].mean())
    return np.full(shape=(steps,), fill_value=level, dtype=float)


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


# Baselines: on garantit des modèles distincts même si la série est courte.
last_pred = _naive_last_forecast(train_ts, steps=len(test_ts))
last_metrics = _eval_forecast(test_ts, last_pred)
print("\n📌 NaiveLast metrics:")
print(pd.Series(last_metrics).to_string())

drift_pred = _naive_drift_forecast(train_ts, steps=len(test_ts))
drift_metrics = _eval_forecast(test_ts, drift_pred)
print("\n📌 NaiveDrift metrics:")
print(pd.Series(drift_metrics).to_string())

ma_pred = _moving_average_forecast(train_ts, steps=len(test_ts), window=3)
ma_metrics = _eval_forecast(test_ts, ma_pred)
print("\n📌 MovingAverage(3) metrics:")
print(pd.Series(ma_metrics).to_string())

naive_metrics = None
if len(train_ts) >= 12:
    naive_pred = _seasonal_naive_forecast(train_ts, steps=len(test_ts))
    # Si SeasonalNaive est identique à NaiveLast (cas fréquent quand aucune ref année-1 n'existe),
    # on préfère l'ignorer pour éviter une comparaison trompeuse.
    if np.allclose(naive_pred, last_pred, rtol=0, atol=1e-12):
        print("\n⚠️ SeasonalNaive dégénère en NaiveLast (pas de saisonnalité annuelle exploitable). Ignoré.")
    else:
        naive_metrics = _eval_forecast(test_ts, naive_pred)
        print("\n📌 Seasonal Naive metrics:")
        print(pd.Series(naive_metrics).to_string())
else:
    print("\n⚠️ SeasonalNaive ignoré: série < 12 mois (pas de référence annuelle).")


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
model_rows.append({'model': 'NaiveLast', **last_metrics})
model_rows.append({'model': 'NaiveDrift', **drift_metrics})
model_rows.append({'model': 'MovingAvg_3', **ma_metrics})
if naive_metrics is not None:
    model_rows.append({'model': 'SeasonalNaive', **naive_metrics})
if sarimax_metrics is not None:
    model_rows.append({'model': 'SARIMAX', **sarimax_metrics})
if ml_metrics is not None:
    model_rows.append({'model': 'ML_Lags_GBR', **ml_metrics})
model_cmp = pd.DataFrame(model_rows).sort_values('RMSE')
print("\n✅ Comparaison modèles:")
print(model_cmp.to_string(index=False))

# Si la série mensuelle est très courte, les modèles baselines donnent souvent une prévision plate.
# Pour éviter un forecast identique chaque mois, on privilégie NaiveDrift (< 12 mois).
short_series = len(ts) < 12
if short_series and 'NaiveDrift' in set(model_cmp['model'].tolist()):
    best_model_name = 'NaiveDrift'
    print("\n⚠️ Série < 12 mois: NaiveDrift choisi pour éviter une prévision constante.")
else:
    # Sélection robuste: en série courte, plusieurs baselines peuvent avoir des métriques très proches
    # (voire identiques). On applique un tie-break qui privilégie les modèles non-plats.
    best_rmse = float(model_cmp['RMSE'].min()) if len(model_cmp) else float('inf')
    eps = 1e-9
    candidates = set(model_cmp.loc[model_cmp['RMSE'] <= best_rmse + eps, 'model'].tolist())
    preference = [
        'ML_Lags_GBR',
        'SARIMAX',
        'SeasonalNaive',
        'NaiveDrift',
        'MovingAvg_3',
        'NaiveLast',
    ]
    best_model_name = None
    for name in preference:
        if name in candidates:
            best_model_name = name
            break
    if best_model_name is None:
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
elif best_model_name == 'NaiveDrift':
    monthly_forecast = pd.Series(
        _naive_drift_forecast(ts, steps=36),
        index=pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=36, freq='MS'),
    )
elif best_model_name == 'NaiveLast':
    monthly_forecast = pd.Series(
        _naive_last_forecast(ts, steps=36),
        index=pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=36, freq='MS'),
    )
elif best_model_name == 'MovingAvg_3':
    monthly_forecast = pd.Series(
        _moving_average_forecast(ts, steps=36, window=3),
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
colors_models = sns.color_palette(n_colors=len(model_cmp))

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
metric_matrix = metric_matrix.apply(pd.to_numeric, errors='coerce')

# Normalisation robuste: évite division par zéro quand toutes les valeurs sont identiques
mm_min = metric_matrix.min(axis=0)
mm_max = metric_matrix.max(axis=0)
denom = (mm_max - mm_min).replace(0, 1.0)
metric_matrix_norm = (metric_matrix - mm_min) / denom

if (mm_max - mm_min).eq(0).all():
    print("⚠️ Toutes les métriques sont identiques entre modèles: heatmap uniforme (normalisation sans division par zéro).")

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


def _mlops_export_objective1() -> None:
    if os.getenv("MLOPS_EXPORT", "0").strip().lower() not in ("1", "true", "yes"):
        return

    from mlops.registry import register_model

    # Guard: ensure the expected globals exist
    if "best_model_name" not in globals() or "ts" not in globals():
        print("⚠️ MLOps export ignoré: variables manquantes (best_model_name/ts).")
        return

    best_kind = str(globals().get("best_model_name"))
    ts_local = globals().get("ts")
    if ts_local is None or len(ts_local) == 0:
        print("⚠️ MLOps export ignoré: série historique vide.")
        return

    # Keep only what's needed for serving baselines and/or iterative lags.
    lags = list(globals().get("lags_used") or [])
    max_needed = max([12, 3, (max(lags) if lags else 1), 48])
    hist_tail = ts_local.tail(int(max_needed))

    meta = {
        "model_kind": best_kind,
        "target_col": str(globals().get("target_col", "")),
        "metrics": (globals().get("model_cmp").to_dict(orient="records") if "model_cmp" in globals() else None),
        "lags": lags,
        "history_index": [str(x) for x in pd.to_datetime(hist_tail.index).to_pydatetime()],
        "history_values": [float(x) for x in hist_tail.values],
        "series_length": int(len(ts_local)),
        "first_value": float(ts_local.iloc[0]),
        "last_value": float(ts_local.iloc[-1]),
    }

    # Choose an exportable model object. For baselines, model object is unused by serving.
    model_obj = {"bundle": True, "kind": best_kind}
    if best_kind == "SARIMAX" and "sarimax_model" in globals() and globals().get("sarimax_model") is not None:
        model_obj = globals().get("sarimax_model")
    elif best_kind == "ML_Lags_GBR" and "ml_best" in globals() and globals().get("ml_best") is not None:
        model_obj = globals().get("ml_best")
    else:
        # Keep the model bundle as metadata-only.
        pass

    try:
        entry = register_model("objective1", model_obj, meta)
        print({"status": "success", "objective": "objective1", "version": entry.version})
    except Exception as e:
        # If SARIMAX is not picklable in a given environment, fallback to metadata-only.
        try:
            meta["export_warning"] = f"Primary export failed ({type(e).__name__}: {e}); fallback metadata-only bundle."
            entry = register_model("objective1", {"bundle": True, "kind": meta["model_kind"]}, meta)
            print({"status": "success", "objective": "objective1", "version": entry.version, "warning": meta["export_warning"]})
        except Exception as e2:
            print({"status": "error", "message": f"MLOps export failed: {e2}"})


_mlops_export_objective1()
