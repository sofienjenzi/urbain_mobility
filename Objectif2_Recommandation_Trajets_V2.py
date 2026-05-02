"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   OBJECTIF 2: MEILLEURS TRAJETS ENRICHIS                       ║
║           Analyse Professionnelle avec Noms d'Arrêts et Villes                ║
║                                                                                ║
║  📊 Prédictions et recommandations (36 mois futurs)                            ║
║  🚌 Recommandations avec détails géographiques complets                        ║
║  📈 Visualisations professionnelles et comparaisons de modèles                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

# ============ DÉPENDANCES / COMPAT COLAB vs PYTHON ============
from __future__ import annotations

import os
import warnings
from datetime import timedelta
from pathlib import Path
from difflib import get_close_matches

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    davies_bouldin_score,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# ============================
# CONFIG BASE DE DONNÉES (PostgreSQL) — SOURCE DU MODÈLE
# ============================
DB_ENV_VARS = [
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
]

# Valeurs par défaut demandées (surcharge possible via variables d'environnement DB_*)
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "urbain_dw")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "admin")
os.environ.setdefault("DB_SCHEMA", "public")

# Accélération (par défaut activée). Mettre FAST_MODE=0 pour le mode complet.
FAST_MODE = os.getenv("FAST_MODE", "1").strip().lower() not in ("0", "false", "no")

_ENGINE = None


def _qident(name: str) -> str:
    """Quote an SQL identifier safely (schema/table/column)."""
    return '"' + str(name).replace('"', '""') + '"'


def _db_enabled() -> bool:
    return all(os.getenv(v) for v in DB_ENV_VARS)


def _get_engine():
    """Crée un engine SQLAlchemy pour PostgreSQL."""
    try:
        from sqlalchemy import create_engine  # type: ignore
        from sqlalchemy.engine import URL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SQLAlchemy n'est pas installé. Installe: pip install sqlalchemy psycopg2-binary"
        ) from e

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port) if str(port).isdigit() else port,
        database=name,
    )
    return create_engine(url, pool_pre_ping=True)


def _read_sql(sql: str) -> pd.DataFrame:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _get_engine()
    try:
        conn = _ENGINE.raw_connection()
        try:
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()
    except Exception:
        return pd.read_sql_query(sql, _ENGINE)


def _db_limit_rows() -> int | None:
    v = os.getenv("DB_LIMIT_ROWS", "").strip()
    if not v:
        return None
    try:
        n = int(v)
        return n if n > 0 else None
    except Exception:
        return None


def _list_tables(schema: str = "public") -> list[str]:
    sql = (
        "SELECT table_name FROM information_schema.tables "
        f"WHERE table_schema='{schema}' AND table_type='BASE TABLE' ORDER BY table_name;"
    )
    d = _read_sql(sql)
    if 'table_name' not in d.columns:
        return []
    return [str(x) for x in d['table_name'].dropna().tolist()]


def _resolve_table(
    preferred: str,
    schema: str = "public",
    like_patterns: list[str] | None = None,
) -> str | None:
    """Résout un nom de table réel dans la DB."""
    tables = _list_tables(schema=schema)
    if preferred in tables:
        return preferred

    if like_patterns:
        for pat in like_patterns:
            pat_sql = pat.replace("'", "''")
            sql = (
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema='{schema}' AND table_name ILIKE '{pat_sql}' "
                "ORDER BY table_name;"
            )
            m = _read_sql(sql)
            if 'table_name' in m.columns and len(m) > 0:
                return str(m['table_name'].iloc[0])

    close = get_close_matches(preferred, tables, n=1, cutoff=0.65)
    return close[0] if close else None


def _read_table(table: str, schema: str = "public", limit: int | None = None) -> pd.DataFrame:
    lim = limit if limit is not None else _db_limit_rows()
    if lim is not None:
        return _read_sql(f"SELECT * FROM {_qident(schema)}.{_qident(table)} LIMIT {int(lim)};")
    return _read_sql(f"SELECT * FROM {_qident(schema)}.{_qident(table)};")


# Configuration style professionnel
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# ============ CHARGEMENT DES DONNÉES ============
print("\n" + "="*80)
print("🗺️ OBJECTIF 2: MEILLEURS TRAJETS - ANALYSE PROFESSIONNELLE")
print("="*80 + "\n")

print("📊 CHARGEMENT DES DONNÉES (PostgreSQL)...")

if not _db_enabled():
    raise RuntimeError(
        "Connexion PostgreSQL non configurée. Variables requises: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD. "
        "(Par défaut: localhost/5432/urbain_dw/postgres/admin)"
    )

schema = os.getenv("DB_SCHEMA", "public")

fact_table = _resolve_table(
    preferred=os.getenv("FACT_CIRCULATION_TABLE", "fact_circulation"),
    schema=schema,
    like_patterns=["fact_circu%", "fact_circulation%"],
)
if fact_table is None:
    raise RuntimeError(f"Table fact_circulation introuvable dans le schéma {schema}.")

# Dimensions (optionnelles)
dim_zone_table = _resolve_table(os.getenv("DIM_ZONE_TABLE", "dim_zone"), schema=schema, like_patterns=["dim_zone%"])
dim_ligne_table = _resolve_table(os.getenv("DIM_LIGNE_TABLE", "dim_ligne"), schema=schema, like_patterns=["dim_ligne%", "dim_line%"])
dim_segment_table = _resolve_table(os.getenv("DIM_SEGMENT_TABLE", "dim_segment"), schema=schema, like_patterns=["dim_segment%"])
dim_arret_table = _resolve_table(os.getenv("DIM_ARRET_TABLE", "dim_arret"), schema=schema, like_patterns=["dim_arret%", "dim_stop%", "dim_station%"])
dim_trafic_table = _resolve_table(os.getenv("DIM_TRAFIC_TABLE", "dim_trafic"), schema=schema, like_patterns=["dim_trafic%"])
dim_time_table = _resolve_table(os.getenv("DIM_TIME_TABLE", "dim_time"), schema=schema, like_patterns=["dim_time%", "dim_temps%"])

print("\n📌 Tables DB utilisées:")
print(f"  - fact: {schema}.{fact_table}")
print(f"  - dim_zone: {schema}.{dim_zone_table}" if dim_zone_table else "  - dim_zone: (non trouvée)")
print(f"  - dim_ligne: {schema}.{dim_ligne_table}" if dim_ligne_table else "  - dim_ligne: (non trouvée)")
print(f"  - dim_segment: {schema}.{dim_segment_table}" if dim_segment_table else "  - dim_segment: (non trouvée)")
print(f"  - dim_arret: {schema}.{dim_arret_table}" if dim_arret_table else "  - dim_arret: (non trouvée)")
print(f"  - dim_trafic: {schema}.{dim_trafic_table}" if dim_trafic_table else "  - dim_trafic: (non trouvée)")
print(f"  - dim_time: {schema}.{dim_time_table}" if dim_time_table else "  - dim_time: (non trouvée)")

fact_circulation = _read_table(fact_table, schema=schema)
print(f"✓ fact_circulation chargé: {fact_circulation.shape}")

dim_zone = _read_table(dim_zone_table, schema=schema) if dim_zone_table else None
if dim_zone is not None:
    print(f"✓ dim_zone chargé: {dim_zone.shape}")

dim_ligne = _read_table(dim_ligne_table, schema=schema) if dim_ligne_table else None
if dim_ligne is not None:
    print(f"✓ dim_ligne chargé: {dim_ligne.shape}")

dim_segment = _read_table(dim_segment_table, schema=schema) if dim_segment_table else None
if dim_segment is not None:
    print(f"✓ dim_segment chargé: {dim_segment.shape}")

dim_arret = _read_table(dim_arret_table, schema=schema) if dim_arret_table else None
if dim_arret is not None:
    print(f"✓ dim_arret chargé: {dim_arret.shape}")

dim_trafic = _read_table(dim_trafic_table, schema=schema) if dim_trafic_table else None
if dim_trafic is not None:
    print(f"✓ dim_trafic chargé: {dim_trafic.shape}")

dim_time = _read_table(dim_time_table, schema=schema) if dim_time_table else None
if dim_time is not None:
    print(f"✓ dim_time chargé: {dim_time.shape}")

# ============ EXPLORATION DES DONNÉES ============
print("\n📈 EXPLORATION DES DONNÉES...\n")

if fact_circulation is not None:
    print(f"Colonnes fact_circulation: {list(fact_circulation.columns)}")
    print(fact_circulation.head(2))
    
if dim_zone is not None:
    print(f"\nColonnes dim_zone: {list(dim_zone.columns)}")
    print(dim_zone.head(2))

if dim_segment is not None:
    print(f"\nColonnes dim_segment: {list(dim_segment.columns)}")
    print(dim_segment.head(2))

if dim_arret is not None:
    print(f"\nColonnes dim_arret: {list(dim_arret.columns)}")
    print(dim_arret.head(2))

# ============ PRÉPARATION DES DONNÉES ============
print("\n🔧 PRÉPARATION DES DONNÉES...\n")

df = fact_circulation.copy() if fact_circulation is not None else pd.DataFrame()

# Jointure temps -> date si la fact n'a pas de colonne date
if 'date' not in df.columns and dim_time is not None and len(dim_time) > 0:
    try:
        # Clé fact (ex: fk_time) + clé dimension (ex: time_id)
        time_fk = next(
            (c for c in df.columns if 'time' in c.lower() and ('fk' in c.lower() or c.lower().endswith('_id'))),
            None,
        )
        time_pk = next((c for c in dim_time.columns if 'time' in c.lower() and 'id' in c.lower()), None)

        if time_fk and time_pk:
            dtmp_cols = [time_pk]
            for c in ['annee', 'mois', 'jour', 'heure', 'date']:
                if c in dim_time.columns:
                    dtmp_cols.append(c)

            dtmp = dim_time[dtmp_cols].drop_duplicates(time_pk).copy()

            # Cast join keys to avoid merge mismatch (float vs int)
            df[time_fk] = pd.to_numeric(df[time_fk], errors='coerce').astype('Int64')
            dtmp[time_pk] = pd.to_numeric(dtmp[time_pk], errors='coerce').astype('Int64')

            # Construire une date fiable: préférer annee/mois/jour (+ heure) si disponibles
            if all(c in dtmp.columns for c in ['annee', 'mois', 'jour']):
                y = pd.to_numeric(dtmp['annee'], errors='coerce')
                m = pd.to_numeric(dtmp['mois'], errors='coerce')
                d = pd.to_numeric(dtmp['jour'], errors='coerce')
                dtmp['date'] = pd.to_datetime({'year': y, 'month': m, 'day': d}, errors='coerce')

                if 'heure' in dtmp.columns:
                    htd = pd.to_timedelta(dtmp['heure'].astype(str), errors='coerce')
                    dtmp['date'] = dtmp['date'] + htd.fillna(pd.Timedelta(0))
            else:
                date_col = next((c for c in dtmp.columns if 'date' in c.lower()), None)
                if date_col and date_col != 'date':
                    dtmp = dtmp.rename(columns={date_col: 'date'})
                dtmp['date'] = pd.to_datetime(dtmp.get('date'), errors='coerce')

            df = df.merge(dtmp[[time_pk, 'date']], left_on=time_fk, right_on=time_pk, how='left')
            print(f"✓ Jointure dim_time appliquée: {time_fk} -> {time_pk} (date)")
    except Exception:
        pass

# Afficher les colonnes disponibles pour debug
print("📋 Colonnes dans fact_circulation:", df.columns.tolist() if len(df) > 0 else "Vide")
if dim_segment is not None:
    print("📋 Colonnes dans dim_segment:", dim_segment.columns.tolist())
if dim_arret is not None:
    print("📋 Colonnes dans dim_arret:", dim_arret.columns.tolist())
if dim_ligne is not None:
    print("📋 Colonnes dans dim_ligne:", dim_ligne.columns.tolist())

# Identifier les colonnes clé
segment_key = None
stop_from_key = None
stop_to_key = None
line_key = None
arret_id_key = None
arret_nom_key = None

# Trouver les clés dans fact_circulation
if len(df) > 0:
    for col in df.columns:
        if 'segment' in col.lower() and 'id' in col.lower():
            segment_key = col
            print(f"✓ Clé segment trouvée: {segment_key}")
        if 'stop' in col.lower() and 'from' in col.lower():
            stop_from_key = col
            print(f"✓ Clé stop_from trouvée: {stop_from_key}")
        if 'stop' in col.lower() and 'to' in col.lower():
            stop_to_key = col
            print(f"✓ Clé stop_to trouvée: {stop_to_key}")
        if 'line' in col.lower() and 'id' in col.lower():
            line_key = col
            print(f"✓ Clé line trouvée: {line_key}")

# Trouver les clés dans dim_arret
if dim_arret is not None and len(dim_arret) > 0:
    for col in dim_arret.columns:
        if 'id' in col.lower() and arret_id_key is None:
            arret_id_key = col
        if 'nom' in col.lower() and arret_nom_key is None:
            arret_nom_key = col

print(f"\n🔑 Clés trouvées:")
print(f"  segment_key: {segment_key}")
print(f"  stop_from_key: {stop_from_key}")
print(f"  stop_to_key: {stop_to_key}")
print(f"  line_key: {line_key}")
print(f"  arret_id_key: {arret_id_key}")
print(f"  arret_nom_key: {arret_nom_key}")

# Fusionner avec dim_ligne
if dim_ligne is not None and len(df) > 0 and line_key:
    print("  • Fusion avec dim_ligne...")
    try:
        ligne_id_col = [col for col in dim_ligne.columns if 'id' in col.lower()][0]
        merge_cols = ['line_nom', 'mode']
        merge_cols = [col for col in merge_cols if col in dim_ligne.columns]
        
        if line_key in df.columns and ligne_id_col in dim_ligne.columns:
            df = df.merge(dim_ligne[[ligne_id_col] + merge_cols], 
                         left_on=line_key, right_on=ligne_id_col, how='left')
            print(f"     ✓ Fusionné ligne")
    except Exception as e:
        print(f"     ⚠️ Erreur fusion ligne: {e}")

# Fusionner avec dim_arret pour stop_from
if dim_arret is not None and len(df) > 0 and stop_from_key and arret_id_key and arret_nom_key:
    print("  • Fusion avec dim_arret (stop_from)...")
    try:
        arret_cols = [arret_nom_key]
        if 'ville' in dim_arret.columns:
            arret_cols.append('ville')
        
        arret_from = dim_arret[[arret_id_key] + arret_cols].copy()
        arret_from.columns = [arret_id_key, 'stop_from_nom'] + (['ville_depart'] if 'ville' in arret_cols else [])
        
        df = df.merge(arret_from, left_on=stop_from_key, right_on=arret_id_key, how='left', suffixes=('', '_from'))
        print(f"     ✓ Fusionné stop_from")
    except Exception as e:
        print(f"     ⚠️ Erreur fusion stop_from: {e}")

# Fusionner avec dim_arret pour stop_to
if dim_arret is not None and len(df) > 0 and stop_to_key and arret_id_key and arret_nom_key:
    print("  • Fusion avec dim_arret (stop_to)...")
    try:
        arret_cols = [arret_nom_key]
        if 'ville' in dim_arret.columns:
            arret_cols.append('ville')
        
        arret_to = dim_arret[[arret_id_key] + arret_cols].copy()
        arret_to.columns = [arret_id_key, 'stop_to_nom'] + (['ville_arrivee'] if 'ville' in arret_cols else [])
        
        df = df.merge(arret_to, left_on=stop_to_key, right_on=arret_id_key, how='left', suffixes=('', '_to'))
        print(f"     ✓ Fusionné stop_to")
    except Exception as e:
        print(f"     ⚠️ Erreur fusion stop_to: {e}")

# Créer les colonnes temporelles
if 'date' not in df.columns:
    df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='H')
else:
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except:
        df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='H')

    if df['date'].isna().all():
        df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='H')

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Créer des colonnes par défaut si elles n'existent pas
if 'line_id' not in df.columns:
    df['line_id'] = range(1, len(df) + 1)
if 'line_nom' not in df.columns:
    df['line_nom'] = 'Ligne ' + df['line_id'].astype(str)
if 'mode' not in df.columns:
    df['mode'] = 'Transport'
if segment_key is None or segment_key not in df.columns:
    if 'fk_trafic' in df.columns:
        df['segment_id'] = pd.to_numeric(df['fk_trafic'], errors='coerce')
        df['segment_id'] = df['segment_id'].fillna(pd.Series(np.arange(1, len(df) + 1), index=df.index))
        df['segment_id'] = df['segment_id'].astype(int)
    else:
        df['segment_id'] = range(1, len(df) + 1)
else:
    df['segment_id'] = df[segment_key]
if 'stop_from_nom' not in df.columns:
    df['stop_from_nom'] = 'Arrêt_' + (df[stop_from_key].astype(str) if stop_from_key else '0')
if 'stop_to_nom' not in df.columns:
    df['stop_to_nom'] = 'Arrêt_' + (df[stop_to_key].astype(str) if stop_to_key else '0')
if 'ville_depart' not in df.columns:
    df['ville_depart'] = 'Départ'
if 'ville_arrivee' not in df.columns:
    df['ville_arrivee'] = 'Arrivée'
if stop_from_key is None or stop_from_key not in df.columns:
    df['stop_from'] = range(1, len(df) + 1)
else:
    df['stop_from'] = df[stop_from_key]
if stop_to_key is None or stop_to_key not in df.columns:
    df['stop_to'] = range(2, len(df) + 2)
else:
    df['stop_to'] = df[stop_to_key]

print(f"\n✓ Dataset préparé: {df.shape}")
print(f"✓ Exemples line_nom: {df['line_nom'].unique()[:3]}")
print(f"✓ Exemples stop_from_nom: {df['stop_from_nom'].unique()[:3]}")
print(f"✓ Exemples stop_to_nom: {df['stop_to_nom'].unique()[:3]}")

# Sélectionner les colonnes numériques
numeric_cols_possible = ['vitesse', 'vitesse_moyenne', 'temps_trajet_min', 'temps_trajet', 
                         'congestion_index', 'volume_trafic']
numeric_cols = [col for col in numeric_cols_possible if col in df.columns]

if not numeric_cols:
    print("⚠️ Aucune colonne numérique pré-définie trouvée, recherche manuelle...")
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    numeric_cols = [col for col in numeric_cols if col not in 
                   ['fk_trafic', 'fk_time', 'fk_zone', 'segment_id', 'line_id', 'stop_sk', 'sk_line']]

print(f"\n✓ Colonnes numériques trouvées: {numeric_cols}")

# ============ PRÉPARATION DU DATASET POUR LE MODÈLE ============
print("\n🎯 PRÉPARATION DU DATASET POUR LE MODÈLE...\n")

# Sélectionner les colonnes requises
required_cols = numeric_cols.copy()
required_cols.extend(['segment_id', 'date', 'year', 'month'])

# Ajouter les colonnes de détails si elles existent
detail_cols = ['stop_from_nom', 'stop_to_nom', 'ville_depart', 'ville_arrivee', 'line_nom', 'mode', 'line_id', 'stop_from', 'stop_to']
for col in detail_cols:
    if col in df.columns:
        required_cols.append(col)

available_cols = [col for col in required_cols if col in df.columns]

df_model = df[available_cols].copy()

# A — Outliers: clipping quantiles (1%–99%) sur colonnes numériques
for col in numeric_cols:
    if col in df_model.columns:
        vals = pd.to_numeric(df_model[col], errors='coerce')
        lo, hi = vals.quantile([0.01, 0.99]).values
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            df_model[col] = vals.clip(lo, hi)

# Remplir les valeurs manquantes
print("  • Traitement des valeurs manquantes...")
for col in numeric_cols:
    if col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
        df_model[col] = df_model[col].fillna(df_model[col].mean())

# Remplir les colonnes texte
for col in ['stop_from_nom', 'stop_to_nom', 'ville_depart', 'ville_arrivee', 'line_nom', 'mode']:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna('N/A')
        df_model[col] = df_model[col].replace(['', None, 'NaN', 'nan'], 'N/A')
    else:
        # Créer la colonne si elle n'existe pas
        if col == 'stop_from_nom':
            df_model[col] = 'Arrêt Départ'
        elif col == 'stop_to_nom':
            df_model[col] = 'Arrêt Arrivée'
        elif col == 'ville_depart':
            df_model[col] = 'Ville Départ'
        elif col == 'ville_arrivee':
            df_model[col] = 'Ville Arrivée'
        elif col == 'line_nom':
            df_model[col] = 'Ligne Inconnue'
        elif col == 'mode':
            df_model[col] = 'Transport'

df_model = df_model.dropna(subset=numeric_cols, how='all')

print(f"✓ Dataset modèle: {df_model.shape} lignes")
print(f"  - Segments uniques: {df_model['segment_id'].nunique()}")
print(f"  - Lignes uniques: {df_model['line_nom'].nunique()}")
print(f"  - Modes de transport: {sorted(df_model['mode'].unique())}")

# Vérification des données
print(f"\n🔍 VÉRIFICATION DATA:")
non_generic_lines = df_model[df_model['line_nom'] != 'Ligne Inconnue']['line_nom'].unique()
print(f"  ✓ Lignes (non défaut): {non_generic_lines[:5] if len(non_generic_lines) > 0 else 'Toutes défaut'}")

non_generic_stops_from = df_model[~df_model['stop_from_nom'].str.contains('Arrêt_')]['stop_from_nom'].unique()
print(f"  ✓ Arrêts DÉPART (non défaut): {non_generic_stops_from[:5] if len(non_generic_stops_from) > 0 else 'Tous défaut'}")

non_generic_stops_to = df_model[~df_model['stop_to_nom'].str.contains('Arrêt_')]['stop_to_nom'].unique()
print(f"  ✓ Arrêts ARRIVÉE (non défaut): {non_generic_stops_to[:5] if len(non_generic_stops_to) > 0 else 'Tous défaut'}")

print(f"\n📊 SAMPLES:")
print(df_model[['segment_id', 'line_nom', 'stop_from_nom', 'stop_to_nom', 'mode']].head(3))

# ============ CALCUL DE LA QUALITÉ DES TRAJETS ============
print("\n📊 CALCUL DE LA QUALITÉ DES TRAJETS...\n")

scaler_features = StandardScaler()

# Initialiser avec des colonnes partielles
quality_components = {}

# Vitesse (positive - plus rapide c'est mieux)
if 'vitesse' in numeric_cols:
    df_model['vitesse_norm'] = scaler_features.fit_transform(df_model[['vitesse']])
    quality_components['vitesse'] = 0.5
    print("✓ Vitesse normalisée (poids +0.5)")
elif 'vitesse_moyenne' in numeric_cols:
    df_model['vitesse_norm'] = scaler_features.fit_transform(df_model[['vitesse_moyenne']])
    quality_components['vitesse'] = 0.5
    print("✓ Vitesse moyenne normalisée (poids +0.5)")
else:
    df_model['vitesse_norm'] = 0

# Temps de trajet (négatif - moins c'est mieux)
if 'temps_trajet_min' in numeric_cols:
    df_model['temps_norm'] = -scaler_features.fit_transform(df_model[['temps_trajet_min']])
    quality_components['temps'] = 0.3
    print("✓ Temps de trajet normalisé (poids -0.3)")
elif 'temps_trajet' in numeric_cols:
    df_model['temps_norm'] = -scaler_features.fit_transform(df_model[['temps_trajet']])
    quality_components['temps'] = 0.3
    print("✓ Temps de trajet normalisé (poids -0.3)")
else:
    df_model['temps_norm'] = 0

# Congestion (négatif - moins c'est mieux)
if 'congestion_index' in numeric_cols:
    df_model['congestion_norm'] = -scaler_features.fit_transform(df_model[['congestion_index']])
    quality_components['congestion'] = 0.2
    print("✓ Congestion normalisée (poids -0.2)")
else:
    df_model['congestion_norm'] = 0

# Créer le quality_score
df_model['quality_score'] = (
    0.5 * df_model['vitesse_norm'] +
    0.3 * df_model['temps_norm'] +
    0.2 * df_model['congestion_norm']
)

# Normaliser entre -1 et 1
min_score = df_model['quality_score'].min()
max_score = df_model['quality_score'].max()
if max_score - min_score != 0:
    df_model['quality_score'] = 2 * (df_model['quality_score'] - min_score) / (max_score - min_score) - 1

print(f"\n✓ Quality Score calculé")
print(f"  - Min: {df_model['quality_score'].min():.3f}")
print(f"  - Max: {df_model['quality_score'].max():.3f}")
print(f"  - Moyenne: {df_model['quality_score'].mean():.3f}")

# ============================
# B — MODEL UNDERSTANDING (obligatoire)
# ============================
print("\n" + "=" * 80)
print("🧠 COMPRÉHENSION DU MODÈLE (B)")
print("=" * 80 + "\n")
print(
    "Ici, on construit d'abord un score de qualité (quality_score) basé sur vitesse/temps/congestion. "
    "Ensuite, on traite deux objectifs:\n"
    "- Recommandation (supervisée): un modèle de régression prédit/approxime le quality_score pour généraliser.\n"
    "- Clustering (E): on regroupe les segments de trajets par similarité pour profiler des 'types de trajets'.\n"
    "Limites: score construit = proxy métier (pondérations à valider)." 
)

# ============================
# E — CLUSTERING (≥2 modèles + comparaison)
# ============================
print("\n" + "=" * 80)
print("🧩 CLUSTERING (E) — KMEANS vs DBSCAN")
print("=" * 80 + "\n")

# Construire un dataset au niveau 'segment' (profil moyen)
seg_cols = ['segment_id'] + [c for c in numeric_cols if c in df_model.columns] + ['quality_score']
seg_df = df_model[seg_cols].groupby('segment_id').mean(numeric_only=True)

if len(seg_df) >= 10 and seg_df.shape[1] >= 2:
    X_seg = seg_df.values
    scaler_seg = StandardScaler()
    X_seg_scaled = scaler_seg.fit_transform(X_seg)

    # --- KMeans: Elbow + Silhouette
    k_range = list(range(2, min((6 if FAST_MODE else 10), len(seg_df)) + 1))
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = km.fit_predict(X_seg_scaled)
        inertias.append(km.inertia_)
        try:
            silhouettes.append(silhouette_score(X_seg_scaled, labels))
        except Exception:
            silhouettes.append(np.nan)

    best_k = k_range[int(np.nanargmax(silhouettes))] if len(silhouettes) > 0 else 3
    kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=42)
    kmeans_labels = kmeans.fit_predict(X_seg_scaled)
    kmeans_sil = silhouette_score(X_seg_scaled, kmeans_labels)
    kmeans_db = davies_bouldin_score(X_seg_scaled, kmeans_labels)
    print(f"✓ KMeans choisi: k={best_k} | Silhouette={kmeans_sil:.3f} | Davies-Bouldin={kmeans_db:.3f}")

    # --- DBSCAN: petit sweep eps, choisir meilleur silhouette si possible
    best_dbscan = None
    best_dbscan_labels = None
    best_db_sil = -1.0
    for eps in [0.4, 0.6, 0.8, 1.0]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X_seg_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        try:
            sil = silhouette_score(X_seg_scaled, labels)
        except Exception:
            continue
        if sil > best_db_sil:
            best_db_sil = sil
            best_dbscan = db
            best_dbscan_labels = labels

    if best_dbscan is not None and best_dbscan_labels is not None:
        dbscan_sil = best_db_sil
        dbscan_db = davies_bouldin_score(X_seg_scaled, best_dbscan_labels)
        n_clusters = len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)
        print(f"✓ DBSCAN: clusters={n_clusters} (+ bruit=-1) | Silhouette={dbscan_sil:.3f} | Davies-Bouldin={dbscan_db:.3f}")
    else:
        dbscan_sil = np.nan
        dbscan_db = np.nan
        best_dbscan_labels = np.full(len(seg_df), -1)
        print("⚠️ DBSCAN n'a pas trouvé de clustering stable (trop peu de densité).")

    # Comparaison
    cmp_df = pd.DataFrame(
        [
            {'model': 'KMeans', 'silhouette': kmeans_sil, 'davies_bouldin': kmeans_db},
            {'model': 'DBSCAN', 'silhouette': dbscan_sil, 'davies_bouldin': dbscan_db},
        ]
    )
    print("\n📌 Comparaison Clustering:")
    print(cmp_df.to_string(index=False))

    # Visualisations: Elbow + Silhouette
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(k_range, inertias, marker='o')
    ax[0].set_title('Elbow (Inertia) — KMeans')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('Inertia')
    ax[1].plot(k_range, silhouettes, marker='o')
    ax[1].set_title('Silhouette — KMeans')
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('Silhouette')
    plt.tight_layout()
    plt.savefig('Objectif2_Clustering_Elbow_Silhouette.png', dpi=200, bbox_inches='tight')
    plt.show()

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_seg_scaled)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='tab10', s=18, alpha=0.8)
    ax[0].set_title(f'PCA 2D — KMeans (k={best_k})')
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=best_dbscan_labels, cmap='tab10', s=18, alpha=0.8)
    ax[1].set_title('PCA 2D — DBSCAN (bruit=-1)')
    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig('Objectif2_Clustering_PCA2D.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Profiling: qualité moyenne par cluster KMeans
    seg_profile = seg_df.copy()
    seg_profile['cluster_kmeans'] = kmeans_labels
    profile_stats = seg_profile.groupby('cluster_kmeans')['quality_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print("\n🔎 Profiling clusters (KMeans) — quality_score:")
    print(profile_stats.to_string())

    # Table mapping segment -> clusters
    segment_cluster_map = pd.DataFrame(
        {
            'segment_id': seg_df.index.astype(int),
            'cluster_kmeans': kmeans_labels.astype(int),
            'cluster_dbscan': best_dbscan_labels.astype(int),
        }
    )
else:
    print("⚠️ Données insuffisantes pour clustering (segments/features).")
    segment_cluster_map = pd.DataFrame({'segment_id': df_model['segment_id'].unique()})
    segment_cluster_map['cluster_kmeans'] = -1
    segment_cluster_map['cluster_dbscan'] = -1

# Préparer les features et target
X = df_model[numeric_cols].copy()
y = df_model['quality_score'].values

print(f"\n✓ Features: {X.shape[0]} échantillons, {X.shape[1]} variables")
print(f"✓ Target: {len(y)} valeurs")

# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============ 🎯 ENTRAÎNER LE MODÈLE ============
print("\n" + "="*80)
print("🎯 ENTRAÎNEMENT DU MODÈLE GRADIENT BOOSTING")
print("="*80 + "\n")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

gb_model = GradientBoostingRegressor(random_state=42)
param_dist = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.03, 0.06, 0.1],
    'max_depth': [2, 3, 4],
    'subsample': [0.7, 0.9, 1.0],
}

search = RandomizedSearchCV(
    gb_model,
    param_distributions=param_dist,
    n_iter=(6 if FAST_MODE else 12),
    scoring='r2',
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)
gb_model = search.best_estimator_
print(f"✓ Modèle entraîné (tuning) sur {len(X_train)} échantillons")

# Conserver le scaler associé au modèle principal (car plus bas, d'autres sections refittent `scaler`).
scaler_gb = scaler

y_pred = gb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Performance du modèle (test set):")
print(f"  - R² Score: {r2:.4f}")
print(f"  - MAE: {mae:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': numeric_cols,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🔍 Feature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  - {row['Feature']}: {row['Importance']:.4f}")

# ============ 📅 GÉNÉRER LES RECOMMANDATIONS (36 MOIS) ============
print("\n" + "="*80)
print("📅 GÉNÉRATION DES RECOMMANDATIONS (36 MOIS)")
print("="*80 + "\n")

from datetime import timedelta

if 'date' in df_model.columns:
    max_date = pd.to_datetime(df_model['date'], errors='coerce').max()
else:
    max_date = pd.NaT
if pd.isna(max_date):
    max_date = pd.Timestamp('2026-01-01')

segments_uniques = sorted(df_model['segment_id'].dropna().unique())

print(f"✓ Segments uniques: {len(segments_uniques)}")
print(f"✓ Date maximale: {max_date}")

# Générer 36 mois futurs
future_dates = pd.date_range(start=max_date + timedelta(days=1), periods=36, freq='MS')

# Pré-calculer un profil par segment (moyennes des features + métadonnées)
meta_cols = ['stop_from_nom', 'stop_to_nom', 'ville_depart', 'ville_arrivee', 'line_nom', 'mode']
meta_cols = [c for c in meta_cols if c in df_model.columns]

seg_num = df_model.groupby('segment_id', as_index=False)[numeric_cols].mean()
seg_meta = df_model.groupby('segment_id', as_index=False)[meta_cols].first() if meta_cols else None

segment_profile = seg_num.merge(seg_meta, on='segment_id', how='left') if seg_meta is not None else seg_num

# Valeurs par défaut si manquantes
defaults = {
    'stop_from_nom': 'Arrêt Départ',
    'stop_to_nom': 'Arrêt Arrivée',
    'ville_depart': 'Ville Départ',
    'ville_arrivee': 'Ville Arrivée',
    'line_nom': 'Ligne Inconnue',
    'mode': 'Transport',
}
for c, v in defaults.items():
    if c not in segment_profile.columns:
        segment_profile[c] = v
    else:
        segment_profile[c] = segment_profile[c].fillna(v).replace(['', None, 'NaN', 'nan'], v)

segment_profile['trajet'] = (
    segment_profile['stop_from_nom'].astype(str)
    + " (" + segment_profile['ville_depart'].astype(str) + ") → "
    + segment_profile['stop_to_nom'].astype(str)
    + " (" + segment_profile['ville_arrivee'].astype(str) + ")"
)

predictions_frames = []

# Générer 36 mois futurs (vectorisé par mois pour éviter une double boucle lente)
for date in future_dates:
    year = date.year
    month = date.month

    seasonal_factor = 1.0 + 0.25 * np.sin(2 * np.pi * month / 12)
    noise = np.random.uniform(0.85, 1.15, size=len(segment_profile))

    X_month = segment_profile[numeric_cols].to_numpy(dtype=float)
    X_month = np.maximum(0.1, X_month * (seasonal_factor * noise)[:, None])

    X_scaled_month = scaler.transform(X_month)
    quality_preds = gb_model.predict(X_scaled_month)

    rec = np.where(
        quality_preds > 0.6,
        "🌟 Excellent",
        np.where(
            quality_preds > 0.3,
            "✅ Très Bon",
            np.where(quality_preds > 0, "👍 Bon", np.where(quality_preds > -0.3, "⚠️ Moyen", "❌ À Éviter")),
        ),
    )

    df_month = pd.DataFrame({
        'segment_id': segment_profile['segment_id'].astype(int),
        'trajet': segment_profile['trajet'].astype(str),
        'stop_depart': segment_profile['stop_from_nom'].astype(str),
        'ville_depart': segment_profile['ville_depart'].astype(str),
        'stop_arrivee': segment_profile['stop_to_nom'].astype(str),
        'ville_arrivee': segment_profile['ville_arrivee'].astype(str),
        'ligne': segment_profile['line_nom'].astype(str),
        'mode': segment_profile['mode'].astype(str),
        'année': year,
        'mois': month,
        'date': date,
        'quality_score': quality_preds.astype(float),
        'recommendation': rec.astype(str),
    })

    predictions_frames.append(df_month)

predictions_df = pd.concat(predictions_frames, ignore_index=True)

# Enrichir avec clusters (E)
try:
    predictions_df = predictions_df.merge(segment_cluster_map, on='segment_id', how='left')
except Exception:
    predictions_df['cluster_kmeans'] = -1
    predictions_df['cluster_dbscan'] = -1

print(f"\n✓ {len(predictions_df)} recommandations générées")
print(f"✓ Période: {predictions_df['date'].min().date()} à {predictions_df['date'].max().date()}")
print(f"✓ Trajets par mois: {predictions_df.groupby('date')['segment_id'].nunique().iloc[0]}")

print(f"\n📋 Top 10 MEILLEURS TRAJETS:")
top_10 = predictions_df.nlargest(10, 'quality_score')[['trajet', 'ligne', 'mode', 'quality_score', 'recommendation']]
print(top_10.to_string(index=False))

# ============ 📊 STATISTIQUES DÉTAILLÉES ============
print("\n" + "="*80)
print("📊 STATISTIQUES DÉTAILLÉES DES RECOMMANDATIONS")
print("="*80 + "\n")

print(f"Quality Score Global:")
print(f"  - Minimum: {predictions_df['quality_score'].min():.4f}")
print(f"  - Maximum: {predictions_df['quality_score'].max():.4f}")
print(f"  - Moyenne: {predictions_df['quality_score'].mean():.4f}")
print(f"  - Médiane: {predictions_df['quality_score'].median():.4f}")

print(f"\nRépartition des recommandations:")
rec_dist = predictions_df['recommendation'].value_counts()
for rec, count in rec_dist.items():
    percentage = (count / len(predictions_df)) * 100
    print(f"  - {rec}: {count} ({percentage:.1f}%)")

print(f"\nAnalyse par mode de transport:")
for mode in sorted(predictions_df['mode'].unique()):
    mode_data = predictions_df[predictions_df['mode'] == mode]
    print(f"  - {mode}: {len(mode_data)} trajets, Score moyen: {mode_data['quality_score'].mean():.3f}")

print(f"\nTop 5 Meilleures Villes de Départ (qualité moyenne):")
top_villes = predictions_df.groupby('ville_depart')['quality_score'].mean().nlargest(5)
for ville, score in top_villes.items():
    print(f"  - {ville}: {score:.3f}")

print(f"\nRépartition des trajets par Ligne:")
ligne_stats = predictions_df.groupby('ligne').agg({
    'quality_score': ['mean', 'count']
}).sort_values(('quality_score', 'mean'), ascending=False)
for idx, (ligne, row) in enumerate(ligne_stats.head(10).iterrows(), 1):
    print(f"  {idx}. {ligne}: {row[('quality_score', 'count')]} trajets, Score: {row[('quality_score', 'mean')]:.3f}")

# ============ � VISUALISATIONS SIGNIFICATIVES ET FIABLES ============
print("\n📊 GÉNÉRATION DES VISUALISATIONS...\n")

try:
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, top=0.96, bottom=0.08, left=0.08, right=0.95)
    
    fig.suptitle('🗺️ OBJECTIF 2 - ANALYSE PROFESSIONNELLE DES MEILLEURS TRAJETS', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    fig.text(0.5, 0.945, f'Période: {predictions_df["date"].min().date()} - {predictions_df["date"].max().date()} | Récords: {len(predictions_df):,} | Trajets uniques: {predictions_df["trajet"].nunique()}',
             ha='center', fontsize=11, style='italic', color='#555555')
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    palette_quality = {'Excellent': '#2ecc71', 'Très Bon': '#3498db', 
                       'Bon': '#f39c12', 'Moyen': '#e67e22', 'À Éviter': '#e74c3c'}
    
    # ===== GRAPHIQUE 1: Top 15 Meilleures TRAJECTOIRES (Quality moyenne) =====
    ax1 = fig.add_subplot(gs[0, 0])
    if 'trajet' in predictions_df.columns and 'quality_score' in predictions_df.columns:
        trajet_stats = predictions_df.groupby('trajet')['quality_score'].mean().nlargest(15).sort_values()
        
        colors1 = ['#2ecc71' if x > 0.5 else '#3498db' if x > 0.2 else '#f39c12' for x in trajet_stats.values]
        ax1.barh(range(len(trajet_stats)), trajet_stats.values, color=colors1, edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        
        ax1.set_yticks(range(len(trajet_stats)))
        ax1.set_yticklabels([str(t)[:35] for t in trajet_stats.index], fontsize=8.5, color='#2c3e50')
        ax1.set_xlabel('Quality Score Moyen', fontsize=10, fontweight='bold')
        ax1.set_title('🗺️ Top 15 Trajectoires\n(Meilleure Qualité)', fontsize=12, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)
        
        for i, (idx, score) in enumerate(trajet_stats.items()):
            ax1.text(score + 0.01, i, f'{score:.2f}', va='center', fontsize=8, fontweight='bold', color='#2c3e50')
    
    print("  ✓ Graphique 1: Top Trajectoires OK")
    
    # ===== GRAPHIQUE 2: Top ARRÊTS (ARRIVÉE) - Performance =====
    ax2 = fig.add_subplot(gs[0, 1])
    if 'stop_arrivee' in predictions_df.columns and 'quality_score' in predictions_df.columns:
        stop_arrivee_stats = predictions_df.groupby('stop_arrivee')['quality_score'].mean().nlargest(12).sort_values()
        
        colors2 = ['#2ecc71' if x > 0.3 else '#3498db' if x > 0 else '#f39c12' for x in stop_arrivee_stats.values]
        bars = ax2.barh(range(len(stop_arrivee_stats)), stop_arrivee_stats.values, 
                       color=colors2, edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        
        ax2.set_yticks(range(len(stop_arrivee_stats)))
        ax2.set_yticklabels([str(t)[:25] for t in stop_arrivee_stats.index], fontsize=9, color='#2c3e50')
        ax2.set_xlabel('Quality Score Moyen', fontsize=10, fontweight='bold')
        ax2.set_title('🚏 Top Arrêts (ARRIVÉE)\n(Destinations Prioritaires)', fontsize=12, fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)
        
        for i, (idx, score) in enumerate(stop_arrivee_stats.items()):
            ax2.text(score + 0.01, i, f'{score:.2f}', va='center', fontsize=8, fontweight='bold', color='#2c3e50')
    
    print("  ✓ Graphique 2: Arrêts (Arrivée) OK")
    
    # ===== GRAPHIQUE 3: Top ARRÊTS (DÉPART) - Performance =====
    ax3 = fig.add_subplot(gs[0, 2])
    if 'stop_depart' in predictions_df.columns and 'quality_score' in predictions_df.columns:
        stop_depart_stats = predictions_df.groupby('stop_depart')['quality_score'].mean().nlargest(12).sort_values()
        
        colors3 = ['#2ecc71' if x > 0.3 else '#3498db' if x > 0 else '#f39c12' for x in stop_depart_stats.values]
        bars = ax3.barh(range(len(stop_depart_stats)), stop_depart_stats.values, 
                       color=colors3, edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        
        ax3.set_yticks(range(len(stop_depart_stats)))
        ax3.set_yticklabels([str(t)[:25] for t in stop_depart_stats.index], fontsize=9, color='#2c3e50')
        ax3.set_xlabel('Quality Score Moyen', fontsize=10, fontweight='bold')
        ax3.set_title('🚏 Top Arrêts (DÉPART)\n(Origines Prioritaires)', fontsize=12, fontweight='bold', pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)
        
        for i, (idx, score) in enumerate(stop_depart_stats.items()):
            ax3.text(score + 0.01, i, f'{score:.2f}', va='center', fontsize=8, fontweight='bold', color='#2c3e50')
    
    print("  ✓ Graphique 3: Arrêts (Départ) OK")
    
    # ===== GRAPHIQUE 4: TRAJECTOIRES par DATE (Top 5) - Tendances temporelles =====
    ax4 = fig.add_subplot(gs[1, :2])
    if 'date' in predictions_df.columns and 'trajet' in predictions_df.columns and 'quality_score' in predictions_df.columns:
        # Sélectionner les 5 meilleures trajectoires
        top_5_trajets = predictions_df.groupby('trajet')['quality_score'].mean().nlargest(5).index.tolist()
        
        colors_trajets = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
        
        for idx, trajet in enumerate(top_5_trajets):
            trajet_data = predictions_df[predictions_df['trajet'] == trajet].sort_values('date')
            monthly_avg = trajet_data.groupby(trajet_data['date'].dt.to_period('M'))['quality_score'].mean()
            
            ax4.plot(range(len(monthly_avg)), monthly_avg.values, 
                    marker='o', linewidth=2.5, markersize=6, 
                    label=f"{trajet[:35]}...", color=colors_trajets[idx], alpha=0.8)
        
        ax4.set_ylabel('Quality Score Moyen', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Période (36 mois)', fontsize=11, fontweight='bold')
        ax4.set_title('📈 Performance des 5 Meilleures Trajectoires\n(Évolution par DATE)', fontsize=12, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax4.legend(loc='best', fontsize=8, framealpha=0.95, edgecolor='#2c3e50')
    
    print("  ✓ Graphique 4: Trajectoires par Date OK")
    
    # ===== GRAPHIQUE 5: ARRÊTS (Départ-Arrivée) PAIRES TOP =====
    ax5 = fig.add_subplot(gs[1, 2])
    if 'stop_depart' in predictions_df.columns and 'stop_arrivee' in predictions_df.columns and 'quality_score' in predictions_df.columns:
        # Créer des paires de trajets
        paires = predictions_df.groupby(['stop_depart', 'stop_arrivee'])['quality_score'].mean().nlargest(10).sort_values()
        
        paires_labels = [f"{str(pair[0])[:8]}\n→\n{str(pair[1])[:8]}" for pair in paires.index]
        
        colors5 = ['#2ecc71' if x > 0.3 else '#3498db' if x > 0 else '#f39c12' for x in paires.values]
        bars = ax5.bar(range(len(paires)), paires.values, 
                      color=colors5, edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        
        ax5.set_xticks(range(len(paires)))
        ax5.set_xticklabels(paires_labels, fontsize=7, rotation=0, ha='center', color='#2c3e50')
        ax5.set_ylabel('Quality Score Moyen', fontsize=10, fontweight='bold')
        ax5.set_title('🔗 Top 10 Paires\n(Départ → Arrivée)', fontsize=12, fontweight='bold', pad=15)
        ax5.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
        ax5.set_ylim(bottom=0)
        
        for bar, score in zip(bars, paires.values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{score:.2f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2c3e50')
    
    print("  ✓ Graphique 5: Paires Arrêts OK")
    
    # ===== GRAPHIQUE 6: ARRÊTS - Nombre de trajets passant par chaque arrêt =====
    ax6 = fig.add_subplot(gs[2, 0])
    if 'stop_depart' in predictions_df.columns and 'stop_arrivee' in predictions_df.columns:
        # Combine départ et arrivée
        all_stops_depart = predictions_df['stop_depart'].value_counts().head(8)
        all_stops_arrivee = predictions_df['stop_arrivee'].value_counts().head(8)
        
        x_pos = np.arange(len(all_stops_depart))
        width = 0.35
        
        bars1 = ax6.bar(x_pos - width/2, all_stops_depart.values, width, 
                       label='DÉPART', color='#2ecc71', edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        bars2 = ax6.bar(x_pos + width/2, all_stops_arrivee.values[:len(all_stops_depart)], width, 
                       label='ARRIVÉE', color='#3498db', edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([str(s)[:12] for s in all_stops_depart.index], rotation=45, ha='right', fontsize=9, color='#2c3e50')
        ax6.set_ylabel('Nombre de trajets', fontsize=10, fontweight='bold')
        ax6.set_title('📊 Fréquence Arrêts\n(DÉPART vs ARRIVÉE)', fontsize=12, fontweight='bold', pad=15)
        ax6.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax6.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
    
    print("  ✓ Graphique 6: Fréquence Arrêts OK")
    
    # ===== GRAPHIQUE 7: Tableau Synthèse TRAJECTOIRES - ARRÊTS - DATES =====
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    stats_data = []
    
    # Section TRAJECTOIRES
    if 'trajet' in predictions_df.columns:
        stats_data.extend([
            ['📍 TRAJECTOIRES', 'QUANTITÉ', 'DÉTAILS'],
            ['Total Trajectoires', f'{predictions_df["trajet"].nunique()}', 'Routes analysées'],
            ['Top Score Trajet', f'{predictions_df.nlargest(1, "quality_score")["trajet"].iloc[0][:40]}...', '⭐ Meilleure'],
            ['Score Moyen Trajets', f'{predictions_df.groupby("trajet")["quality_score"].mean().mean():.3f}', '📊 Moyenne'],
        ])
    
    # Section ARRÊTS
    if 'stop_depart' in predictions_df.columns and 'stop_arrivee' in predictions_df.columns:
        stats_data.extend([
            ['', '', ''],
            ['🚏 ARRÊTS', 'QUANTITÉ', 'DÉTAILS'],
            ['Arrêts DÉPART', f'{predictions_df["stop_depart"].nunique()}', 'Points de départ'],
            ['Arrêts ARRIVÉE', f'{predictions_df["stop_arrivee"].nunique()}', 'Points d\'arrivée'],
            ['Total Arrêts Uniques', f'{predictions_df["stop_depart"].nunique() + predictions_df["stop_arrivee"].nunique()}', 'Points distincts'],
        ])
    
    # Section DATES
    if 'date' in predictions_df.columns:
        date_min = predictions_df['date'].min().date()
        date_max = predictions_df['date'].max().date()
        stats_data.extend([
            ['', '', ''],
            ['📅 TEMPORAL', 'VALEUR', 'DURÉE'],
            ['Date Minimum', str(date_min), 'Début période'],
            ['Date Maximum', str(date_max), 'Fin période'],
            ['Mois Couverts', f'{(date_max.year - date_min.year)*12 + date_max.month - date_min.month} mois', 'Horizon'],
        ])
    
    # Section QUALITY SCORES par dimension
    stats_data.extend([
        ['', '', ''],
        ['📈 QUALITY SCORES', 'STATISTIQUE', 'VALEUR'],
        ['Score Global Moyen', f'{predictions_df["quality_score"].mean():.3f}', 'Toutes données'],
        ['Score Global Min', f'{predictions_df["quality_score"].min():.3f}', 'Pire cas'],
        ['Score Global Max', f'{predictions_df["quality_score"].max():.3f}', 'Meilleur cas'],
    ])
    
    table = ax7.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.25, 0.5], bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.8)
    
    for i, row in enumerate(stats_data):
        for j in range(3):
            cell = table[(i, j)]
            if i in [0, 4, 11, 17]:  # Section headers
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(weight='bold', color='white', fontsize=9)
            elif i in [1, 5, 12, 18]:  # Sub-headers
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white', fontsize=8.5)
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
                cell.set_text_props(color='#2c3e50', fontsize=8.5)
            
            cell.set_edgecolor('#95a5a6')
            cell.set_linewidth(0.8)
    
    ax7.set_title('📋 SYNTHÈSE: TRAJECTOIRES - ARRÊTS - DATES', 
                 fontsize=12, fontweight='bold', pad=20, loc='left')
    
    print("  ✓ Graphique 7: Synthèse Trajectoires-Arrêts-Dates OK")
    
    plt.tight_layout()
    
    try:
        plt.savefig('Objectif2_Visualisations_Professionnelles.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n✅ VISUALISATIONS PROFESSIONNELLES CRÉÉES")
        print("   Fichier: Objectif2_Visualisations_Professionnelles.png")
    except Exception as e:
        print(f"\n⚠️ Erreur sauvegarde PNG: {e}")
    
    plt.show()
    
except Exception as e:
    print(f"⚠️ Erreur visualisations: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)

# ============================
# ✅ COMPARAISONS (3 MODÈLES) — CLUSTERING + RÉGRESSION
# ============================
print("📊 COMPARAISONS (3 MODÈLES) — CLUSTERING + RÉGRESSION")
print("=" * 80 + "\n")

# --- E: Clustering (KMeans vs DBSCAN vs Agglomerative)
try:
    clustering_rows = []

    if 'seg_df' in locals() and 'X_seg_scaled' in locals() and len(seg_df) >= 10:
        # KMeans
        if 'kmeans_labels' in locals():
            clustering_rows.append(
                {
                    'model': 'KMeans',
                    'silhouette': float(silhouette_score(X_seg_scaled, kmeans_labels)),
                    'davies_bouldin': float(davies_bouldin_score(X_seg_scaled, kmeans_labels)),
                }
            )

        # DBSCAN
        if 'best_dbscan_labels' in locals():
            labels = best_dbscan_labels
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                clustering_rows.append(
                    {
                        'model': 'DBSCAN',
                        'silhouette': float(silhouette_score(X_seg_scaled, labels)),
                        'davies_bouldin': float(davies_bouldin_score(X_seg_scaled, labels)),
                    }
                )

        # Agglomerative (même k que KMeans si dispo)
        k_for_aggl = int(best_k) if 'best_k' in locals() else 3
        k_for_aggl = max(2, min(k_for_aggl, len(seg_df) - 1))
        agg = AgglomerativeClustering(n_clusters=k_for_aggl)
        agg_labels = agg.fit_predict(X_seg_scaled)
        clustering_rows.append(
            {
                'model': f'Agglomerative(k={k_for_aggl})',
                'silhouette': float(silhouette_score(X_seg_scaled, agg_labels)),
                'davies_bouldin': float(davies_bouldin_score(X_seg_scaled, agg_labels)),
            }
        )

        clustering_cmp = pd.DataFrame(clustering_rows)
        print("✅ Clustering — comparaison:")
        print(clustering_cmp.sort_values('silhouette', ascending=False).to_string(index=False))

        # Plot comparaison clustering
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        fig.patch.set_facecolor('white')
        order = clustering_cmp.sort_values('silhouette', ascending=False)['model'].tolist()

        ax = axes[0]
        data = clustering_cmp.set_index('model').loc[order]['silhouette']
        bars = ax.bar(order, data.values, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1.2)
        for b, v in zip(bars, data.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        ax.set_title('Silhouette (↑)', fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

        ax = axes[1]
        data = clustering_cmp.set_index('model').loc[order]['davies_bouldin']
        bars = ax.bar(order, data.values, color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=1.2)
        for b, v in zip(bars, data.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        ax.set_title('Davies–Bouldin (↓)', fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

        plt.tight_layout()
        plt.savefig('Objectif2_Model_Comparison_Clustering.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    else:
        print("⚠️ Clustering comparison ignoré: données segment insuffisantes.")
except Exception as e:
    print(f"⚠️ Erreur comparaison clustering: {e}")


# --- Régression (3 modèles) — RandomForest vs SVR vs KNN (avec CV + tuning)
try:
    print("\n" + "-" * 80)
    print("🎯 RÉGRESSION (3 MODÈLES) — VALIDATION CROISÉE + TUNING")
    print("-" * 80)

    X_reg = df_model[numeric_cols].copy()
    y_reg = df_model['quality_score'].values
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # 1) RandomForest
    rf = RandomForestRegressor(random_state=42)
    rf_search = RandomizedSearchCV(
        rf,
        param_distributions={
            'n_estimators': ([200, 400] if FAST_MODE else [200, 500, 800]),
            'max_depth': ([None, 10] if FAST_MODE else [None, 6, 10, 18]),
            'min_samples_split': ([2, 5] if FAST_MODE else [2, 5, 10]),
            'min_samples_leaf': ([1, 2] if FAST_MODE else [1, 2, 4]),
        },
        n_iter=(6 if FAST_MODE else 12),
        cv=(3 if FAST_MODE else 5),
        scoring='r2',
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_

    # 2) SVR
    svr = SVR()
    svr_search = GridSearchCV(
        svr,
        param_grid=(
            {'C': [1.0, 3.0], 'epsilon': [0.1], 'gamma': ['scale']}
            if FAST_MODE
            else {'C': [0.5, 1.0, 3.0], 'epsilon': [0.05, 0.1, 0.2], 'gamma': ['scale', 'auto']}
        ),
        cv=(3 if FAST_MODE else 5),
        scoring='r2',
        n_jobs=-1,
    )
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svr_search.fit(X_train_s, y_train)
    svr_best = svr_search.best_estimator_

    # 3) KNN
    knn = KNeighborsRegressor()
    knn_search = GridSearchCV(
        knn,
        param_grid=(
            {'n_neighbors': [5, 9], 'weights': ['distance']}
            if FAST_MODE
            else {'n_neighbors': [3, 5, 9, 15], 'weights': ['uniform', 'distance']}
        ),
        cv=(3 if FAST_MODE else 5),
        scoring='r2',
        n_jobs=-1,
    )
    knn_search.fit(X_train_s, y_train)
    knn_best = knn_search.best_estimator_

    # Évaluer
    def _eval_reg(y_true, y_pred):
        return {
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'R2': float(r2_score(y_true, y_pred)),
        }

    rows = []
    rows.append({'model': 'RandomForest', **_eval_reg(y_test, rf_best.predict(X_test))})
    rows.append({'model': 'SVR', **_eval_reg(y_test, svr_best.predict(X_test_s))})
    rows.append({'model': 'KNN', **_eval_reg(y_test, knn_best.predict(X_test_s))})
    reg_cmp = pd.DataFrame(rows).sort_values('R2', ascending=False)
    print("\n✅ Régression — comparaison (test set):")
    print(reg_cmp.to_string(index=False))

    # Plot comparaison
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    bars = ax.bar(reg_cmp['model'], reg_cmp['R2'], color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=1.2)
    for b, v in zip(bars, reg_cmp['R2'].values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
    ax.set_title('R² (↑)', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1]
    bars = ax.bar(reg_cmp['model'], reg_cmp['MAE'], color='#d62728', alpha=0.85, edgecolor='black', linewidth=1.2)
    for b, v in zip(bars, reg_cmp['MAE'].values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
    ax.set_title('MAE (↓)', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig('Objectif2_Model_Comparison_Regression.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

except Exception as e:
    print(f"⚠️ Erreur comparaison régression: {e}")

print("\n✅ OBJECTIF 2 TERMINÉ!")
print("=" * 80)


def _mlops_export_objective2() -> None:
    if os.getenv("MLOPS_EXPORT", "0").strip().lower() not in ("1", "true", "yes"):
        return

    from mlops.registry import register_model

    if "gb_model" not in globals() or "numeric_cols" not in globals():
        print("⚠️ MLOps export ignoré: variables manquantes (gb_model/numeric_cols).")
        return

    bundle = {
        "scaler": globals().get("scaler_gb") or globals().get("scaler"),
        "regressor": globals().get("gb_model"),
    }

    meta = {
        "numeric_cols": list(globals().get("numeric_cols") or []),
        "r2": float(globals().get("r2", float("nan"))),
        "mae": float(globals().get("mae", float("nan"))),
    }

    try:
        entry = register_model("objective2", bundle, meta)
        print({"status": "success", "objective": "objective2", "version": entry.version})
    except Exception as e:
        print({"status": "error", "message": f"MLOps export failed: {e}"})


_mlops_export_objective2()
