"""OBJECTIF 4: PRÉDIRE LES ÉMISSIONS CO2 ET LA CONSOMMATION D'ÉNERGIE.

Ce script compare plusieurs modèles de régression (avec tuning/CV), puis génère
des prédictions sur 36 mois et des visualisations professionnelles.
"""

# ============ DÉPENDANCES / COMPAT COLAB vs PYTHON ============
from __future__ import annotations

import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from difflib import get_close_matches

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

from mlops.transformers import QuantileClipper

warnings.filterwarnings("ignore")

# Windows terminals may default to cp1252 which can crash on non-ASCII output.
# Force UTF-8 where supported.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================
# CONFIG BASE DE DONNÉES (PostgreSQL)
# ============================
# Mode par défaut: si les variables d'environnement DB_* sont définies, on lit depuis PostgreSQL.
# Sinon, fallback: lecture CSV depuis le dossier courant (sans upload Colab).
DB_ENV_VARS = [
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
]

# Connexion PostgreSQL (valeurs par défaut demandées)
# NB: vous pouvez toujours surcharger via variables d'environnement DB_*.
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "urbain_dw")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "admin")
os.environ.setdefault("DB_SCHEMA", "public")

# Accélération (par défaut)
# - FAST_MODE=1 : tuning réduit (plus rapide)
# - ONLY_DB=1 : pas de plots/rapport, on écrit en base puis on s'arrête
FAST_MODE = os.getenv("FAST_MODE", "1").strip().lower() not in ("0", "false", "no")
ONLY_DB = os.getenv("ONLY_DB", "1").strip().lower() not in ("0", "false", "no")

_ENGINE = None


def _qident(name: str) -> str:
    """Quote an SQL identifier safely (schema/table/column)."""
    return '"' + str(name).replace('"', '""') + '"'


def _db_enabled() -> bool:
    return all(os.getenv(v) for v in DB_ENV_VARS)


def _get_engine():
    """Crée un engine SQLAlchemy pour PostgreSQL.

    Dépendances:
      - sqlalchemy
      - psycopg2-binary
    """
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
    # NOTE: Certaines combinaisons pandas/SQLAlchemy 2.x peuvent provoquer
    # des erreurs côté driver quand la requête contient des '%' (ILIKE patterns)
    # à cause d'un "params" vide transmis au driver. On contourne en passant
    # par une connexion DB-API (psycopg2) via raw_connection().
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
    sql = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = %(schema)s
      AND table_type = 'BASE TABLE'
    ORDER BY table_name;
    """.strip()
    df_t = _read_sql(sql.replace("%(schema)s", f"'{schema}'"))
    if 'table_name' not in df_t.columns:
        return []
    return [str(x) for x in df_t['table_name'].dropna().tolist()]


def _resolve_table(
    preferred: str,
    schema: str = "public",
    like_patterns: list[str] | None = None,
) -> str | None:
    """Résout un nom de table réel dans la DB.

    - Essaie d'abord `preferred`.
    - Puis des patterns ILIKE.
    - Puis un fuzzy match (proche) via difflib.
    """
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
            df_m = _read_sql(sql)
            if 'table_name' in df_m.columns and len(df_m) > 0:
                return str(df_m['table_name'].iloc[0])

    # Fuzzy match
    close = get_close_matches(preferred, tables, n=1, cutoff=0.65)
    return close[0] if close else None


def _read_table(table: str, schema: str = "public", limit: int | None = None) -> pd.DataFrame:
    lim = limit if limit is not None else _db_limit_rows()
    if lim is not None:
        return _read_sql(f"SELECT * FROM {_qident(schema)}.{_qident(table)} LIMIT {int(lim)};")
    return _read_sql(f"SELECT * FROM {_qident(schema)}.{_qident(table)};")


def _table_columns(schema: str, table: str) -> set[str]:
    sql = (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema='{schema}' AND table_name='{table}' ORDER BY ordinal_position;"
    )
    d = _read_sql(sql)
    if 'column_name' not in d.columns:
        return set()
    return set(str(x) for x in d['column_name'].dropna().tolist())


def _try_read_joined_fact(
    schema: str,
    fact_table: str,
    dim_emission_table: str | None,
    dim_energie_table: str | None,
    dim_zone_table: str | None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Tente une requête SQL jointe (LEFT JOIN) pour obtenir un df déjà enrichi.

    Retourne (df, sql) si OK, sinon (None, None).
    """
    f_cols = _table_columns(schema, fact_table)

    select_parts = ["f.*"]
    join_parts: list[str] = []

    if dim_emission_table:
        e_cols = _table_columns(schema, dim_emission_table)
        if ('fk_emco2' in f_cols) and ('emission_id' in e_cols):
            if 'mode' in e_cols:
                select_parts.append('e.mode')
            if 'activity_type' in e_cols:
                select_parts.append('e.activity_type')
            join_parts.append(
                f"LEFT JOIN {_qident(schema)}.{_qident(dim_emission_table)} e ON f.fk_emco2 = e.emission_id"
            )

    if dim_energie_table:
        t_cols = _table_columns(schema, dim_energie_table)
        if ('fk_energie' in f_cols) and ('energie_id' in t_cols):
            if 'type_energie' in t_cols:
                select_parts.append('t.type_energie')
            if 'source_energie' in t_cols:
                select_parts.append('t.source_energie')
            join_parts.append(
                f"LEFT JOIN {_qident(schema)}.{_qident(dim_energie_table)} t ON f.fk_energie = t.energie_id"
            )

    if dim_zone_table:
        z_cols = _table_columns(schema, dim_zone_table)
        if ('fk_zone' in f_cols) and ('zone_id' in z_cols):
            if 'zone_name' in z_cols:
                select_parts.append('z.zone_name')
            elif 'zone_nom' in z_cols:
                select_parts.append('z.zone_nom AS zone_name')
            join_parts.append(
                f"LEFT JOIN {_qident(schema)}.{_qident(dim_zone_table)} z ON f.fk_zone = z.zone_id"
            )

    sql = (
        "SELECT\n    "
        + ",\n    ".join(select_parts)
        + f"\nFROM {_qident(schema)}.{_qident(fact_table)} f\n"
        + ("\n".join(join_parts) + "\n" if join_parts else "")
    )

    lim = _db_limit_rows()
    if lim is not None:
        sql += f"LIMIT {int(lim)}\n"

    sql = (sql + ";").strip()

    try:
        return _read_sql(sql), sql
    except Exception:
        return None, None


def _write_predictions_to_postgres(predictions_df: pd.DataFrame) -> None:
    """Crée 2 tables dans PostgreSQL et upsert les prédictions 2027-2029."""
    schema = os.getenv("DB_SCHEMA", "public")
    co2_table = os.getenv("DB_PRED_CO2_TABLE", "predictions_co2_2027_2029")
    energy_table = os.getenv("DB_PRED_ENERGIE_TABLE", "predictions_energie_2027_2029")

    if predictions_df is None or len(predictions_df) == 0:
        print("⚠️ Aucune prédiction à écrire en base")
        return

    dfp = predictions_df.copy()
    dfp["date"] = pd.to_datetime(dfp.get("date"), errors="coerce")
    dfp = dfp[dfp["date"].notna()]
    dfp = dfp[(dfp["date"] >= pd.Timestamp("2027-01-01")) & (dfp["date"] <= pd.Timestamp("2029-12-31"))]
    if len(dfp) == 0:
        print("⚠️ Aucune ligne dans l'intervalle 2027-2029 (rien à écrire)")
        return

    # Normalisation types
    dfp["pred_date"] = dfp["date"].dt.date
    if "année" in dfp.columns:
        dfp["annee"] = pd.to_numeric(dfp["année"], errors="coerce").fillna(dfp["date"].dt.year).astype(int)
    else:
        dfp["annee"] = dfp["date"].dt.year.astype(int)
    if "mois" in dfp.columns:
        dfp["mois_int"] = pd.to_numeric(dfp["mois"], errors="coerce").fillna(dfp["date"].dt.month).astype(int)
    else:
        dfp["mois_int"] = dfp["date"].dt.month.astype(int)

    # Connexion DB via l'engine existant
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _get_engine()

    # DDL + UPSERT
    try:
        from psycopg2.extras import execute_values  # type: ignore
    except Exception as e:
        raise ImportError("psycopg2 est requis pour écrire en base (pip install psycopg2-binary)") from e

    conn = _ENGINE.raw_connection()
    try:
        cur = conn.cursor()

        ddl_co2 = f"""
        CREATE TABLE IF NOT EXISTS {_qident(schema)}.{_qident(co2_table)} (
            pred_date date NOT NULL,
            annee integer NOT NULL,
            mois integer NOT NULL,
            zone_id integer NOT NULL,
            zone_name text NULL,
            mode_transport text NOT NULL,
            co2_kg_predite double precision NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (pred_date, zone_id, mode_transport)
        );
        """.strip()

        ddl_energy = f"""
        CREATE TABLE IF NOT EXISTS {_qident(schema)}.{_qident(energy_table)} (
            pred_date date NOT NULL,
            annee integer NOT NULL,
            mois integer NOT NULL,
            zone_id integer NOT NULL,
            zone_name text NULL,
            mode_transport text NOT NULL,
            energie_kwh_predite double precision NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (pred_date, zone_id, mode_transport)
        );
        """.strip()

        cur.execute(ddl_co2)
        cur.execute(ddl_energy)

        co2_rows = list(
            dfp[["pred_date", "annee", "mois_int", "zone_id", "zone_name", "mode_transport", "co2_kg_predite"]]
            .fillna({"zone_name": ""})
            .itertuples(index=False, name=None)
        )
        energy_rows = list(
            dfp[["pred_date", "annee", "mois_int", "zone_id", "zone_name", "mode_transport", "energie_kwh_predite"]]
            .fillna({"zone_name": ""})
            .itertuples(index=False, name=None)
        )

        ins_co2 = f"""
        INSERT INTO {_qident(schema)}.{_qident(co2_table)}
            (pred_date, annee, mois, zone_id, zone_name, mode_transport, co2_kg_predite)
        VALUES %s
        ON CONFLICT (pred_date, zone_id, mode_transport)
        DO UPDATE SET
            annee = EXCLUDED.annee,
            mois = EXCLUDED.mois,
            zone_name = EXCLUDED.zone_name,
            co2_kg_predite = EXCLUDED.co2_kg_predite;
        """.strip()

        ins_energy = f"""
        INSERT INTO {_qident(schema)}.{_qident(energy_table)}
            (pred_date, annee, mois, zone_id, zone_name, mode_transport, energie_kwh_predite)
        VALUES %s
        ON CONFLICT (pred_date, zone_id, mode_transport)
        DO UPDATE SET
            annee = EXCLUDED.annee,
            mois = EXCLUDED.mois,
            zone_name = EXCLUDED.zone_name,
            energie_kwh_predite = EXCLUDED.energie_kwh_predite;
        """.strip()

        execute_values(cur, ins_co2, co2_rows, page_size=1000)
        execute_values(cur, ins_energy, energy_rows, page_size=1000)

        conn.commit()

        print("\n✅ Tables PostgreSQL mises à jour:")
        print(f"  - {schema}.{co2_table}  (lignes upsert: {len(co2_rows):,})")
        print(f"  - {schema}.{energy_table}  (lignes upsert: {len(energy_rows):,})")
    finally:
        try:
            conn.close()
        except Exception:
            pass


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



# ============ CHARGEMENT DES DONNÉES ============
print("[OBJECTIF 4] ESTIMER CO2 ET CONSOMMATION ENERGETIQUE\n")
print("[DATA] CHARGEMENT DES DONNEES...")

LOADED_FROM_DB = False
LOADED_FROM_DB_JOIN = False
SQL_JOIN_QUERY = None

if _db_enabled():
    print("🗄️ Mode DB: lecture depuis PostgreSQL (variables DB_* détectées)")
    LOADED_FROM_DB = True
    schema = os.getenv("DB_SCHEMA", "public")

    # Résolution robuste des noms de tables (supporte fautes de frappe dans la DB)
    fact_table = _resolve_table(
        preferred="fact_energieconsomation",
        schema=schema,
        like_patterns=["fact_energie%", "fact_energi%", "fact_energiec%"],
    )
    dim_em_table = _resolve_table(
        preferred="dim_emission_co2",
        schema=schema,
        like_patterns=[
            "dim_emission%",
            "dim_em_%co2%",
            "dim_em_emission%",
            "dim_emco2%",
        ],
    )
    dim_en_table = _resolve_table(
        preferred="dim_energietransport",
        schema=schema,
        like_patterns=["dim_energie%", "dim_energi%"],
    )
    dim_zone_table = _resolve_table(
        preferred="dim_zone",
        schema=schema,
        like_patterns=["dim_zone%"],
    )

    if fact_table is None:
        raise RuntimeError(
            "Table fact introuvable dans la DB (schema public). "
            "Vérifie le nom (ex: fact_energieconsomation / fact_energiecondomation)."
        )

    print("\n📌 Tables DB utilisées:")
    print(f"  - fact: {schema}.{fact_table}")
    if dim_em_table:
        print(f"  - dim_emission: {schema}.{dim_em_table}")
    else:
        print("  - dim_emission: (non trouvée)")
    if dim_en_table:
        print(f"  - dim_energie: {schema}.{dim_en_table}")
    else:
        print("  - dim_energie: (non trouvée)")
    if dim_zone_table:
        print(f"  - dim_zone: {schema}.{dim_zone_table}")
    else:
        print("  - dim_zone: (non trouvée)")

    # Charger les tables (pour exploration/logs)
    fact_energieconsomation = _read_table(fact_table, schema=schema)
    dim_emission_co2 = _read_table(dim_em_table, schema=schema) if dim_em_table else pd.DataFrame()
    dim_energietransport = _read_table(dim_en_table, schema=schema) if dim_en_table else pd.DataFrame()
    dim_zone = _read_table(dim_zone_table, schema=schema) if dim_zone_table else None

    # Optionnel
    try:
        fact_pollution = _read_table("fact_pollution", schema=schema)
    except Exception:
        fact_pollution = None

    # Requête jointe recommandée (peut échouer si fk_zone n'existe pas, etc.)
    joined_df, SQL_JOIN_QUERY = _try_read_joined_fact(
        schema=schema,
        fact_table=fact_table,
        dim_emission_table=dim_em_table,
        dim_energie_table=dim_en_table,
        dim_zone_table=dim_zone_table,
    )
    if joined_df is not None:
        df = joined_df
        LOADED_FROM_DB_JOIN = True
        print("✓ Requête SQL jointe exécutée (fact + dimensions)")
        print("\n🧾 Requête SQL utilisée:")
        print(SQL_JOIN_QUERY)
    else:
        # Fallback: on utilisera les merges pandas plus bas
        df = fact_energieconsomation.copy()
        print("⚠️ Requête SQL jointe non disponible; fallback merges pandas")

else:
    raise RuntimeError(
        "Mode fichiers désactivé. Ce script est configuré pour utiliser PostgreSQL uniquement. "
        "Vérifie DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD (par défaut: localhost/5432/urbain_dw/postgres)."
    )

# Mapping noms de zones si dispo
if dim_zone is not None and isinstance(dim_zone, pd.DataFrame) and len(dim_zone) > 0:
    zone_id_col = 'zone_id' if 'zone_id' in dim_zone.columns else None
    zone_name_col = 'zone_name' if 'zone_name' in dim_zone.columns else (
        'zone_nom' if 'zone_nom' in dim_zone.columns else None
    )
    if zone_id_col and zone_name_col:
        zone_name_mapping = dict(zip(dim_zone[zone_id_col], dim_zone[zone_name_col]))
    else:
        zone_name_mapping = {}
    print(f"✓ dim_zone chargé: {dim_zone.shape}")
    print(f"  Colonnes: {list(dim_zone.columns)}")
else:
    zone_name_mapping = {}
    print("⚠️ dim_zone non disponible - utilisation des IDs")

print(f"✓ fact_energieconsomation chargé: {fact_energieconsomation.shape}")
print(f"✓ dim_emission_co2 chargé: {dim_emission_co2.shape}")
print(f"✓ dim_energietransport chargé: {dim_energietransport.shape}")
if fact_pollution is not None:
    print(f"✓ fact_pollution chargé: {fact_pollution.shape}")
else:
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

# Fusion des données (si df n'est pas déjà enrichi via SQL JOIN)
if not LOADED_FROM_DB_JOIN:
    df = df.copy()

# Joindre dim_emission_co2 - utiliser seulement les colonnes disponibles
if (not LOADED_FROM_DB_JOIN) and ('fk_emco2' in df.columns):
    emco2_cols = ['emission_id']
    if 'mode' in dim_emission_co2.columns:
        emco2_cols.append('mode')
    if 'activity_type' in dim_emission_co2.columns:
        emco2_cols.append('activity_type')
    if len(emco2_cols) > 1:
        df = df.merge(dim_emission_co2[emco2_cols], 
                      left_on='fk_emco2', right_on='emission_id', how='left')

# Joindre dim_energietransport - utiliser seulement les colonnes disponibles
if (not LOADED_FROM_DB_JOIN) and ('fk_energie' in df.columns):
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

# Joindre dim_zone si possible (pour zone_name)
if (not LOADED_FROM_DB_JOIN) and (dim_zone is not None) and isinstance(dim_zone, pd.DataFrame):
    try:
        if 'fk_zone' in df.columns and 'zone_id' in dim_zone.columns:
            keep = ['zone_id'] + ([c for c in ['zone_name', 'zone_nom'] if c in dim_zone.columns][:1])
            if len(keep) > 1:
                df = df.merge(dim_zone[keep], left_on='fk_zone', right_on='zone_id', how='left')
    except Exception:
        pass

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

    # Mode rapide: on réduit fortement le tuning (beaucoup plus rapide)
    cv = KFold(n_splits=3 if FAST_MODE else 5, shuffle=True, random_state=42)

    # Modèle 1: Ridge (toujours)
    ridge_pipe = Pipeline(steps=[('prep', preprocess), ('reg', Ridge())])
    ridge_grid = {'reg__alpha': ([1.0, 10.0] if FAST_MODE else [0.1, 1.0, 10.0, 50.0])}
    ridge_search = GridSearchCV(
        ridge_pipe,
        ridge_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )

    searches = [('Ridge', ridge_search)]

    if not FAST_MODE:
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
    if not ONLY_DB:
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

    if not ONLY_DB:
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


def _mlops_export_objective4() -> None:
    if os.getenv("MLOPS_EXPORT", "0").strip().lower() not in ("1", "true", "yes"):
        return

    from mlops.registry import register_model

    if "feature_cols" not in globals() or "best_model_co2" not in globals():
        print("⚠️ MLOps export ignoré: variables manquantes (feature_cols/best_model_co2).")
        return

    models = {"co2_kg": best_model_co2}
    targets = ["co2_kg"]
    if best_model_energy is not None:
        models["energie_kwh"] = best_model_energy
        targets.append("energie_kwh")

    bundle = {"models": models}
    meta = {
        "feature_cols": list(feature_cols),
        "targets": targets,
        "fast_mode": bool(FAST_MODE),
        "only_db": bool(ONLY_DB),
    }

    try:
        entry = register_model("objective4", bundle, meta)
        print({"status": "success", "objective": "objective4", "version": entry.version})
    except Exception as e:
        print({"status": "error", "message": f"MLOps export failed: {e}"})


_mlops_export_objective4()

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

# Période demandée: 2027, 2028, 2029 (36 mois)
future_dates = pd.date_range(start=pd.Timestamp("2027-01-01"), periods=36, freq='MS')

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

# Écriture dans PostgreSQL (2 tables) — années 2027-2029
try:
    _write_predictions_to_postgres(predictions_df)
except Exception as e:
    print(f"⚠️ Écriture PostgreSQL ignorée (erreur): {e}")

print(f"\n✓ {len(predictions_df):,} prédictions générées")
print(f"✓ Période: {predictions_df['date'].min().strftime('%Y-%m-%d')} à {predictions_df['date'].max().strftime('%Y-%m-%d')}")

if ONLY_DB:
    print("\n✅ Terminé (ONLY_DB=1): tables écrites dans PostgreSQL, rapport/visualisations ignorés.")
    raise SystemExit(0)

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
# IMPORTANT (robustesse export): on borne la taille des bulles pour éviter des figures énormes.
efficiency_data = []
for mode in predictions_df['mode_transport'].unique():
    mode_data = predictions_df[predictions_df['mode_transport'] == mode]
    total_energy_kwh = float(mode_data['energie_kwh_predite'].sum())
    total_co2_kg = float(mode_data['co2_kg_predite'].sum())

    # Conversion: MWh = kWh / 1e3 ; ratio en kg CO2 / MWh (plus bas = plus écologique)
    total_energy_mwh = total_energy_kwh / 1e3
    total_co2_tonnes = total_co2_kg / 1000.0
    ratio_kg_per_mwh = (total_co2_kg / total_energy_mwh) if total_energy_mwh > 0 else np.nan

    efficiency_data.append(
        {
            'mode': str(mode)[:15],
            'ratio_kg_per_mwh': ratio_kg_per_mwh,
            'energy_mwh': total_energy_mwh,
            'co2_tonnes': total_co2_tonnes,
        }
    )

efficiency_df = pd.DataFrame(efficiency_data).sort_values('ratio_kg_per_mwh')
efficiency_df = efficiency_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ratio_kg_per_mwh'])

# Bubble chart : x=énergie, y=CO2, size=ratio (normalisé + borné)
if len(efficiency_df) > 0:
    r = efficiency_df['ratio_kg_per_mwh'].to_numpy(dtype=float)
    r_min = float(np.nanmin(r))
    r_max = float(np.nanmax(r))
    denom = (r_max - r_min) if (r_max > r_min) else 1.0
    r_norm = (r - r_min) / denom
    # tailles en points^2, bornées pour éviter les exports monstrueux
    sizes = 200.0 + 900.0 * np.clip(r_norm, 0.0, 1.0)

    scatter = ax11.scatter(
        efficiency_df['energy_mwh'],
        efficiency_df['co2_tonnes'],
        s=sizes,
        c=efficiency_df['ratio_kg_per_mwh'],
        cmap='RdYlGn_r',
        alpha=0.75,
        edgecolors='#2c3e50',
        linewidth=2,
    )
else:
    scatter = None

# Ajouter labels sur points
for idx, row in efficiency_df.iterrows():
    ax11.annotate(row['mode'], (row['energy_mwh'], row['co2_tonnes']), 
                 fontsize=9, fontweight='bold', ha='center', va='center')

ax11.set_xlabel('Énergie Totale (MWh)', fontsize=12, fontweight='bold')
ax11.set_ylabel('CO2 Total (Tonnes)', fontsize=12, fontweight='bold')
ax11.set_title('♻️ EFFICACITÉ ÉCOLOGIQUE\n(CO2/MWh)', fontsize=13, fontweight='bold')
ax11.grid(True, alpha=0.3, linestyle='--')
ax11.set_facecolor('#f8f9fa')

# Colorbar
if scatter is not None:
    cbar = plt.colorbar(scatter, ax=ax11)
    cbar.set_label('kg CO2 / MWh', fontweight='bold')
fig_energie.suptitle('⚡ RAPPORT COMPLET - ANALYSE DE LA CONSOMMATION ÉNERGÉTIQUE - PRÉDICTIONS 36 MOIS (AMÉLIORÉ)', 
                     fontsize=19, fontweight='bold', y=0.998)
fig_energie.patch.set_facecolor('white')

try:
    fig_energie.savefig(
        'Visualisations_Energie_Professional.png',
        dpi=220,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
except ValueError as e:
    print(f"⚠️ Export tight échoué ({e}). Export sans bbox_inches='tight' ...")
    fig_energie.savefig(
        'Visualisations_Energie_Professional.png',
        dpi=180,
        facecolor='white',
        edgecolor='none',
    )

print("✓ Énergie Professional (AMÉLIORÉ avec efficacité écologique): Visualisations_Energie_Professional.png")
plt.show()

print("\n✅ Visualisations professionnelles créées avec succès!")

print("\n✅ OBJECTIF 4 TERMINÉ!")
print("=" * 70)
