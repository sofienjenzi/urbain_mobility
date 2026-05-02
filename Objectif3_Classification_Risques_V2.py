"""OBJECTIF 3 — Classification des niveaux de risque (3 modèles).

Ce module est utilisé par l'API (voir api2.py). Contraintes importantes:
- Ne PAS entraîner le modèle à l'import.
- Entraîner/charger à la demande (cache .pkl).
- Comparer 3 modèles (LogReg / RandomForest / SVM RBF) et retenir le meilleur (F1).

Correction principale vs versions "light":
- Target plus logique (on évite d'utiliser le volume comme indicateur de risque par défaut).
- Seuils de création de la target calculés uniquement sur le train (évite la fuite de données).
"""

from __future__ import annotations

# IMPORTANT: limiter les threads BLAS *avant* l'import de numpy/scikit
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC

LOGGER = logging.getLogger(__name__)


# ============================
# CONFIG DB (surcharge possible via env DB_*)
# ============================
DB_ENV_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "urbain_dw")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "admin")
os.environ.setdefault("DB_SCHEMA", "public")

# FAST_MODE=1 => tuning plus léger + chargement DB limité (recommandé pour API)
FAST_MODE = os.getenv("FAST_MODE", "1").strip().lower() not in ("0", "false", "no")

_ENGINE = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _qident(name: str) -> str:
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
    """Lit DB_LIMIT_ROWS si défini, sinon applique un défaut en FAST_MODE."""
    v = os.getenv("DB_LIMIT_ROWS", "").strip()
    if v:
        try:
            n = int(v)
            return n if n > 0 else None
        except Exception:
            return None
    return 20000 if FAST_MODE else None


def _list_tables(schema: str = "public") -> list[str]:
    sql = (
        "SELECT table_name FROM information_schema.tables "
        f"WHERE table_schema='{schema}' AND table_type='BASE TABLE' ORDER BY table_name;"
    )
    d = _read_sql(sql)
    if "table_name" not in d.columns:
        return []
    return [str(x) for x in d["table_name"].dropna().tolist()]


def _resolve_table(
    preferred: str,
    schema: str = "public",
    like_patterns: list[str] | None = None,
) -> str | None:
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
            if "table_name" in m.columns and len(m) > 0:
                return str(m["table_name"].iloc[0])

    close = get_close_matches(preferred, tables, n=1, cutoff=0.65)
    return close[0] if close else None


def _read_table(table: str, schema: str = "public", limit: int | None = None) -> pd.DataFrame:
    lim = limit if limit is not None else _db_limit_rows()
    if lim is not None:
        return _read_sql(f"SELECT * FROM {_qident(schema)}.{_qident(table)} LIMIT {int(lim)};")
    return _read_sql(f"SELECT * FROM {_qident(schema)}.{_qident(table)};")


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clippe les valeurs extrêmes par quantiles, colonne par colonne."""

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


# Rend la sérialisation (pickle/joblib) plus robuste quand le script est exécuté en tant que __main__.
# Sans ça, les pickles peuvent référencer '__main__.QuantileClipper' et être impossibles à recharger depuis l'API.
try:
    QuantileClipper.__module__ = Path(__file__).stem
except Exception:
    pass


# ============================
# MODEL CACHE
# ============================
@dataclass(frozen=True)
class ModelInfo:
    best_model: str
    best_f1: float
    metrics: list[dict[str, Any]]
    feature_cols: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    target_mode: str
    target_definition: str
    target_thresholds: dict[str, float]
    target_metrics: list[str]
    trained_at: str


def _cache_paths() -> tuple[Path, Path]:
    base = Path(__file__).resolve().parent
    # v2: évite d'essayer de charger d'anciens pickles incompatibles
    model_path = base / "Objectif3_best_model_v2.pkl"
    info_path = base / "Objectif3_model_info_v2.pkl"
    return model_path, info_path


def _load_cached() -> tuple[Pipeline | None, ModelInfo | None]:
    model_path, info_path = _cache_paths()
    if not model_path.exists() or not info_path.exists():
        return None, None
    try:
        with model_path.open("rb") as f:
            model = pickle.load(f)
        with info_path.open("rb") as f:
            info = pickle.load(f)
        if not isinstance(info, ModelInfo):
            return None, None
        return model, info
    except Exception as e:
        LOGGER.warning("Cache model load failed: %s", e)
        return None, None


def _save_cached(model: Pipeline, info: ModelInfo) -> None:
    model_path, info_path = _cache_paths()
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    with info_path.open("wb") as f:
        pickle.dump(info, f)


def _norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip().lower()
    s = (
        s.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("ç", "c")
    )
    return s


def _gravite_to_risk(gravite: Any) -> int | None:
    """Mappe une gravité accident en binaire.

    Retourne 1 si gravité sévère, 0 si non-sévère, None si inconnue.
    """
    s = _norm_text(gravite)
    if not s or s in ("nan", "none", "n/a"):
        return None

    severe = {
        "tres grave",
        "tres_grave",
        "tresgrave",
        "grave",
        "eleve",
        "elevé",
        "mortel",
        "fatal",
    }
    mild = {
        "faible",
        "moyen",
        "mineur",
        "leger",
        "legere",
        "modere",
        "moderé",
    }

    if s in severe:
        return 1
    if s in mild:
        return 0

    # Si on a un chiffre déguisé
    try:
        v = float(s)
        return 1 if v >= 4 else 0
    except Exception:
        return None


# ============================
# DATA LOADING / PREP
# ============================
def _load_from_db() -> pd.DataFrame:
    if not _db_enabled():
        raise RuntimeError(
            "Connexion PostgreSQL non configurée. Variables requises: "
            "DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD."
        )

    schema = os.getenv("DB_SCHEMA", "public")

    fact_table = _resolve_table(
        preferred=os.getenv("FACT_SAFETYROAD_TABLE", "fact_safetyroad"),
        schema=schema,
        like_patterns=["fact_safety%", "fact_safe%", "fact_safetyroad%"],
    )
    if fact_table is None:
        raise RuntimeError(f"Table fact_safetyroad introuvable dans le schéma {schema}.")

    dim_zone_table = _resolve_table(
        preferred=os.getenv("DIM_ZONE_TABLE", "dim_zone"),
        schema=schema,
        like_patterns=["dim_zone%"],
    )
    if dim_zone_table is None:
        raise RuntimeError(f"Table dim_zone introuvable dans le schéma {schema}.")

    dim_acc_table = _resolve_table(
        preferred=os.getenv("DIM_ACCIDENTS_TABLE", "dim_accidents"),
        schema=schema,
        like_patterns=["dim_accident%", "dim_accidents%"],
    )
    dim_del_table = _resolve_table(
        preferred=os.getenv("DIM_DELINQUENCE_TABLE", "dim_delinquence"),
        schema=schema,
        like_patterns=["dim_delin%", "dim_crime%", "dim_delinquence%"],
    )

    fact_safetyroad = _read_table(fact_table, schema=schema)
    dim_zone = _read_table(dim_zone_table, schema=schema)
    dim_accidents = _read_table(dim_acc_table, schema=schema) if dim_acc_table else None
    dim_delinquence = _read_table(dim_del_table, schema=schema) if dim_del_table else None

    df = fact_safetyroad.copy()
    if len(df) == 0:
        raise ValueError("Aucune donnée disponible: fact_safetyroad est vide")

    # Harmoniser types clés
    for k in ["fk_zone", "fk_accident", "fk_crime"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")

    if "zone_id" in dim_zone.columns:
        dim_zone["zone_id"] = pd.to_numeric(dim_zone["zone_id"], errors="coerce").astype("Int64")

    if dim_accidents is not None and "accident_id" in dim_accidents.columns:
        dim_accidents["accident_id"] = pd.to_numeric(dim_accidents["accident_id"], errors="coerce").astype("Int64")

    if dim_delinquence is not None and "crime_id" in dim_delinquence.columns:
        dim_delinquence["crime_id"] = pd.to_numeric(dim_delinquence["crime_id"], errors="coerce").astype("Int64")
        if "periode_mois" in dim_delinquence.columns and "date" not in dim_delinquence.columns:
            pm = dim_delinquence["periode_mois"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 6)
            dim_delinquence["date"] = pd.to_datetime(pm, format="%Y%m", errors="coerce")

    # Merge dim_zone
    zone_name_col = "zone_nom" if "zone_nom" in dim_zone.columns else ("zone_name" if "zone_name" in dim_zone.columns else None)
    zone_keep_cols = ["zone_id"] + ([zone_name_col] if zone_name_col else [])
    if "fk_zone" in df.columns and "zone_id" in dim_zone.columns:
        df = df.merge(dim_zone[zone_keep_cols], left_on="fk_zone", right_on="zone_id", how="left")
    else:
        if "zone_id" not in df.columns:
            df["zone_id"] = 1
        if zone_name_col and zone_name_col not in df.columns:
            df[zone_name_col] = df["zone_id"].apply(lambda z: f"Zone_{z}")

    if zone_name_col is None:
        df["zone_nom"] = df.get("zone_nom", df.get("zone_name", "Zone"))
    else:
        if zone_name_col != "zone_nom":
            df["zone_nom"] = df[zone_name_col]

    # Merge dim_accidents
    if dim_accidents is not None and "fk_accident" in df.columns and "accident_id" in dim_accidents.columns:
        keep = ["accident_id"]
        for c in ["type", "gravite", "severity", "accident_type"]:
            if c in dim_accidents.columns:
                keep.append(c)
        df = df.merge(dim_accidents[keep], left_on="fk_accident", right_on="accident_id", how="left")

    # Merge dim_delinquence
    if dim_delinquence is not None and "fk_crime" in df.columns and "crime_id" in dim_delinquence.columns:
        keep = ["crime_id"]
        for c in ["periode_mois", "categorie", "category", "date"]:
            if c in dim_delinquence.columns:
                keep.append(c)
        df = df.merge(dim_delinquence[keep], left_on="fk_crime", right_on="crime_id", how="left")

    return df


def _make_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns and pd.to_datetime(df["date"], errors="coerce").notna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "periode_mois" in df.columns and df["periode_mois"].notna().any():
        pm = df["periode_mois"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 6)
        df["date"] = pd.to_datetime(pm, format="%Y%m", errors="coerce")
        if df["date"].isna().all():
            df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
    else:
        df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Retourne df_model + listes (num_cols, cat_cols) utilisables."""
    df = _make_time_columns(df)

    def to_float(col: str) -> pd.Series:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        # Supporte virgule décimale + séparateurs divers
        s2 = s.astype(str).str.strip().str.replace(" ", "")
        s2 = s2.str.replace(",", ".", regex=False)
        return pd.to_numeric(s2, errors="coerce")

    numeric_cols: list[str] = []
    if "volume" in df.columns:
        df["volume"] = to_float("volume")
        numeric_cols.append("volume")
    if "taux_1000" in df.columns:
        df["taux_1000"] = to_float("taux_1000")
        numeric_cols.append("taux_1000")

    if "gravite" in df.columns:
        gravite_map = {"faible": 1, "moyen": 2, "élevé": 3, "grave": 4, "très grave": 5}
        df["gravite_score"] = df["gravite"].map(gravite_map).fillna(2)
        numeric_cols.append("gravite_score")

    if "usager_vulnerable" in df.columns:
        df["vulnerable_count"] = pd.to_numeric(df["usager_vulnerable"], errors="coerce").fillna(0)
        numeric_cols.append("vulnerable_count")

    for c in ["year", "month", "zone_id"]:
        if c in df.columns and c not in numeric_cols:
            numeric_cols.append(c)

    cat_cols: list[str] = []
    for c in ["type", "gravite", "categorie", "zone_nom"]:
        if c in df.columns:
            cat_cols.append(c)

    base_cols = list(dict.fromkeys(numeric_cols + cat_cols + ["date"]))
    base_cols = [c for c in base_cols if c in df.columns]
    if not base_cols:
        raise ValueError(f"Aucune colonne utilisable pour le ML. Colonnes dispo: {list(df.columns)}")
    df_model = df[base_cols].copy().replace([np.inf, -np.inf], np.nan)

    if FAST_MODE and len(df_model) > 5000:
        df_model = df_model.sample(5000, random_state=42)

    return df_model, numeric_cols, cat_cols


def _compute_thresholds(train_df: pd.DataFrame, metrics: list[str], q: float = 0.75) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for m in metrics:
        if m not in train_df.columns:
            continue
        ser = pd.to_numeric(train_df[m], errors="coerce")
        thr = ser.quantile(q)
        if pd.isna(thr):
            continue
        thresholds[m] = float(thr)
    return thresholds


def _apply_threshold_target(df: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    if not thresholds:
        return np.zeros(len(df), dtype=int)
    y = np.zeros(len(df), dtype=int)
    for m, thr in thresholds.items():
        ser = pd.to_numeric(df[m], errors="coerce")
        y = np.maximum(y, (ser > thr).astype(int).to_numpy())
    return y.astype(int)


def _ensure_two_classes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_metrics: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float], list[str]]:
    """Construit y_train/y_test. Fallback proxy si 1 seule classe."""
    thresholds = _compute_thresholds(train_df, candidate_metrics, q=0.75)
    y_train = _apply_threshold_target(train_df, thresholds)
    y_test = _apply_threshold_target(test_df, thresholds)

    if len(np.unique(y_train)) >= 2:
        return y_train, y_test, thresholds, list(thresholds.keys())

    # Fallback: proxy score basé sur rangs percentiles (train uniquement)
    proxy_feats = [c for c in candidate_metrics if c in train_df.columns and train_df[c].notna().any()]
    if not proxy_feats:
        proxy_feats = [c for c in train_df.columns if c not in ("date",) and train_df[c].dtype != object]
    if not proxy_feats:
        return y_train, y_test, thresholds, list(thresholds.keys())

    tmp_train = train_df[proxy_feats].apply(pd.to_numeric, errors="coerce")
    tmp_train = tmp_train.fillna(tmp_train.median(numeric_only=True))
    proxy_score_train = tmp_train.rank(pct=True).mean(axis=1)
    proxy_thr = float(proxy_score_train.quantile(0.75))

    tmp_test = test_df[proxy_feats].apply(pd.to_numeric, errors="coerce")
    tmp_test = tmp_test.fillna(tmp_train.median(numeric_only=True))
    proxy_score_test = tmp_test.rank(pct=True).mean(axis=1)

    y_train = (proxy_score_train > proxy_thr).astype(int).to_numpy()
    y_test = (proxy_score_test > proxy_thr).astype(int).to_numpy()
    thresholds = {"proxy_score": proxy_thr}
    return y_train, y_test, thresholds, ["proxy_score"]


def _build_preprocess(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", QuantileClipper(lower_q=0.01, upper_q=0.99)),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


# ============================
# TRAINING
# ============================
def _train_models(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, preprocess: ColumnTransformer):
    cv = StratifiedKFold(n_splits=(3 if FAST_MODE else 5), shuffle=True, random_state=42)

    pip_lr = Pipeline(
        steps=[
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    grid_lr = {
        "clf__C": ([1.0, 5.0] if FAST_MODE else [0.1, 1.0, 5.0]),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }

    pip_rf = Pipeline(
        steps=[
            ("prep", preprocess),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
        ]
    )
    grid_rf = {
        "clf__n_estimators": ([200] if FAST_MODE else [200, 400]),
        "clf__max_depth": ([None, 14] if FAST_MODE else [None, 8, 14]),
        "clf__min_samples_split": ([2, 5] if FAST_MODE else [2, 5]),
    }

    pip_svm = Pipeline(
        steps=[
            ("prep", preprocess),
            ("clf", SVC(class_weight="balanced", probability=True)),
        ]
    )
    grid_svm = {
        "clf__C": ([1.0] if FAST_MODE else [0.5, 1.0, 5.0]),
        "clf__gamma": (["scale"] if FAST_MODE else ["scale", "auto"]),
        "clf__kernel": ["rbf"],
    }

    searches = [
        ("LogisticRegression", GridSearchCV(pip_lr, grid_lr, cv=cv, scoring="f1", n_jobs=-1)),
        ("RandomForest", GridSearchCV(pip_rf, grid_rf, cv=cv, scoring="f1", n_jobs=-1)),
        ("SVM_RBF", GridSearchCV(pip_svm, grid_svm, cv=cv, scoring="f1", n_jobs=-1)),
    ]

    results: list[dict[str, Any]] = []
    best_name: str | None = None
    best_est: Pipeline | None = None
    best_f1 = -1.0

    for name, search in searches:
        LOGGER.info("GridSearchCV: %s", name)
        search.fit(X_train, y_train)
        est: Pipeline = search.best_estimator_
        y_hat = est.predict(X_test)
        y_proba = est.predict_proba(X_test)[:, 1]

        row = {
            "model": name,
            "best_params": search.best_params_,
            "accuracy": float(accuracy_score(y_test, y_hat)),
            "precision": float(precision_score(y_test, y_hat, zero_division=0)),
            "recall": float(recall_score(y_test, y_hat, zero_division=0)),
            "f1": float(f1_score(y_test, y_hat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) >= 2 else float("nan"),
            "confusion_matrix": confusion_matrix(y_test, y_hat, labels=[0, 1]).tolist(),
        }
        results.append(row)

        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_name = name
            best_est = est

    if best_est is None or best_name is None:
        raise RuntimeError("Aucun modèle n'a pu être entraîné")

    results_sorted = sorted(results, key=lambda r: (r["f1"], (r["roc_auc"] if not np.isnan(r["roc_auc"]) else -1.0)), reverse=True)
    return best_name, best_est, best_f1, results_sorted


def train_and_cache(force: bool = False) -> tuple[Pipeline, ModelInfo]:
    cached_model, cached_info = _load_cached()
    if cached_model is not None and cached_info is not None and not force:
        return cached_model, cached_info

    LOGGER.info("Training Objective3 models (FAST_MODE=%s)", FAST_MODE)
    df_raw = _load_from_db()
    df_model, numeric_cols, cat_cols = _build_features(df_raw)

    # ===== Target + features (anti-leakage) =====
    # 1) Mode préféré: label basé sur la gravité (plus logique), et on retire gravite/gravite_score des features.
    target_mode = "quantile"  # fallback
    target_definition = ""
    thresholds: dict[str, float] = {}
    used_metrics: list[str] = []

    grav_ok = False
    if "gravite" in df_model.columns:
        y_all = df_model["gravite"].map(_gravite_to_risk)
        y_all = y_all.dropna()
        if len(y_all) >= 100:
            classes, counts = np.unique(y_all.astype(int).to_numpy(), return_counts=True)
            if len(classes) >= 2 and counts.min() >= 10:
                grav_ok = True

    if grav_ok:
        target_mode = "gravite"
        target_definition = (
            "risk_level=1 si la gravité accident est sévère (grave/très grave/élevé/mortel), "
            "sinon 0. Les lignes sans gravité sont ignorées pour l'entraînement."
        )
        y_series = df_model["gravite"].map(_gravite_to_risk)
        mask = y_series.notna()
        y_all = y_series.loc[mask].astype(int).to_numpy()

        # features: tout sauf la colonne label et ses dérivés
        feature_cols = [c for c in (numeric_cols + cat_cols) if c in df_model.columns]
        for drop_col in ["gravite", "gravite_score"]:
            if drop_col in feature_cols:
                feature_cols.remove(drop_col)

        X_all = df_model.loc[mask, feature_cols].copy()

        if len(X_all) < 50:
            raise ValueError("Pas assez de données avec gravité pour entraîner un modèle (min 50 lignes).")

        # Split avec stratify si possible
        classes, counts = np.unique(y_all, return_counts=True)
        stratify_arg = y_all if (len(classes) >= 2 and counts.min() >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=42,
            stratify=stratify_arg,
        )
        used_metrics = ["gravite"]
    else:
        # 2) Fallback: quantile-based sur taux_1000 (seuils calculés sur train), ET on retire taux_1000 des features.
        target_mode = "quantile"
        target_definition = (
            "risk_level=1 si au-dessus du P75 (calculé sur train) d'au moins un indicateur de risque; "
            "par défaut: taux_1000 si disponible."
        )

        # features: éviter la tautologie -> on retire taux_1000 des features si utilisé pour créer la target.
        feature_cols = [c for c in (numeric_cols + cat_cols) if c in df_model.columns]
        if "taux_1000" in feature_cols:
            feature_cols.remove("taux_1000")

        X_all = df_model[feature_cols].copy()
        if len(X_all) < 50:
            raise ValueError("Pas assez de données pour entraîner un modèle (min 50 lignes).")

        X_train, X_test = train_test_split(X_all, test_size=0.2, random_state=42)
        candidate_metrics = [c for c in ["taux_1000", "vulnerable_count"] if c in df_model.columns]
        y_train, y_test, thresholds, used_metrics = _ensure_two_classes(
            train_df=df_model.loc[X_train.index],
            test_df=df_model.loc[X_test.index],
            candidate_metrics=candidate_metrics,
        )
        if len(np.unique(y_train)) < 2:
            raise ValueError("La target 'risk_level' ne contient qu'une seule classe après construction.")

    # Preprocess: num vs cat
    num_cols = [c for c in numeric_cols if c in feature_cols]
    cat_cols2 = [c for c in cat_cols if c in feature_cols]
    preprocess = _build_preprocess(num_cols=num_cols, cat_cols=cat_cols2)

    best_name, best_est, best_f1, results_sorted = _train_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocess=preprocess,
    )

    info = ModelInfo(
        best_model=best_name,
        best_f1=float(best_f1),
        metrics=results_sorted,
        feature_cols=feature_cols,
        num_cols=num_cols,
        cat_cols=cat_cols2,
        target_mode=target_mode,
        target_definition=target_definition,
        target_thresholds=thresholds,
        target_metrics=used_metrics,
        trained_at=_now_iso(),
    )

    _save_cached(best_est, info)
    return best_est, info


# ============================
# API ENTRYPOINT
# ============================
def run_Classification(input_data: dict[str, Any] | None = None, *, force_retrain: bool = False) -> dict[str, Any]:
    """Fonction appelée par l'API.

    - Si input_data est fourni: prédit le risque pour une observation.
    - Sinon: utilise un exemple par défaut.
    """

    try:
        model, info = train_and_cache(force=force_retrain)
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": _now_iso(),
        }

    if input_data is None:
        # Valeurs par défaut cohérentes avec les données DB (taux_1000 ~ 0.6–1.0)
        input_data = {"volume": 1500, "taux_1000": 0.8}

    # Construire une ligne avec toutes les colonnes attendues par le pipeline
    row: dict[str, Any] = {}
    for c in info.feature_cols:
        v = input_data.get(c, np.nan)
        if c in info.num_cols and v is not np.nan:
            if isinstance(v, str):
                v = v.strip().replace(" ", "").replace(",", ".")
            try:
                v = float(v)
            except Exception:
                v = np.nan
        row[c] = v
    X_one = pd.DataFrame([row], columns=info.feature_cols)

    try:
        pred = int(model.predict(X_one)[0])
        proba = float(model.predict_proba(X_one)[0][1])
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {e}",
            "timestamp": _now_iso(),
            "expected_features": info.feature_cols,
        }

    return {
        "status": "success",
        "model": info.best_model,
        "f1_score": float(info.best_f1),
        "input": input_data,
        "prediction": pred,
        "risk_score": proba,
        "risk_level": "HIGH" if pred == 1 else "LOW",
        "trained_at": info.trained_at,
        "timestamp": _now_iso(),
        # Rétrocompat / debug
        "all_models": info.metrics,
        "target": {
            "mode": info.target_mode,
            "definition": info.target_definition,
            "metrics_used": info.target_metrics,
            "thresholds": info.target_thresholds,
        },
    }


def mlops_export_latest(*, force_retrain: bool = False) -> dict[str, Any]:
    """Exporte le meilleur modèle dans le registry local (models/objective3/latest).

    Ne s'exécute que si appelé explicitement (ex: via env MLOPS_EXPORT=1).
    """

    model, info = train_and_cache(force=force_retrain)
    from mlops.registry import register_model

    meta = asdict(info)
    entry = register_model("objective3", model, meta)
    return {
        "status": "success",
        "objective": "objective3",
        "version": entry.version,
        "model_path": str(entry.model_path),
        "metadata_path": str(entry.metadata_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(run_Classification())

    if os.getenv("MLOPS_EXPORT", "0").strip().lower() in ("1", "true", "yes"):
        try:
            print(mlops_export_latest(force_retrain=False))
        except Exception as e:
            print({"status": "error", "message": f"MLOps export failed: {e}"})
