from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("PREDICTION_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Urban MLOps Web App", layout="centered")


def _fmt_float(x: Any, ndigits: int = 3) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{int(ndigits)}f}"
    except Exception:
        return str(x)


def _render_result_header(res: dict[str, Any]) -> None:
    v = res.get("model_version")
    d = res.get("duration_ms")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Version modèle")
        st.write(str(v) if v else "—")
    with c2:
        st.caption("Latence")
        st.write(f"{_fmt_float(d, 1)} ms" if d is not None else "—")


def _render_details(res: dict[str, Any]) -> None:
    with st.expander("Détails (JSON)", expanded=False):
        st.json(res)


def _api_post(objective: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any] | None:
    # If the API isn't reachable (common in "Deploy" scenarios when FastAPI isn't running),
    # fall back to local inference by loading the latest model from the registry.
    try:
        r = requests.post(f"{API_URL}/predict/{objective}", json={"input": payload}, timeout=timeout)
    except Exception as e:
        # Local fallback (enabled by default)
        if os.getenv("ALLOW_LOCAL_FALLBACK", "1").strip().lower() in {"1", "true", "yes"}:
            try:
                return _local_predict(objective, payload)
            except Exception as e2:
                st.error(f"Appel API impossible: {e}\nFallback local échoué: {e2}")
                return None

        st.error(f"Appel API impossible: {e}")
        return None

    if r.status_code >= 400:
        st.error(f"Erreur API ({r.status_code}): {r.text}")
        return None
    return r.json()


def _local_predict(objective: str, payload: dict[str, Any]) -> dict[str, Any]:
    # Lazy import to keep Streamlit startup fast.
    import time

    from mlops.registry import load_latest_model

    # Reuse the serving implementation so results match the API.
    from serving.api import _predict_with_loaded

    start = time.perf_counter()
    model, meta = load_latest_model(f"objective{objective}")
    out = _predict_with_loaded(str(objective), model, meta, payload)
    dur_ms = (time.perf_counter() - start) * 1000.0

    return {"objective": str(objective), "model_version": meta.get("version"), "duration_ms": dur_ms, **out}


def _load_cities_from_db() -> list[str] | None:
    # Defaults align with objective scripts
    os.environ.setdefault("DB_HOST", "localhost")
    os.environ.setdefault("DB_PORT", "5432")
    os.environ.setdefault("DB_NAME", "urbain_dw")
    os.environ.setdefault("DB_USER", "postgres")
    os.environ.setdefault("DB_PASSWORD", "admin")
    os.environ.setdefault("DB_SCHEMA", "public")

    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")
    db_schema = os.getenv("DB_SCHEMA", "public")

    if not all([db_host, db_name, db_user, db_password]):
        return None

    try:
        from sqlalchemy import create_engine  # type: ignore
        from sqlalchemy.engine import URL  # type: ignore
        import pandas as pd  # local import to keep startup light
    except Exception:
        return None

    try:
        url = URL.create(
            drivername="postgresql+psycopg2",
            username=db_user,
            password=db_password,
            host=db_host,
            port=int(db_port),
            database=db_name,
        )
        engine = create_engine(url)
        q = f"SELECT DISTINCT ville FROM {db_schema}.dim_zone WHERE ville IS NOT NULL ORDER BY ville;"
        df = pd.read_sql(q, engine)
        cities = [str(x) for x in df["ville"].dropna().tolist()]
        return cities or None
    except Exception:
        return None


def _load_zones_from_db() -> list[tuple[int, str]] | None:
    """Return list of (zone_id, label) sorted by label."""
    try:
        # Align defaults with objective scripts
        os.environ.setdefault("DB_HOST", "localhost")
        os.environ.setdefault("DB_PORT", "5432")
        os.environ.setdefault("DB_NAME", "urbain_dw")
        os.environ.setdefault("DB_USER", "postgres")
        os.environ.setdefault("DB_PASSWORD", "admin")
        os.environ.setdefault("DB_SCHEMA", "public")

        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
        import pandas as pd  # local import

        host = os.getenv("PGHOST") or os.getenv("DB_HOST") or "localhost"
        port = int(os.getenv("PGPORT") or os.getenv("DB_PORT") or "5432")
        database = os.getenv("PGDATABASE") or os.getenv("DB_NAME") or "urbain_dw"
        user = os.getenv("PGUSER") or os.getenv("DB_USER") or "postgres"
        password = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD") or "admin"
        db_schema = os.getenv("PGSCHEMA") or os.getenv("DB_SCHEMA") or "public"

        url = URL.create(
            drivername="postgresql+psycopg2",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        engine = create_engine(url)
        q = (
            f"SELECT DISTINCT zone_id, btrim(zone_nom) AS zone_nom, ville "
            f"FROM {db_schema}.dim_zone "
            f"WHERE zone_id IS NOT NULL AND zone_nom IS NOT NULL "
            f"ORDER BY zone_nom;"
        )

        df = pd.read_sql(q, engine)
        zones: list[tuple[int, str]] = []
        for _, r in df.iterrows():
            zid = int(r["zone_id"])
            name = str(r.get("zone_nom") or "").strip()
            ville = str(r.get("ville") or "").strip()
            label = name if not ville else f"{name} — {ville}"
            zones.append((zid, label))
        return zones or None
    except Exception:
        return None


st.title("Urban MLOps — Dashboard")
st.caption("API URL: " + API_URL)

tabs = st.tabs(
    [
        "Objectif 1 — Trafic (forecast)",
        "Objectif 2 — Recommandation (trajets)",
        "Objectif 3 — Risques (classification)",
        "Objectif 4 — CO2 & Énergie (estimation)",
    ]
)


with tabs[0]:
    st.subheader("Prévision du trafic")
    steps = st.slider("Nombre de mois à prédire", min_value=1, max_value=36, value=12, step=1)
    if st.button("Prédire (Objectif 1)"):
        res = _api_post("1", {"steps": int(steps)}, timeout=120)
        if res:
            forecast = res.get("forecast") or []
            _render_result_header(res)

            if forecast:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Horizon", value=f"{len(forecast)} mois")
                with c2:
                    st.metric("Prévision M+1", value=_fmt_float(forecast[0], 3))
                with c3:
                    idx = min(11, len(forecast) - 1)
                    st.metric(f"Prévision M+{idx + 1}", value=_fmt_float(forecast[idx], 3))

            # Streamlit uses the DataFrame index as X-axis. If we pass a raw list,
            # it defaults to 0..N-1 which is confusing. Use 1..steps (horizon in months).
            try:
                import pandas as pd

                df = pd.DataFrame(
                    {
                        "Horizon (mois)": list(range(1, len(forecast) + 1)),
                        "traffic_forecast": forecast,
                    }
                ).set_index("Horizon (mois)")
                st.line_chart(df)
            except Exception:
                st.line_chart({"traffic_forecast": forecast})
            _render_details(res)


with tabs[1]:
    st.subheader("Recommandation de trajets — score de qualité")

    cities = _load_cities_from_db()
    if not cities:
        st.warning("DB non disponible: liste des villes indisponible. Saisis une ville manuellement.")
        city = st.text_input("Ville", value="")
    else:
        city = st.selectbox("Ville", cities, index=0)
    vitesse = st.number_input("Vitesse (km/h)", min_value=0.0, max_value=130.0, value=25.0, step=1.0)

    if st.button("Recommander (Objectif 2)"):
        if not str(city).strip():
            st.error("Veuillez renseigner une ville.")
            st.stop()

        payload = {"ville": str(city), "vitesse": float(vitesse)}

        res = _api_post("2", payload, timeout=120)
        if res:
            _render_result_header(res)
            score = res.get("quality_score")
            recommendation = res.get("recommendation")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Score qualité", value=_fmt_float(score, 3))
            with c2:
                rec = str(recommendation or "—").upper()
                if rec in {"GOOD", "OK"}:
                    st.success(f"Recommandation: {rec}")
                elif rec in {"BAD"}:
                    st.error(f"Recommandation: {rec}")
                else:
                    st.info(f"Recommandation: {rec}")

            outputs = res.get("outputs") or {}
            tmin = outputs.get("temps_trajet_min")
            cong = outputs.get("congestion_index")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Temps trajet", value=f"{_fmt_float(tmin, 1)} min")
            with c2:
                st.metric("Congestion", value=f"{_fmt_float(cong, 2)} / 9")

            if isinstance(tmin, (int, float)) and isinstance(cong, (int, float)):
                st.bar_chart({"vitesse": float(vitesse), "temps_trajet_min": float(tmin), "congestion_index": float(cong)})

            _render_details(res)


with tabs[2]:
    st.subheader("Classification des risques")
    col1, col2 = st.columns(2)
    with col1:
        volume = st.number_input("Volume", min_value=0.0, value=1500.0, step=50.0)
    with col2:
        taux_1000 = st.slider("Taux / 1000", min_value=0.0, max_value=5.0, value=0.8, step=0.05)

    if st.button("Classifier (Objectif 3)"):
        res = _api_post("3", {"volume": float(volume), "taux_1000": float(taux_1000)}, timeout=120)
        if res:
            _render_result_header(res)
            risk_score = res.get("risk_score")
            risk_level = res.get("risk_level")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Niveau de risque", value=str(risk_level or "—"))
            with c2:
                st.metric("Score", value=_fmt_float(risk_score, 3))

            if isinstance(risk_score, (int, float)):
                st.progress(min(max(float(risk_score), 0.0), 1.0))

            _render_details(res)


with tabs[3]:
    st.subheader("Estimation CO2 & énergie")
    col1, col2, col3 = st.columns(3)
    with col1:
        zones = _load_zones_from_db()
        if zones:
            selected = st.selectbox("Zone", zones, format_func=lambda z: z[1])
            zone_id = int(selected[0])
        else:
            zone_id = st.number_input("Zone ID", min_value=1, value=1, step=1)
    with col2:
        year = st.number_input("Année", min_value=2000, max_value=2100, value=2026, step=1)
    with col3:
        month = st.number_input("Mois", min_value=1, max_value=12, value=4, step=1)

    mode = st.selectbox("Mode", ["bus", "métro", "tram", "voiture"], index=0)
    activity_value = st.number_input("Activity value", min_value=0.0, value=120.0, step=5.0)
    emission_factor = st.number_input("Emission factor", min_value=0.0, value=0.2, step=0.05)

    if st.button("Estimer (Objectif 4)"):
        payload = {
            "zone_id": int(zone_id),
            "year": int(year),
            "month": int(month),
            "mode": str(mode),
            "activity_value": float(activity_value),
            "emission_factor": float(emission_factor),
        }
        res = _api_post("4", payload, timeout=120)
        if res:
            _render_result_header(res)
            pred = (res.get("prediction") or {})
            co2 = pred.get("co2_kg")
            energy = pred.get("energie_kwh")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("CO2", value=f"{_fmt_float(co2, 3)} kg")
            with c2:
                st.metric("Énergie", value=f"{_fmt_float(energy, 3)} kWh")

            if isinstance(co2, (int, float)) and isinstance(energy, (int, float)):
                st.bar_chart({"co2_kg": float(co2), "energie_kwh": float(energy)})
            _render_details(res)

