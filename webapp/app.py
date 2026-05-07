from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("PREDICTION_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Urbain Mobility MLOps", layout="wide")


def _asset_data_uri(name: str) -> str:
    path = Path(__file__).resolve().parent / "assets" / name
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _inject_theme() -> None:
    bg_uri = _asset_data_uri("urbain-mobility-bg.png")
    st.markdown(
        f"""
        <style>
        :root {{
            --um-teal: #2f9694;
            --um-teal-dark: #1e6f70;
            --um-ink: #1d2b35;
            --um-muted: #6c7a82;
            --um-gold: #f3a91f;
            --um-panel: rgba(255, 255, 255, 0.86);
            --um-border: rgba(47, 150, 148, 0.18);
        }}

        .stApp {{
            background:
                linear-gradient(90deg, rgba(255,255,255,0.70), rgba(255,255,255,0.88)),
                url("{bg_uri}") center center / cover fixed no-repeat;
            color: var(--um-ink);
        }}

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        footer {{
            background: transparent;
        }}

        [data-testid="stMainBlockContainer"] {{
            max-width: 1180px;
            padding: 2rem 2rem 3rem;
        }}

        .um-shell {{
            background: transparent;
            border: 0;
            border-radius: 0;
            padding: 1.2rem 1rem 1.4rem;
            box-shadow: none;
            backdrop-filter: none;
        }}

        .um-hero {{
            display: block;
            text-align: center;
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: 0;
        }}

        .um-brand {{
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--um-teal-dark);
            font-weight: 700;
            margin-bottom: 0.45rem;
            text-shadow: 0 1px 12px rgba(255,255,255,0.8);
        }}

        .um-title {{
            font-size: clamp(1.7rem, 2.4vw, 2.55rem);
            line-height: 1.04;
            margin: 0;
            color: var(--um-ink);
            font-weight: 800;
            text-shadow: 0 2px 18px rgba(255,255,255,0.9);
        }}

        .um-subtitle {{
            margin-top: 0.5rem;
            color: var(--um-muted);
            font-size: 0.95rem;
            max-width: 680px;
            margin-left: auto;
            margin-right: auto;
            text-shadow: 0 1px 12px rgba(255,255,255,0.9);
        }}

        .um-status {{
            display: none;
            color: var(--um-muted);
            font-size: 0.82rem;
            text-align: right;
            white-space: nowrap;
            background: rgba(255,255,255,0.34);
            border: 1px solid rgba(47, 150, 148, 0.14);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid rgba(47, 150, 148, 0.16);
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 44px;
            padding: 0 14px;
            border-radius: 10px 10px 0 0;
            color: var(--um-muted);
            font-weight: 650;
        }}

        .stTabs [aria-selected="true"] {{
            color: var(--um-teal-dark);
            background: rgba(47, 150, 148, 0.10);
        }}

        h2, h3 {{
            color: var(--um-ink);
            letter-spacing: 0;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(47, 150, 148, 0.14);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 12px 32px rgba(35, 98, 99, 0.08);
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--um-muted);
            font-weight: 650;
        }}

        .stButton > button {{
            border: 0;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--um-teal), var(--um-teal-dark));
            color: white;
            font-weight: 750;
            min-height: 44px;
            padding: 0.6rem 1rem;
            box-shadow: 0 14px 28px rgba(47, 150, 148, 0.22);
        }}

        .stButton > button:hover {{
            color: white;
            filter: brightness(1.04);
            transform: translateY(-1px);
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {{
            border-radius: 12px;
            border-color: rgba(47, 150, 148, 0.14);
            background-color: rgba(255,255,255,0.78);
        }}

        .stAlert {{
            border-radius: 12px;
        }}

        [data-testid="stExpander"] {{
            border-radius: 12px;
            border-color: rgba(47, 150, 148, 0.16);
            background: rgba(255,255,255,0.66);
        }}

        @media (max-width: 760px) {{
            [data-testid="stMainBlockContainer"] {{
                padding: 1rem;
            }}
            .um-shell {{
                padding: 1rem;
                border-radius: 14px;
            }}
            .um-hero {{
                align-items: flex-start;
                flex-direction: column;
            }}
            .um-status {{
                text-align: left;
                white-space: normal;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _default_db_host() -> str:
    return "host.docker.internal" if Path("/.dockerenv").exists() else "localhost"


def _db_settings() -> dict[str, Any]:
    return {
        "host": os.getenv("PGHOST") or os.getenv("DB_HOST") or _default_db_host(),
        "port": int(os.getenv("PGPORT") or os.getenv("DB_PORT") or "5432"),
        "database": os.getenv("PGDATABASE") or os.getenv("DB_NAME") or "urbain_dw",
        "user": os.getenv("PGUSER") or os.getenv("DB_USER") or "postgres",
        "password": os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD") or "admin",
        "schema": os.getenv("PGSCHEMA") or os.getenv("DB_SCHEMA") or "public",
    }


def _db_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    cfg = _db_settings()
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
    )
    return create_engine(url, connect_args={"connect_timeout": 5})


def _load_cities_from_db() -> list[str] | None:
    try:
        import pandas as pd  # local import to keep startup light
    except Exception:
        return None

    try:
        cfg = _db_settings()
        engine = _db_engine()
        db_schema = cfg["schema"]
        q = f"SELECT DISTINCT ville FROM {db_schema}.dim_zone WHERE ville IS NOT NULL ORDER BY ville;"
        df = pd.read_sql(q, engine)
        cities = [str(x) for x in df["ville"].dropna().tolist()]
        return cities or None
    except Exception:
        return None


def _load_zones_from_db() -> list[tuple[int, str]] | None:
    """Return list of (zone_id, label) sorted by label."""
    try:
        import pandas as pd  # local import

        cfg = _db_settings()
        engine = _db_engine()
        db_schema = cfg["schema"]
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


def _load_modes_from_db() -> list[str] | None:
    try:
        import pandas as pd

        cfg = _db_settings()
        engine = _db_engine()
        schema = cfg["schema"]
        q = (
            f"SELECT DISTINCT mode FROM {schema}.dim_emission_co2 WHERE mode IS NOT NULL "
            f"UNION SELECT DISTINCT mode FROM {schema}.dim_energietransport WHERE mode IS NOT NULL "
            f"ORDER BY mode;"
        )
        df = pd.read_sql(q, engine)
        modes = sorted(
            {_canonical_mode(str(x)) for x in df["mode"].dropna().tolist()},
            key=lambda x: {"bus": 0, "métro": 1, "tram": 2, "voiture": 3}.get(x, 99),
        )
        return modes or None
    except Exception:
        return None


def _canonical_mode(mode: str) -> str:
    value = str(mode or "").strip()
    key = value.lower().replace("é", "e").replace("è", "e").replace("ê", "e")
    aliases = {
        "bus": "bus",
        "metro": "métro",
        "tram": "tram",
        "voiture": "voiture",
        "car": "voiture",
    }
    return aliases.get(key, value.lower())


def _parse_powerbi_dt(value: Any):
    import pandas as pd

    if not value:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def _event_duration_ms(event: dict[str, Any]) -> float:
    start = _parse_powerbi_dt(event.get("start"))
    end = _parse_powerbi_dt(event.get("end"))
    if str(start) == "NaT" or str(end) == "NaT":
        return 0.0
    try:
        return max((end - start).total_seconds() * 1000.0, 0.0)
    except Exception:
        return 0.0


def _latest_powerbi_export_path() -> Path | None:
    downloads = Path.home() / "Downloads"
    candidates = [
        Path(os.getenv("POWERBI_EXPORT_DIR", "/app/powerbi_exports")) / "PowerBIPerformanceData.json",
        downloads / "Analyser Performance" / "PowerBIPerformanceData.json",
        downloads / "PowerBIPerformanceData.json",
    ]
    existing = [p for p in candidates if p.exists() and p.is_file()]
    if not existing:
        return None
    return max(set(existing), key=lambda p: p.stat().st_mtime)


def _load_powerbi_json(uploaded_file: Any | None, fallback_path: Path | None) -> tuple[dict[str, Any] | None, str]:
    try:
        if uploaded_file is not None:
            raw = uploaded_file.getvalue().decode("utf-8-sig")
            return json.loads(raw), uploaded_file.name
        if fallback_path and fallback_path.exists():
            return json.loads(fallback_path.read_text(encoding="utf-8-sig")), str(fallback_path)
    except Exception as exc:
        st.error(f"Lecture JSON impossible: {exc}")
        return None, ""
    return None, ""


def _powerbi_perf_frames(data: dict[str, Any]):
    import pandas as pd

    events = data.get("events") or []
    rows: list[dict[str, Any]] = []
    containers: dict[str, dict[str, Any]] = {}

    for ev in events:
        if not isinstance(ev, dict):
            continue
        metrics = ev.get("metrics") or {}
        duration = _event_duration_ms(ev)
        row = {
            "Nom": ev.get("name") or "",
            "Composant": ev.get("component") or "",
            "Debut": _parse_powerbi_dt(ev.get("start")),
            "Fin": _parse_powerbi_dt(ev.get("end")),
            "Duree_ms": round(duration, 2),
            "ParentId": ev.get("parentId") or "",
            "Id": ev.get("id") or "",
            "Visual": metrics.get("visualTitle") or "",
            "Type": metrics.get("visualType") or "",
            "Lignes": metrics.get("RowCount"),
            "DAX": metrics.get("QueryText") or "",
        }
        rows.append(row)

        if row["Nom"] == "Visual Container Lifecycle":
            containers[str(row["Id"])] = {
                "Visual": row["Visual"] or "Sans titre",
                "Type": row["Type"] or "unknown",
                "Duree totale (ms)": duration,
                "Query (ms)": 0.0,
                "Render (ms)": 0.0,
                "DAX (ms)": 0.0,
                "Data transform (ms)": 0.0,
                "Autres (ms)": 0.0,
                "Nb requetes": 0,
                "Lignes": 0,
                "Initial load": metrics.get("initialLoad"),
            }

    event_df = pd.DataFrame(rows)

    if not event_df.empty and containers:
        id_to_parent = {
            str(row.get("Id") or ""): str(row.get("ParentId") or "")
            for _, row in event_df.iterrows()
            if str(row.get("Id") or "")
        }

        def container_for(parent_id: str) -> dict[str, Any] | None:
            current = str(parent_id or "")
            seen: set[str] = set()
            while current and current not in seen:
                if current in containers:
                    return containers[current]
                seen.add(current)
                current = id_to_parent.get(current, "")
            return None

        for _, row in event_df.iterrows():
            container = container_for(str(row.get("ParentId") or ""))
            if container is None:
                continue
            name = str(row.get("Nom") or "")
            duration = float(row.get("Duree_ms") or 0.0)
            if name == "Query":
                container["Query (ms)"] += duration
                container["Nb requetes"] += 1
            elif name == "Render":
                container["Render (ms)"] += duration
            elif name == "Execute DAX Query":
                container["DAX (ms)"] += duration
            elif name == "Data View Transform":
                container["Data transform (ms)"] += duration
            elif name != "Visual Container Lifecycle":
                container["Autres (ms)"] += duration

            lines = row.get("Lignes")
            if lines not in (None, ""):
                try:
                    container["Lignes"] += int(lines)
                except Exception:
                    pass

    visual_df = pd.DataFrame(containers.values())
    if not visual_df.empty:
        numeric_cols = [
            "Duree totale (ms)",
            "Query (ms)",
            "Render (ms)",
            "DAX (ms)",
            "Data transform (ms)",
            "Autres (ms)",
        ]
        visual_df[numeric_cols] = visual_df[numeric_cols].round(2)
        visual_df["Niveau"] = pd.cut(
            visual_df["Duree totale (ms)"],
            bins=[-1, 250, 750, 1500, float("inf")],
            labels=["Rapide", "Correct", "Lent", "Critique"],
        ).astype(str)
        visual_df = visual_df.sort_values("Duree totale (ms)", ascending=False).reset_index(drop=True)

    return event_df, visual_df


def _render_powerbi_performance_dashboard() -> None:
    import pandas as pd

    st.subheader("Dashboard Performance Power BI")
    default_path = _latest_powerbi_export_path()
    uploaded = st.file_uploader("Importer l'export JSON Power BI", type=["json"])

    if default_path and uploaded is None:
        st.caption(f"Export detecte automatiquement: {default_path}")
    elif uploaded is None:
        st.info("Exporte le fichier depuis l'analyseur de performance Power BI puis importe le JSON ici.")

    data, source = _load_powerbi_json(uploaded, default_path)
    if not data:
        return

    event_df, visual_df = _powerbi_perf_frames(data)
    if event_df.empty:
        st.warning("Aucun evenement Power BI exploitable dans ce JSON.")
        return

    report_start = event_df["Debut"].min()
    report_end = event_df["Fin"].max()
    total_duration = 0.0
    if pd.notna(report_start) and pd.notna(report_end):
        total_duration = max((report_end - report_start).total_seconds(), 0.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Visuels analyses", len(visual_df))
    with c2:
        st.metric("Evenements", len(event_df))
    with c3:
        slow_count = int((visual_df["Duree totale (ms)"] > 750).sum()) if not visual_df.empty else 0
        st.metric("Visuels lents", slow_count)
    with c4:
        st.metric("Session", f"{total_duration:.1f} s")

    st.caption(f"Source: {source} | Session ID: {data.get('sessionId', '-')}")

    if visual_df.empty:
        st.warning("Le JSON contient des evenements, mais aucun cycle de vie de visuel n'a ete trouve.")
        st.dataframe(event_df, use_container_width=True, hide_index=True)
        return

    left, right = st.columns([1.25, 1])
    with left:
        st.bar_chart(visual_df.head(12).set_index("Visual")[["Duree totale (ms)"]])
    with right:
        by_type = visual_df.groupby("Type", as_index=True)["Duree totale (ms)"].mean().sort_values(ascending=False)
        st.bar_chart(by_type)

    st.markdown("#### Details par visuel")
    selected_level = st.multiselect(
        "Filtrer par niveau",
        ["Rapide", "Correct", "Lent", "Critique"],
        default=["Rapide", "Correct", "Lent", "Critique"],
    )
    filtered = visual_df[visual_df["Niveau"].isin(selected_level)] if selected_level else visual_df
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    worst = visual_df.iloc[0]
    st.info(
        "Visuel a optimiser en premier: "
        f"{worst['Visual']} ({worst['Type']}) avec {worst['Duree totale (ms)']:.0f} ms."
    )

    dax_events = event_df[event_df["DAX"].astype(str).str.len() > 0].copy()
    if not dax_events.empty:
        st.markdown("#### Requetes DAX les plus couteuses")
        dax_events = dax_events.sort_values("Duree_ms", ascending=False)
        st.dataframe(
            dax_events[["Nom", "Duree_ms", "Lignes", "DAX"]].head(20),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Evenements bruts", expanded=False):
        st.dataframe(event_df, use_container_width=True, hide_index=True)


_inject_theme()

st.markdown(
    f"""
    <section class="um-shell">
        <div class="um-hero">
            <div>
                <div class="um-brand">Urbain Mobility</div>
                <h1 class="um-title">Mobility Intelligence Hub</h1>
                <div class="um-subtitle">Prévisions, recommandations et indicateurs opérationnels pour la mobilité urbaine.</div>
            </div>
            <div class="um-status">API connectée<br><strong>{API_URL}</strong></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Power BI - Performance",
        "Objectif 1 — Trafic (forecast)",
        "Objectif 2 — Recommandation (trajets)",
        "Objectif 3 — Risques (classification)",
        "Objectif 4 — CO2 & Énergie (estimation)",
    ]
)


with tabs[0]:
    _render_powerbi_performance_dashboard()


with tabs[1]:
    st.subheader("Prévision du trafic")
    steps = st.slider("Nombre de mois à prédire", min_value=1, max_value=36, value=12, step=1)
    if st.button("Prédire (Objectif 1)"):
        res = _api_post("1", {"steps": int(steps)}, timeout=120)
        if res:
            forecast = res.get("forecast") or []
            _render_result_header(res)

            if forecast:
                summary = res.get("traffic_summary") or {}
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Horizon", value=f"{len(forecast)} mois")
                with c2:
                    st.metric("Niveau moyen", value=str(summary.get("average_level") or "—"))
                with c3:
                    st.metric("Mois le plus chargé", value=str(summary.get("peak_month") or "—"))

                avg_desc = summary.get("average_description")
                if avg_desc:
                    st.info(str(avg_desc))

            # Streamlit uses the DataFrame index as X-axis. If we pass a raw list,
            # it defaults to 0..N-1 which is confusing. Use 1..steps (horizon in months).
            try:
                import pandas as pd

                classified = res.get("traffic_classification") or []
                df = pd.DataFrame(
                    {
                        "Mois": [
                            row.get("month", f"M+{i + 1}") if isinstance(row, dict) else f"M+{i + 1}"
                            for i, row in enumerate(classified or forecast)
                        ],
                        "Indice congestion": forecast,
                    }
                ).set_index("Mois")
                st.line_chart(df)
                if classified:
                    table_df = pd.DataFrame(classified)
                    table_df = table_df.rename(
                        columns={
                            "month": "Mois",
                            "value": "Indice",
                            "level": "Niveau",
                            "description": "Signification",
                        }
                    )[["Mois", "Indice", "Niveau", "Signification"]]
                    table_df["Indice"] = table_df["Indice"].map(lambda x: round(float(x), 2))
                    st.dataframe(table_df, use_container_width=True, hide_index=True)
            except Exception:
                st.line_chart({"Indice congestion": forecast})
            _render_details(res)


with tabs[2]:
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
                st.metric("Score qualité", value=f"{_fmt_float(score, 1)} / 100")
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


with tabs[3]:
    st.subheader("Classification des risques")
    zones = _load_zones_from_db()
    if zones:
        selected_risk_zone = st.selectbox("Zone", zones, format_func=lambda z: z[1], key="risk_zone")
        risk_zone_id = int(selected_risk_zone[0])
    else:
        st.warning("DB non disponible: sélection de zone indisponible. Saisie manuelle activée.")
        risk_zone_id = st.number_input("Zone ID", min_value=1, value=1, step=1, key="risk_zone_manual")

    col1, col2 = st.columns(2)
    with col1:
        risk_year = st.number_input("Année", min_value=2000, max_value=2100, value=2026, step=1, key="risk_year")
    with col2:
        risk_month = st.number_input("Mois", min_value=1, max_value=12, value=4, step=1, key="risk_month")

    if st.button("Classifier (Objectif 3)"):
        res = _api_post("3", {"zone_id": int(risk_zone_id), "year": int(risk_year), "month": int(risk_month)}, timeout=120)
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

            derived = res.get("derived_from_db")
            if derived:
                st.caption("Features dérivées depuis PostgreSQL urbain_dw")
                st.json(derived)

            _render_details(res)


with tabs[4]:
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

    modes = _load_modes_from_db() or ["bus", "métro", "tram", "voiture"]
    mode = st.selectbox("Mode", modes, index=0)
    st.caption("Activity value, emission factor, déchets et station kWh sont dérivés depuis PostgreSQL urbain_dw.")

    if st.button("Estimer (Objectif 4)"):
        payload = {
            "zone_id": int(zone_id),
            "year": int(year),
            "month": int(month),
            "mode": _canonical_mode(str(mode)),
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
            derived = res.get("derived_from_db")
            if derived:
                st.caption("Features dérivées depuis PostgreSQL urbain_dw")
                st.json(derived)
            _render_details(res)



