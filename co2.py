"""
API FLASK — CO2 / QUALITÉ DE L'AIR REPORT
Route: GET /co2report
Port: 5002
Retourne: JSON avec KPIs qualité de l'air
"""

from flask import Flask, jsonify
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============================
# CONFIG POSTGRESQL
# ============================
os.environ.setdefault("DB_HOST",     "localhost")
os.environ.setdefault("DB_PORT",     "5432")
os.environ.setdefault("DB_NAME",     "urbain_dw")
os.environ.setdefault("DB_USER",     "postgres")
os.environ.setdefault("DB_PASSWORD", "admin")
os.environ.setdefault("DB_SCHEMA",   "public")

_ENGINE = None

def _get_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME"),
    )
    return create_engine(url, pool_pre_ping=True)

def _read_sql(sql):
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

# ============================
# FONCTION PRINCIPALE
# ============================
def run_co2_report():
    schema = os.getenv("DB_SCHEMA", "public")

    # --- Chargement données ---
    sql = f"""
        SELECT 
            fp.pm_25,
            fp.no2,
            fp.aqi_index,
            fp.fk_zone,
            dt.date,
            dt.mois,
            dt.annee,
            dt.periode,
            dz.zone_id
        FROM "{schema}"."fact_pollution" fp
        LEFT JOIN "{schema}"."dim_zone" dz  ON fp.fk_zone = dz.zone_id
        LEFT JOIN "{schema}"."dim_time" dt  ON fp.fk_zone = dt.time_id
        ORDER BY dt.date DESC
        LIMIT 1000;
    """
    df = _read_sql(sql)

    if len(df) == 0:
        raise ValueError("Aucune donnée dans fact_pollution")

    # --- Calcul KPIs ---
    count = len(df)

    df['pm_25']     = pd.to_numeric(df['pm_25'],     errors='coerce').fillna(0)
    df['no2']       = pd.to_numeric(df['no2'],       errors='coerce').fillna(0)
    df['aqi_index'] = pd.to_numeric(df['aqi_index'], errors='coerce').fillna(0)

    avg_pm25 = round(float(df['pm_25'].mean()), 2)
    avg_no2  = round(float(df['no2'].mean()),   2)
    avg_aqi  = round(float(df['aqi_index'].mean()), 2)

    high_pollution = int((df['aqi_index'] > 100).sum())

    # Niveau global
    if avg_aqi > 150:      level = "Dangereux"
    elif avg_aqi > 100:    level = "Mauvais"
    elif avg_aqi > 50:     level = "Modéré"
    else:                  level = "Bon"

    # Couleur selon niveau
    couleur = {
        "Dangereux": "#e74c3c",
        "Mauvais":   "#e67e22",
        "Modéré":    "#f1c40f",
        "Bon":       "#27ae60",
    }.get(level, "#27ae60")

    # --- Génération HTML ---
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f6f9; padding: 20px; }}
    .container {{ max-width: 600px; margin: auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
    .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; text-align: center; }}
    .header h1 {{ margin: 0; font-size: 24px; }}
    .header p {{ margin: 5px 0 0; opacity: 0.8; }}
    .level-badge {{ display: inline-block; background: {couleur}; color: white; padding: 8px 20px; border-radius: 20px; font-size: 18px; font-weight: bold; margin: 20px auto; }}
    .content {{ padding: 25px; }}
    .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
    .kpi-card {{ background: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; border-left: 4px solid #3498db; }}
    .kpi-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
    .kpi-card .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; text-transform: uppercase; }}
    .alert-card {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 10px; padding: 15px; margin: 15px 0; }}
    .footer {{ background: #f8f9fa; padding: 15px; text-align: center; color: #7f8c8d; font-size: 12px; border-top: 1px solid #eee; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🌍 Rapport Qualité de l'Air</h1>
      <p>Rapport automatique — {pd.Timestamp.now().strftime('%d/%m/%Y')}</p>
    </div>
    <div class="content">
      <div style="text-align:center;">
        <div class="level-badge">Niveau : {level}</div>
      </div>
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="value">{avg_pm25}</div>
          <div class="label">PM2.5 moyen (µg/m³)</div>
        </div>
        <div class="kpi-card">
          <div class="value">{avg_no2}</div>
          <div class="label">NO2 moyen (µg/m³)</div>
        </div>
        <div class="kpi-card">
          <div class="value" style="color:{couleur}">{avg_aqi}</div>
          <div class="label">Indice AQI moyen</div>
        </div>
        <div class="kpi-card">
          <div class="value">{count}</div>
          <div class="label">Mesures analysées</div>
        </div>
      </div>
      <div class="alert-card">
        ⚠️ <strong>{high_pollution} mesures</strong> avec un AQI supérieur à 100 détectées
      </div>
    </div>
    <div class="footer">
      Généré automatiquement par n8n • {pd.Timestamp.now().strftime('%d/%m/%Y')}
    </div>
  </div>
</body>
</html>
"""

    return {
        "status":            "success",
        "run_date":          str(pd.Timestamp.now().date()),
        "level":             level,
        "avgPM25":           avg_pm25,
        "avgNO2":            avg_no2,
        "avgAQI":            avg_aqi,
        "highPollutionDays": high_pollution,
        "totalRows":         count,
        "html":              html,
    }

# ============================
# ROUTES FLASK
# ============================
@app.route("/co2report", methods=["GET"])
def co2report():
    try:
        result = run_co2_report()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "co2_report"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)