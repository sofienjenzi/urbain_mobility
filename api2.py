import os
import sys

from flask import Flask, jsonify, request

from Objectif3_Classification_Risques_V2 import run_Classification

# Permet d'importer des modules placés dans le dossier parent (ex: C:\Users\user\Downloads)
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

try:
    import objectif1_final_v2
except Exception as _e:  # import optionnel: l'API doit pouvoir démarrer même si Objectif 1 n'est pas dispo
    objectif1_final_v2 = None
    _objectif1_import_error = _e

app = Flask(__name__)

@app.route("/classification", methods=["GET", "POST"])
def predict():
    input_data = None
    if request.method == "POST":
        if request.is_json:
            input_data = request.get_json(silent=True) or None
        else:
            # fallback: form data
            input_data = request.form.to_dict() or None

    result = run_Classification(input_data)
    return jsonify(result)


@app.route("/predict", methods=["GET"])
def predict_objectif1():
    if objectif1_final_v2 is None:
        return jsonify(
            {
                "status": "error",
                "message": "Import objectif1_final_v2 impossible",
                "details": str(_objectif1_import_error),
            }
        ), 500

    try:
        result = objectif1_final_v2.run_prediction()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/")
def home():
    return "API ML is running 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
