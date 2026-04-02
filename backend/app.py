import os
import joblib
from flask import Flask, request, jsonify
from utils import validate_input, build_dataframe, interpret_prediction, rank_planets

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "ExoHabitAI",
        "description": "Exoplanet Habitability Prediction API",
        "model": type(model).__name__,
        "endpoints": {
            "POST /predict": "Predict habitability for a single exoplanet",
            "POST /rank":    "Rank a list of exoplanets by habitability score",
            "GET  /features":"List required input features with valid ranges",
            "GET  /health":  "Health check",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
    })


@app.route("/features", methods=["GET"])
def features():
    from utils import REQUIRED_FEATURES, FEATURE_RANGES
    return jsonify({
        "status": "success",
        "total_features": len(REQUIRED_FEATURES),
        "features": {
            f: {"min": FEATURE_RANGES[f][0], "max": FEATURE_RANGES[f][1], "unit": FEATURE_RANGES[f][2]}
            for f in REQUIRED_FEATURES
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON."}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Empty or malformed JSON body."}), 400

    cleaned, err = validate_input(data)
    if err:
        return jsonify({"status": "error", "message": err}), 422

    df = build_dataframe(cleaned)
    label = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])

    return jsonify({
        "status": "success",
        "prediction": interpret_prediction(label, proba),
    })


@app.route("/rank", methods=["POST"])
def rank():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON."}), 400

    data = request.get_json(silent=True)
    if not data or "planets" not in data:
        return jsonify({"status": "error", "message": "Request body must contain a 'planets' list."}), 400

    planets = data["planets"]
    if not isinstance(planets, list) or len(planets) == 0:
        return jsonify({"status": "error", "message": "'planets' must be a non-empty list."}), 422

    ranked = rank_planets(planets, model)

    return jsonify({
        "status": "success",
        "count": len(ranked),
        "ranked_planets": ranked,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)