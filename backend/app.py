import os
import joblib
from flask import Flask, request, jsonify

# Import safely (in case utils missing)
try:
    from utils import validate_input, build_dataframe, interpret_prediction, rank_planets
    from utils import REQUIRED_FEATURES, FEATURE_RANGES
except Exception as e:
    print("⚠️ Utils import error:", e)
    validate_input = build_dataframe = interpret_prediction = rank_planets = None
    REQUIRED_FEATURES = FEATURE_RANGES = {}

app = Flask(__name__)

# ✅ Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")

# ✅ Safe model loading (important)
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", e)
else:
    print("⚠️ Model file not found at:", MODEL_PATH)


# ---------------- ROUTES ---------------- #

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "ExoHabitAI",
        "status": "running",
        "model_loaded": model is not None
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
    })


@app.route("/features", methods=["GET"])
def features():
    if not REQUIRED_FEATURES:
        return jsonify({"status": "error", "message": "Features not available"}), 500

    return jsonify({
        "status": "success",
        "total_features": len(REQUIRED_FEATURES),
        "features": {
            f: {
                "min": FEATURE_RANGES[f][0],
                "max": FEATURE_RANGES[f][1],
                "unit": FEATURE_RANGES[f][2]
            }
            for f in REQUIRED_FEATURES
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Empty JSON"}), 400

    if validate_input is None:
        return jsonify({"status": "error", "message": "Utils not loaded"}), 500

    cleaned, err = validate_input(data)
    if err:
        return jsonify({"status": "error", "message": err}), 422

    df = build_dataframe(cleaned)

    try:
        label = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({
        "status": "success",
        "prediction": interpret_prediction(label, proba),
    })


@app.route("/rank", methods=["POST"])
def rank():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json(silent=True)
    if not data or "planets" not in data:
        return jsonify({"status": "error", "message": "Missing 'planets' list"}), 400

    planets = data["planets"]
    if not isinstance(planets, list) or len(planets) == 0:
        return jsonify({"status": "error", "message": "Invalid planets list"}), 422

    try:
        ranked = rank_planets(planets, model)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({
        "status": "success",
        "count": len(ranked),
        "ranked_planets": ranked,
    })


# ✅ IMPORTANT: Render port binding fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)