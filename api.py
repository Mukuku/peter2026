
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "hypertension_model_v2.pkl"

# Load the existing trained model
model = joblib.load(MODEL_PATH)

FEATURES = [
    "bmi",
    "family_history",
    "smoking_status",
    "stress_score",
    "bp_history"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "Hypertension XGBoost Prediction API"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "JSON input is required"
            }), 400

        missing = [
            feature for feature in FEATURES
            if feature not in data or data[feature] is None
        ]

        if missing:
            return jsonify({
                "error": "Missing required model inputs",
                "missing_features": missing
            }), 400

        # Preserve the feature names and ordering
        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = int(model.predict(input_data)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        result = (
            "HYPERTENSIVE"
            if prediction == 1
            else "NOT HYPERTENSIVE"
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "classification": result,
            "hypertension_probability": probability,
            "model": "hypertension_model_v2"
        })

    except Exception as error:
        app.logger.exception("Prediction error")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
