import argparse
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# ---------------------------
# Parse command-line arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Heart Disease Prediction Flask API")
parser.add_argument("--model", required=True, help="Path to trained model file (joblib)")
parser.add_argument("--metadata", required=True, help="Path to metadata file (json)")
args = parser.parse_args()

# ---------------------------
# Load model & metadata
# ---------------------------
try:
    logger.info(f"Loading model from {args.model}")
    model = joblib.load(args.model)

    logger.info(f"Loading metadata from {args.metadata}")
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names", [])
    logger.info(f"Loaded features: {feature_names}")

except Exception as e:
    logger.error(f"❌ Failed to load model or metadata: {e}")
    raise e

# ---------------------------
# Initialize Flask
# ---------------------------
app = Flask(__name__)
CORS(app)  # Allow API to be called from frontend apps (Streamlit, React, etc.)

@app.route("/")
def home():
    return jsonify({
        "message": "✅ Heart Disease Prediction API is running!",
        "available_endpoints": ["/predict"],
        "required_features": feature_names
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"Incoming request data: {data}")

        # Validate input
        input_features = []
        missing = []
        for feature in feature_names:
            if feature not in data:
                missing.append(feature)
            else:
                input_features.append(data[feature])

        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing features: {missing}"
            }), 400

        # Convert to numpy array
        input_array = np.array(input_features).reshape(1, -1)

        # Run prediction
        prediction = int(model.predict(input_array)[0])
        probability = float(model.predict_proba(input_array)[0][1])

        risk_label = "High" if prediction == 1 else "Low"

        response = {
            "status": "success",
            "input": data,
            "prediction": prediction,
            "probability": probability,
            "risk": risk_label
        }

        logger.info(f"Prediction result: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Error during prediction: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------
# Run Flask server
# ---------------------------
if __name__ == "__main__":
    print("\n🚀 Flask API is starting on http://127.0.0.1:5000 ...")
    print("👉 Test with: curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{...}'")
    app.run(host="0.0.0.0", port=5000, debug=True)


