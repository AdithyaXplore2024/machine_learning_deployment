from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Create Flask app
app = Flask(__name__)

print("Flask app initialized successfully.")

# Load trained model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        input_df = pd.DataFrame([data])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_no_diabetes": float(prediction_proba[0][0]),
            "probability_diabetes": float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/")
def home():
    return "Diabetes Prediction API Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
