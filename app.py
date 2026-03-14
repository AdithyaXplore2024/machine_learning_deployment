from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

print("Flask app initialized successfully.")


# Define the filename for the exported model

# Load trained model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json(force=True)

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_no_diabetes": float(prediction_proba[0][0]),
            "probability_diabetes": float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)

print("'/predict' endpoint defined successfully.")
