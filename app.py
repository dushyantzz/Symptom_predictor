from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load the trained model
with open("Symptom_detector.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Define symptoms and weights mapping (based on your dataset)
SYMPTOM_WEIGHTS = {
    "fatigue": 1, "yellowish_skin": 2, "loss_of_appetite": 3, "yellowing_of_eyes": 4,
    "family_history": 5, "stomach_pain": 6, "ulcers_on_tongue": 7, "vomiting": 8,
    "cough": 9, "chest_pain": 10, "itching": 11, "skin_rash": 12, "nodal_skin_eruptions": 13,
    "continuous_sneezing": 14, "shivering": 15, "chills": 16, "joint_pain": 17
}


@app.route("/predict", methods=["POST"])
def predict():
    """
    API Endpoint: Predict disease based on symptoms
    Request Body: JSON {"symptoms": ["symptom1", "symptom2", ...]}
    """
    data = request.json

    # Validate input
    symptoms = data.get("symptoms", [])
    if not symptoms or not isinstance(symptoms, list):
        return jsonify({"error": "Invalid input. Provide a list of symptoms."}), 400

    # Convert symptoms to weights
    symptom_weights = [SYMPTOM_WEIGHTS.get(s, 0) for s in symptoms]

    # Pad to ensure 17 features (model input size)
    if len(symptom_weights) < 17:
        symptom_weights += [0] * (17 - len(symptom_weights))  # Add zeros for missing features
    elif len(symptom_weights) > 17:
        symptom_weights = symptom_weights[:17]  # Trim excess features if more than 17

    # Predict using the model
    try:
        prediction = model.predict([symptom_weights])  # Input must be a 2D array
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
