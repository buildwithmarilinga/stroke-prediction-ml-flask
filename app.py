print("Starting app...")

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

print("Loading model...")

# Safe model loading
try:
    model = pickle.load(open("model/xgb_model.pkl", "rb"))
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

# Mappings
GENDER_MAP = {'Male': 1, 'Female': 0, 'Other': 2}
SMOKING_MAP = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Predict route (handles both GET & POST)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    try:
        data = request.form

        # 🔥 Validate inputs (avoid empty errors)
        required_fields = ["gender", "age", "hypertension", "heart_disease",
                           "avg_glucose_level", "bmi", "smoking_status"]

        for field in required_fields:
            if field not in data or data[field] == "":
                return render_template("error.html", error=f"{field} is missing")

        # Convert input
        input_data = {
            "gender": GENDER_MAP[data["gender"]],
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": SMOKING_MAP[data["smoking_status"]],
        }

        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = "⚠️ Stroke Risk Detected" if prediction == 1 else "✅ No Stroke Risk"

        return render_template(
            "result.html",
            prediction=result,
            prob=round(probability * 100, 2)
        )

    except Exception as e:
        return render_template("error.html", error=str(e))


if __name__ == "__main__":
    print("Running Flask...")
    app.run(debug=True)