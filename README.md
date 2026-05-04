# 🧠 Stroke Prediction using Machine Learning (XGBoost + Flask)

This project is a Machine Learning web application that predicts the risk of stroke based on user health parameters. It uses an XGBoost classifier and is deployed using Flask.

---

## 🚀 Features

- User-friendly web interface
- Predicts stroke risk (Yes/No)
- Displays prediction probability
- Handles real-time user input
- Built using Flask

---

## 🧠 Model Details

- Algorithm: XGBoost Classifier
- Problem Type: Binary Classification
- Dataset: Stroke Prediction Dataset
- Features Used:
  - Age
  - Gender
  - Hypertension
  - Heart Disease
  - Avg Glucose Level
  - BMI
  - Smoking Status

---

## 🛠️ Tech Stack

- Python  
- XGBoost  
- Pandas / NumPy  
- Flask  
- HTML / CSS  

---

## 📂 Project Structure
stroke-prediction/
│
├── app.py # Flask application
├── train_model.py # Model training
├── requirements.txt
│
├── data/
│ └── stroke_data.csv # Dataset
│
├── model/
│ └── xgb_model.pkl # Trained model
│
├── templates/
│ ├── index.html
│ ├── result.html
│ └── error.html
│
├── static/
│ ├── style.css
│ └── script.js


---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt

2️⃣ Train the model
python train_model.py

3️⃣ Run Flask app
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000/
