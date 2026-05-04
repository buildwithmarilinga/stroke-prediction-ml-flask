import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import os

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/stroke_data.csv")

# Drop ID column if exists
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# Handle missing values
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Convert categorical to numeric
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})

df['smoking_status'] = df['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'Unknown': 3
})

# 🔥 IMPORTANT: Drop extra columns (to match Flask input)
columns_to_drop = ['ever_married', 'work_type', 'Residence_type']
for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Final check (VERY IMPORTANT)
print("\nFinal columns used for training:")
print(df.columns)

# Split data
X = df.drop("stroke", axis=1)
y = df["stroke"]

print("\nTraining model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
pickle.dump(model, open("model/xgb_model.pkl", "wb"))

print("\n✅ Model saved successfully!")