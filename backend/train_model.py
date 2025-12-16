import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("../data/sample_dataset.csv")

# Encode category
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"])

# Feature engineering
df["price_difference"] = df["current_price"] - df["competitor_price"]
df["price_ratio"] = df["current_price"] / df["competitor_price"]
df["is_holiday"] = df["season"].apply(lambda x: 1 if x.lower() == "holiday" else 0)
df["is_weekend"] = df["day"].apply(lambda x: 1 if x.lower() == "weekend" else 0)
df["day_of_week"] = df["is_weekend"].apply(lambda x: 6 if x == 1 else 2)
df["month"] = 6

FEATURE_COLUMNS = [
    "current_price",
    "competitor_price",
    "price_difference",
    "price_ratio",
    "category_encoded",
    "is_holiday",
    "is_weekend",
    "day_of_week",
    "month",
]

X = df[FEATURE_COLUMNS]
y = np.maximum(20, 150 - 0.002 * df["current_price"])

# Train model
model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    objective="reg:squarederror"
)

model.fit(X, y)

# ✅ SAVE MODEL SAFELY (NO PICKLE)
model.save_model("pricing_model.json")

# Save encoder (sklearn pickle is OK)
import pickle
with open("category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)

print("Model saved as pricing_model.json")
