import os
import pickle
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from xgboost import XGBRegressor
from model import DynamicPricingModel

# ----------------------------
# Feature configuration
# ----------------------------
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

pricing_model = None
category_encoder = None


# ----------------------------
# Utilities
# ----------------------------
def load_category_encoder():
    path = os.path.join(os.path.dirname(__file__), "category_encoder.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("⚠️ Failed to load category encoder:", e)
    return None


# ----------------------------
# App lifespan (startup)
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pricing_model, category_encoder

    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "pricing_model.json")

    category_encoder = load_category_encoder()

    # Try loading XGBoost model safely
    try:
        model = XGBRegressor()
        model.load_model(model_path)
        print("✅ XGBoost model loaded successfully")

    except Exception as e:
        print("⚠️ Model load failed. Using fallback model instead.")
        print("Reason:", e)

        # Fallback model (deployment-safe)
        class FallbackModel:
            def predict(self, X):
                price = X[:, 0]
                return np.maximum(20, 150 - 0.002 * price)

        model = FallbackModel()

    pricing_model = DynamicPricingModel(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        category_encoder=category_encoder
    )

    print("🚀 Application startup complete")
    yield


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Dynamic Pricing API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Schemas
# ----------------------------
class ProductInput(BaseModel):
    product: str
    category: str
    current_price: float
    competitor_price: float
    season: str
    day: str


# ----------------------------
# Encoding logic
# ----------------------------
def encode(p: ProductInput):
    is_holiday = 1 if p.season.lower() == "holiday" else 0
    is_weekend = 1 if p.day.lower() == "weekend" else 0
    day_of_week = 6 if is_weekend else 2

    comp = p.competitor_price or p.current_price or 1.0

    try:
        if category_encoder is not None:
            cat = float(category_encoder.transform([p.category])[0])
        else:
            raise ValueError("Encoder not available")
    except Exception:
        cat = float(abs(hash(p.category)) % 50)

    return {
        "current_price": p.current_price,
        "competitor_price": comp,
        "price_difference": p.current_price - comp,
        "price_ratio": p.current_price / comp,
        "category_encoded": cat,
        "is_holiday": is_holiday,
        "is_weekend": is_weekend,
        "day_of_week": day_of_week,
        "month": 6.0,
    }


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
async def root():
    return {"status": "Dynamic Pricing API is running"}


@app.post("/predict")
async def predict(p: ProductInput):
    if pricing_model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return pricing_model.recommend_price(encode(p))


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    file.file.seek(0)
    df = pd.read_csv(file.file)

    required = ["product", "category", "current_price",
                "competitor_price", "season", "day"]

    for col in required:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")

    rows = []
    cur_total = opt_total = 0.0

    for _, r in df.iterrows():
        p = ProductInput(**r.to_dict())
        res = pricing_model.recommend_price(encode(p))

        cur_total += res["current_revenue"]
        opt_total += res["predicted_revenue"]

        rows.append({
            "product": p.product,
            "current_revenue": res["current_revenue"],
            "optimized_revenue": res["predicted_revenue"],
            "growth_percent": (
                (res["predicted_revenue"] - res["current_revenue"])
                / res["current_revenue"] * 100
                if res["current_revenue"] > 0 else 0
            )
        })

    return {
        "summary": {
            "avg_current_revenue": cur_total / len(rows),
            "avg_optimized_revenue": opt_total / len(rows),
            "growth_percent": (
                (opt_total - cur_total) / cur_total * 100
                if cur_total > 0 else 0
            ),
        },
        "rows": rows
    }
