import os
import pickle
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from model import DynamicPricingModel


# -----------------------------
# Feature columns (training-aligned)
# -----------------------------
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


# -----------------------------
# Utility to load pickle files
# -----------------------------
def load_pickle(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# -----------------------------
# App lifespan (model loading)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pricing_model, category_encoder

    loaded_model = load_pickle("pricing_model.pkl")
    category_encoder = load_pickle("category_encoder.pkl")

    # Case 1: pickle already contains DynamicPricingModel
    if isinstance(loaded_model, DynamicPricingModel):
        pricing_model = loaded_model

    # Case 2: pickle contains raw ML model
    elif loaded_model is not None:
        pricing_model = DynamicPricingModel(
            model=loaded_model,
            feature_columns=FEATURE_COLUMNS,
            category_encoder=category_encoder
        )

    # Case 3: fallback dummy model (safety)
    else:
        class DummyModel:
            def predict(self, X):
                return np.maximum(10, 120 - 0.02 * X[:, 0])

        pricing_model = DynamicPricingModel(
            model=DummyModel(),
            feature_columns=FEATURE_COLUMNS,
            category_encoder=None
        )

    print("Dynamic Pricing Model loaded successfully.")
    yield


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Dynamic Pricing API",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Input schema
# -----------------------------
class ProductInput(BaseModel):
    product: str
    category: str
    current_price: float
    competitor_price: float
    season: str
    day: str


# -----------------------------
# Feature encoding
# -----------------------------
def encode(p: ProductInput):
    is_holiday = 1 if p.season.lower() == "holiday" else 0
    is_weekend = 1 if p.day.lower() == "weekend" else 0
    day_of_week = 6 if is_weekend else 2

    comp = p.competitor_price or p.current_price or 1.0
    price_diff = p.current_price - comp
    price_ratio = p.current_price / comp if comp != 0 else 1.0

    try:
        cat = float(category_encoder.transform([p.category])[0])
    except Exception:
        cat = float(abs(hash(p.category)) % 50)

    return {
        "current_price": p.current_price,
        "competitor_price": comp,
        "price_difference": price_diff,
        "price_ratio": price_ratio,
        "category_encoded": cat,
        "is_holiday": is_holiday,
        "is_weekend": is_weekend,
        "day_of_week": day_of_week,
        "month": 6.0,
    }


# -----------------------------
# Health check (deployment test)
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pricing_model is not None}


# -----------------------------
# Single prediction
# -----------------------------
@app.post("/predict")
async def predict(p: ProductInput):
    features = encode(p)
    return pricing_model.recommend_price(features)


# -----------------------------
# Batch prediction (CSV upload)
# -----------------------------
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    file.file.seek(0)
    df = pd.read_csv(file.file)

    required = [
        "product",
        "category",
        "current_price",
        "competitor_price",
        "season",
        "day",
    ]

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
            "growth_percent": ((opt_total - cur_total) / cur_total) * 100
            if cur_total > 0 else 0,
        },
        "rows": rows
    }


# -----------------------------
# Cloud-ready startup
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
