# Backend API Documentation

## FastAPI Dynamic Pricing API

This directory contains the backend API server for the Dynamic Pricing Model.

### Files:
- `main.py` - FastAPI application with prediction endpoints
- `train_model.py` - Model training script
- `pricing_model.pkl` - Trained Random Forest model
- `category_encoder.pkl` - Category label encoder
- `requirements.txt` - Python dependencies

### Setup:
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python train_model.py`
3. Start server: `uvicorn main:app --reload`

### API Endpoints:
- GET `/` - Root endpoint
- GET `/health` - Health check
- POST `/predict` - Price prediction
- GET `/docs` - Interactive API documentation
