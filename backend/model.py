# backend/model.py
import numpy as np

class DynamicPricingModel:
    def __init__(self, model, feature_columns, category_encoder=None):
        self.model = model
        self.feature_columns = feature_columns
        self.category_encoder = category_encoder

    def _to_array(self, features: dict):
        return np.array([[features[col] for col in self.feature_columns]])

    def _predict_demand(self, X):
        """
        Handles both raw ML models and accidental nested wrappers
        """
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        elif hasattr(self.model, "model"):
            return self.model.model.predict(X)
        else:
            raise RuntimeError("Invalid model object loaded")

    def recommend_price(self, features: dict):
        X = self._to_array(features)

        current_price = features["current_price"]
        current_demand = float(self._predict_demand(X)[0])
        current_revenue = current_price * current_demand

        price_range = np.linspace(0.7 * current_price, 1.3 * current_price, 20)

        best_price = current_price
        best_revenue = current_revenue
        best_demand = current_demand

        for p in price_range:
            features["current_price"] = p
            features["price_difference"] = p - features["competitor_price"]
            features["price_ratio"] = p / features["competitor_price"]

            demand = float(self._predict_demand(self._to_array(features))[0])
            revenue = p * demand

            if revenue > best_revenue:
                best_revenue = revenue
                best_price = p
                best_demand = demand

        return {
            "current_price": current_price,
            "recommended_price": best_price,
            "current_demand": current_demand,
            "predicted_demand_at_recommended_price": best_demand,
            "current_revenue": current_revenue,
            "predicted_revenue": best_revenue,
            "revenue_increase": best_revenue - current_revenue
        }
