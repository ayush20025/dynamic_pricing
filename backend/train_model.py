# backend/train_model.py
"""
Train an XGBoost demand model with a short hyperparameter search and save a
DynamicPricingModel wrapper (pricing_model.pkl + category_encoder.pkl).
Run:
    cd backend
    python train_model.py
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from model import DynamicPricingModel

warnings.filterwarnings("ignore")
np.random.seed(42)


def load_and_preprocess_data(csv_path):
    """Load and preprocess the retail dataset (expects columns used in code)."""
    print("Loading dataset:", csv_path)
    df = pd.read_csv(csv_path)

    # Basic feature engineering
    df['price_difference'] = df['current_price'] - df['competitor_price']
    # avoid division by zero
    df['price_ratio'] = df['current_price'] / df['competitor_price'].replace(0, 1.0)

    # Encode category
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'].astype(str))

    # Ensure necessary columns exist and fill NA
    required = ['current_price', 'competitor_price', 'price_difference', 'price_ratio',
                'category_encoded', 'is_holiday', 'is_weekend', 'day_of_week', 'month', 'demand']
    for c in required:
        if c not in df.columns:
            df[c] = 0.0
    df[required] = df[required].fillna(0.0)
    return df, le_category


def prepare_features(df):
    feature_columns = [
        'current_price', 'competitor_price', 'price_difference', 'price_ratio',
        'category_encoded', 'is_holiday', 'is_weekend', 'day_of_week', 'month'
    ]
    X = df[feature_columns].astype(float)
    y = df['demand'].astype(float).fillna(0.0)
    return X, y, feature_columns


def train_xgb_with_tuning(X, y):
    """
    Runs a compact GridSearchCV over XGBoost hyperparameters and returns best model.
    The grid is intentionally small to keep runtime reasonable while exploring key axes.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    xgb = XGBRegressor(objective='reg:squarederror', tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.6, 0.8]
    }

    gs = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    print("Starting GridSearchCV for XGBoost (compact grid)...")
    gs.fit(X_train, y_train)

    print("Grid search completed.")
    print("Best params:", gs.best_params_)
    best_model = gs.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE: {mae:.3f}, Test R2: {r2:.4f}")

    return best_model, (mae, r2), (X_train, X_test, y_train, y_test)


def save_feature_importance_plot(model, feature_columns, out_path):
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = sorted(zip(feature_columns, importances), key=lambda x: -x[1])
            names = [x[0] for x in feat_imp]
            vals = [x[1] for x in feat_imp]
            plt.figure(figsize=(8, 4))
            plt.barh(range(len(names)), vals[::-1])
            plt.yticks(range(len(names)), names[::-1])
            plt.title("Feature importances (XGBoost)")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print("Saved feature importance plot to:", out_path)
        else:
            print("Model has no feature_importances_.")
    except Exception as e:
        print("Failed to save feature importance plot:", e)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv = os.path.join(base_dir, '..', 'data', 'retail_data_sample.csv')
    data_csv = os.path.abspath(data_csv)

    df, label_encoder = load_and_preprocess_data(data_csv)
    X, y, feature_columns = prepare_features(df)

    # Run hyperparameter search and train best model
    model, metrics, splits = train_xgb_with_tuning(X, y)
    mae, r2 = metrics

    # Optionally retrain best model on full dataset (recommended)
    print("Retraining best model on full dataset...")
    model.fit(X, y)

    # Save artifacts
    pricing_model = DynamicPricingModel(model, feature_columns, category_encoder=label_encoder)

    model_path = os.path.join(base_dir, 'pricing_model.pkl')
    encoder_path = os.path.join(base_dir, 'category_encoder.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pricing_model, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Saved pricing_model.pkl and category_encoder.pkl to", base_dir)
    print("Finished training. MAE: {:.3f}, R2: {:.4f}".format(mae, r2))

    # Feature importance plot
    fig_path = os.path.join(base_dir, "xgb_feature_importances.png")
    save_feature_importance_plot(model, feature_columns, fig_path)


if __name__ == "__main__":
    main()
