import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
import joblib
import os

class EnsembleStackingModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Base estimators
        self.rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=self.random_state)
        
        # Stacking Regressor
        estimators = [
            ('rf', self.rf),
            ('xgb', self.xgb)
        ]
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(),
            cv=5
        )
        
        self.X_train = None
        self.y_train = None
        self.feature_cols = None

    def fit(self, X, y, feature_cols=None, alpha_arr=None, sample_weights=None):
        """
        Fits the ensemble model with optional sample weights.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_cols: Feature column names
            alpha_arr: Legacy parameter for GP compatibility (unused)
            sample_weights: Per-sample weights (e.g., for Lab vs Literature data)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.feature_cols = feature_cols
        
        # Apply sample weights if provided
        if sample_weights is not None:
            sample_weights = np.array(sample_weights)
            print(f"Training with sample weights (min={sample_weights.min():.1f}, max={sample_weights.max():.1f})")
            self.model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        else:
            self.model.fit(self.X_train, self.y_train)
        
        print(f"Ensemble Stacking Model fitted with {len(self.X_train)} samples.")

    def predict(self, X, return_std=False):
        X_array = np.array(X)
        y_pred = self.model.predict(X_array)
        
        if return_std:
            # Heuristic: Use the disagreement between base models as uncertainty
            # We need to access the trained base models within the stacking regressor
            preds = []
            for name, est in self.model.named_estimators_.items():
                preds.append(est.predict(X_array))
            
            # Standard deviation across base model predictions
            std = np.std(preds, axis=0) + 1e-2 # Add small floor for EI calculation
            return y_pred, std
        
        return y_pred

    def save(self, path):
        joblib.dump({
            'model': self.model,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'feature_cols': self.feature_cols
        }, path)
        print(f"Ensemble model saved to {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.X_train = data['X_train']
        instance.y_train = data['y_train']
        instance.feature_cols = data['feature_cols']
        return instance
