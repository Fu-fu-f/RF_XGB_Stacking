# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from . import config
from .ensemble_model import EnsembleStackingModel

class ModelTrainer:
    def __init__(self, data_path=None, models_dir='trained_models'):
        self.data_path = data_path or config.FINAL_FILE
        self.models_dir = models_dir
        self.df = None
        self.feature_cols = []
        self.targets = ['viability']
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_and_prepare_data(self):
        print(f"Loading training data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # 1. Identify chemical features
        meta_cols = ['original_ingredients', 'viability', 'cooling_rate', 'source']
        chem_features = [c for c in self.df.columns if c not in meta_cols]
        
        # 2. Internal One-Hot for Cooling Rate
        rate_dummies = pd.get_dummies(self.df['cooling_rate'], prefix='rate')
        self.train_df = pd.concat([self.df, rate_dummies], axis=1)
        
        # Note: Ensemble models like RF/XGB are less sensitive to noise assignments 
        # via alpha like GP. We'll skip per-sample noise for now but keep feature selection.
        
        self.feature_cols = chem_features + rate_dummies.columns.tolist()
        print(f"Training with {len(chem_features)} chemicals + {len(rate_dummies.columns)} rate categories.")

    def train(self):
        if self.df is None:
            self.load_and_prepare_data()

        X = self.train_df[self.feature_cols]
        y = self.train_df['viability']
        
        # === 交叉验证评分 (Cross-Validation) ===
        print("\n=== Model Evaluation (Ensemble Stacking 5-Fold CV) ===")
        ensemble = EnsembleStackingModel()
        
        # Use sklearn's cross_val_score with the internal model
        r2_scores = cross_val_score(ensemble.model, X, y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(ensemble.model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        print(f"📊 R² Score: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        print(f"   [各折分数: {', '.join([f'{s:.3f}' for s in r2_scores])}]")
        print(f"📊 MSE: {mse_scores.mean():.2f} (±{mse_scores.std():.2f})")
        
        if r2_scores.mean() < 0:
            print("⚠️  警告: R² 为负，模型比平均值更差，数据可能有问题!")
        
        print("\n=== Training Final Ensemble Stacking Model ===")
        final_model = EnsembleStackingModel()
        final_model.fit(X, y, feature_cols=self.feature_cols)
        
        save_path = os.path.join(self.models_dir, "viability_model.joblib")
        final_model.save(save_path)
        print(f"Ensemble model saved to {save_path}")


if __name__ == "__main__":
    t = ModelTrainer()
    t.train()
