# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
from . import config
from .gp_model import GaussianProcessModel

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
        
        # 3. NOISE ASSIGNMENT (Expert Strategy C)
        # Higher alpha = more noise (less trust)
        # Literature (default/0) gets 0.05, Lab data gets 1e-10 (high trust)
        self.train_df['alpha'] = 0.05
        if 'source' in self.df.columns:
            # Ensure source column is handled as string even if it contains NaNs
            source_series = self.df['source'].fillna('').astype(str).str.lower()
            is_lab = source_series == 'lab'
            self.train_df.loc[is_lab, 'alpha'] = 1e-10
            lab_count = is_lab.sum()
            print(f"Detected {lab_count} Lab-validated samples (High Trust).")
        
        self.feature_cols = chem_features + rate_dummies.columns.tolist()
        print(f"Training with {len(chem_features)} chemicals + {len(rate_dummies.columns)} rate categories.")

    def train(self):
        if self.df is None:
            self.load_and_prepare_data()

        X = self.train_df[self.feature_cols]
        y = self.train_df['viability']
        alphas = self.train_df['alpha'].values
        
        print("\n=== Training GP Model with Heteroscedastic Noise ===")
        gp_model = GaussianProcessModel()
        gp_model.fit(X, y, feature_cols=self.feature_cols, alpha_arr=alphas)
        
        save_path = os.path.join(self.models_dir, "viability_model.joblib")
        gp_model.save(save_path)
        print(f"Expert-Aligned model saved to {save_path}")

if __name__ == "__main__":
    t = ModelTrainer()
    t.train()
