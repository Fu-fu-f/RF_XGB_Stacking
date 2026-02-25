# src/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score
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
        
        # 3. Add critical interaction features
        interaction_features = self._create_interaction_features(chem_features)
        
        # 4. Calculate dynamic sample weights based on data source
        self.sample_weights = self._calculate_sample_weights()
        
        self.feature_cols = chem_features + interaction_features + rate_dummies.columns.tolist()
        print(f"Training with {len(chem_features)} chemicals + {len(interaction_features)} interactions + {len(rate_dummies.columns)} rate categories.")
    
    def _create_interaction_features(self, chem_features):
        """
        Create biologically meaningful interaction terms.
        
        Key interactions based on cryobiology literature:
        1. DMSO × Sugars (synergistic membrane protection)
        2. DMSO × Proteins (colligative + protective synergy)
        3. Sugar × Protein (glass transition enhancement)
        """
        interaction_cols = []
        
        # Find relevant chemical indices
        dmso_cols = [c for c in chem_features if 'dmso' in c.lower()]
        sugar_cols = [c for c in chem_features if any(s in c.lower() for s in ['trehalose', 'sucrose', 'glucose'])]
        protein_cols = [c for c in chem_features if any(p in c.lower() for p in ['fbs', 'hsa', 'albumin'])]
        
        # DMSO × Sugar interactions
        for dmso in dmso_cols:
            for sugar in sugar_cols:
                interaction_name = f"{dmso}_x_{sugar}"
                self.train_df[interaction_name] = self.train_df[dmso] * self.train_df[sugar]
                interaction_cols.append(interaction_name)
        
        # DMSO × Protein interactions
        for dmso in dmso_cols:
            for protein in protein_cols:
                interaction_name = f"{dmso}_x_{protein}"
                self.train_df[interaction_name] = self.train_df[dmso] * self.train_df[protein]
                interaction_cols.append(interaction_name)
        
        # Sugar × Protein interactions
        for sugar in sugar_cols:
            for protein in protein_cols:
                interaction_name = f"{sugar}_x_{protein}"
                self.train_df[interaction_name] = self.train_df[sugar] * self.train_df[protein]
                interaction_cols.append(interaction_name)
        
        if interaction_cols:
            print(f"   Created {len(interaction_cols)} interaction features")
        
        return interaction_cols
    
    def _calculate_sample_weights(self):
        """
        Dynamic weighting strategy for Lab vs Literature data.
        
        Strategy:
        - 0-10 Lab samples: 15x weight (bootstrap phase, need strong signal)
        - 11-30 Lab samples: 10x weight (learning phase, balance exploration)
        - 31-50 Lab samples: 6x weight (refinement phase)
        - 50+ Lab samples: 3x weight (mature phase, let data speak)
        """
        source = self.train_df['source']
        n_lab = (source == 'Lab').sum()
        n_lit = (source == 'Literature').sum()
        
        if n_lab == 0:
            print("📊 No Lab data yet. All samples weighted equally.")
            return np.ones(len(source))
        
        # Dynamic weight based on Lab data quantity
        if n_lab <= 10:
            lab_weight = 15.0
            phase = "Bootstrap"
        elif n_lab <= 30:
            lab_weight = 10.0
            phase = "Learning"
        elif n_lab <= 50:
            lab_weight = 6.0
            phase = "Refinement"
        else:
            lab_weight = 3.0
            phase = "Mature"
        
        weights = np.where(source == 'Lab', lab_weight, 1.0)
        
        # Calculate effective sample contribution
        lab_effective = n_lab * lab_weight
        total_effective = lab_effective + n_lit
        lab_influence_pct = (lab_effective / total_effective) * 100
        
        print(f"\n📊 Sample Weighting Strategy ({phase} Phase):")
        print(f"   Lab data: {n_lab} samples × {lab_weight:.1f} weight = {lab_effective:.0f} effective samples")
        print(f"   Literature data: {n_lit} samples × 1.0 weight = {n_lit} effective samples")
        print(f"   Lab influence: {lab_influence_pct:.1f}% of total training signal")
        
        return weights

    def train(self):
        if self.df is None:
            self.load_and_prepare_data()

        X = self.train_df[self.feature_cols]
        y = self.train_df['viability']
        
        # === 交叉验证评分 (Cross-Validation) ===
        print("\n=== Model Evaluation (Ensemble Stacking 5-Fold CV) ===")
        ensemble = EnsembleStackingModel()
        
        # Note: cross_val_score doesn't support sample_weight directly with StackingRegressor
        # We'll evaluate without weights for CV, but train final model with weights
        r2_scores = cross_val_score(ensemble.model, X, y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(ensemble.model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        print(f"📊 R² Score: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        print(f"   [各折分数: {', '.join([f'{s:.3f}' for s in r2_scores])}]")
        print(f"📊 MSE: {mse_scores.mean():.2f} (±{mse_scores.std():.2f})")
        
        if r2_scores.mean() < 0:
            print("⚠️  警告: R² 为负，模型比平均值更差，数据可能有问题!")
        
        print("\n=== Training Final Ensemble Stacking Model (with sample weights) ===")
        final_model = EnsembleStackingModel()
        final_model.fit(X, y, feature_cols=self.feature_cols, sample_weights=self.sample_weights)
        
        save_path = os.path.join(self.models_dir, "viability_model.joblib")
        final_model.save(save_path)
        print(f"Ensemble model saved to {save_path}")


if __name__ == "__main__":
    t = ModelTrainer()
    t.train()
