import joblib
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from . import config
import warnings

warnings.filterwarnings("ignore")

from .ensemble_model import EnsembleStackingModel

class Recommender:
    def __init__(self, model_path='trained_models/viability_model.joblib'):
        # Use the wrapper class to load so we have the predict(return_std) capability
        self.wrapper = EnsembleStackingModel.load(model_path)
        self.model = self.wrapper # Use the wrapper as the model
        self.features = self.wrapper.feature_cols
        self.y_train_max = np.max(self.wrapper.y_train)
        
        # Identify indices
        self.rate_indices = [i for i, f in enumerate(self.features) if f.startswith('rate_')]
        self.slow_freeze_idx = next((i for i, f in enumerate(self.features) if f == 'rate_slow freeze'), None)
        self.presence_indices = [i for i, f in enumerate(self.features) if '(present)' in f]
        
        # Chemical features (not rates)
        self.chem_indices = [i for i in range(len(self.features)) if i not in self.rate_indices]

    def expected_improvement(self, X, xi=0.01):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        sigma[sigma <= 0] = 1e-10
        mu_sample_opt = self.y_train_max

        with np.errstate(divide='warn'):
            imp = (mu.reshape(-1, 1) - mu_sample_opt - xi)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-10] = 0.0
        return ei

    def bayesian_optimize(self, seed=None):
        # Expert Bounds Implementation (UNIT AWARE)
        bounds = []
        for i, feat in enumerate(self.features):
            feat_lower = feat.lower()
            if i in self.rate_indices:
                if i == self.slow_freeze_idx: bounds.append((1.0, 1.0)) # Force slow freeze
                else: bounds.append((0.0, 0.0))
            elif i in self.presence_indices:
                bounds.append((0.0, 1.0))
            
            # --- DMSO Boundary Logic ---
            elif 'dmso' in feat_lower:
                if '(%)' in feat:
                    bounds.append((0.0, 10.0)) # 10%
                else: # assumed mM
                    bounds.append((0.0, 1280.0)) # 10% = 1280 mM
            
            # --- Sugars (Trehalose/Sucrose) Boundary Logic ---
            elif 'trehalose' in feat_lower or 'sucrose' in feat_lower:
                if '(%)' in feat:
                    bounds.append((0.0, 15.0)) # ~450mM
                else: # assumed mM
                    bounds.append((0.0, 500.0)) 
            
            # --- General Molecule Boundaries ---
            elif '(%)' in feat:
                bounds.append((0.0, 20.0))
            elif '(mm)' in feat_lower:
                bounds.append((0.0, 500.0))
            else:
                bounds.append((0.0, 10.0)) # Fallback

        def objective(x):
            x_eval = x.copy()
            
            # Find chemicals sorted by importance (heuristic: value / max_bound)
            chem_vals = []
            for idx in self.chem_indices:
                norm_val = x[idx] / bounds[idx][1] if bounds[idx][1] > 0 else 0
                chem_vals.append((idx, norm_val))
            
            chem_vals.sort(key=lambda item: item[1], reverse=True)
            top_8_indices = [item[0] for item in chem_vals[:8]]
            
            # Zero out non-top-8 chemicals
            for idx in self.chem_indices:
                if idx not in top_8_indices:
                    x_eval[idx] = 0.0
                elif idx in self.presence_indices:
                    x_eval[idx] = 1.0 if x[idx] > 0.5 else 0.0

            X = x_eval.reshape(1, -1)
            ei = self.expected_improvement(X)[0][0]
            
            # Safety penalty for extreme total concentration (%)
            total_perc = 0
            for i in self.chem_indices:
                feat = self.features[i]
                if '(%)' in feat:
                    total_perc += x_eval[i]
                elif '(mM)' in feat: # Very simplified sum for safety
                    total_perc += (x_eval[i] * 0.1) / 3.0 # Approx 300mM -> 1%
            
            penalty = 0
            if total_perc > 60: penalty += (total_perc - 60) * 50
            
            return -ei + penalty

        result = differential_evolution(
            objective, bounds, strategy='best1bin', 
            maxiter=20, popsize=10, seed=seed
        )
        
        # Post-process strictly to 8 ingredients
        final_x = result.x.copy()
        chem_vals = []
        for idx in self.chem_indices:
            norm_val = final_x[idx] / bounds[idx][1] if bounds[idx][1] > 0 else 0
            chem_vals.append((idx, norm_val))
        chem_vals.sort(key=lambda item: item[1], reverse=True)
        top_8_indices = [item[0] for item in chem_vals[:8]]
        
        for idx in self.chem_indices:
            if idx not in top_8_indices or final_x[idx] < (bounds[idx][1] * 0.001):
                final_x[idx] = 0.0
            elif idx in self.presence_indices:
                final_x[idx] = 1.0 if final_x[idx] > 0.5 else 0.0
                
        pred_viability = self.model.predict(final_x.reshape(1, -1))[0]
        # Clip to realistic range for display and storage
        final_score = np.clip(pred_viability, 0, 100)
        return final_x, final_score

    def suggest_batch_experiment(self, n=8):
        print(f"--- Global Optimization (Top-8, Unit-Aware Expert Bounds) ---")
        save_data = []
        
        for i in range(n):
            print(f"Refining Variant #{i+1}...")
            vec, score = self.bayesian_optimize(seed=i*123)
            
            ingredients = []
            for idx, val in enumerate(vec):
                if val > 0 and idx not in self.rate_indices:
                    ingredients.append((self.features[idx], val))
            
            ingredients.sort(key=lambda x: x[1], reverse=True)
            ing_str = " + ".join([f"{v:.2f} {f}" for f, v in ingredients])
            
            print(f"  Viability Predict: {score:.2f}% | Ingredients: {len(ingredients)}")
            print(f"  Recipe: {ing_str}")
            
            save_data.append({
                'Recipe_ID': i+1,
                'Predicted_Viability': score,
                'Ingredients': ing_str
            })
            
        pd.DataFrame(save_data).to_csv('latest_batch_recipes.csv', index=False)
        print("\n[Output] Results exported to 'latest_batch_recipes.csv'.")

if __name__ == "__main__":
    r = Recommender()
    r.suggest_batch_experiment(n=8)
