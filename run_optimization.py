# run_optimization.py
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingredient_suggestion import Recommender

def main():
    try:
        rec = Recommender()
        
        # Check if we have lab data to decide which method to use
        import pandas as pd
        df = pd.read_csv('cleaned_data_2026.csv')
        n_lab = (df['source'] == 'Lab').sum()
        
        if n_lab < 8:
            print("📊 No sufficient Lab data detected. Using Latin Hypercube Sampling for exploration.")
            print("   (Switch to Bayesian optimization after collecting 8+ lab results)\n")
            rec.suggest_batch_experiment(n=8, method='lhs')
        else:
            print("📊 Lab data detected. Using Bayesian Optimization for exploitation.\n")
            rec.suggest_batch_experiment(n=8, method='bayesian')
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
