# 🧠 Module 2: The Ensemble Stacking Model

For this project, we utilize a powerful **Ensemble Stacking** architecture as our AI's "brain," combining the strengths of multiple machine learning algorithms.

## 1. Why Stacking? (Robustness and Synergy)

Instead of relying on a single model, we use a **Stacking Regressor** that aggregates predictions from two diverse base learners:
*   **Random Forest (RF)**: Excellent at capturing non-linear relationships and handling high-dimensional data without overfitting.
*   **XGBoost (Extreme Gradient Boosting)**: A state-of-the-art boosting algorithm that excels at finding fine-tuned patterns in tabular data.
*   **Final Estimator (Ridge Regression)**: A meta-learner that learns how to best weigh the advice from RF and XGBoost to produce the final viability prediction.

## 2. Key Technical Features

### A. Uncertainty Estimation (The Heuristic Approach)
Standard ensemble models don't naturally provide uncertainty like Gaussian Processes. However, for our **Bayesian Optimization**, we've implemented a heuristic:
*   **Disagreement as Uncertainty**: The model calculates the standard deviation ($\sigma$) between the base learners (RF vs XGBoost).
*   **The Logic**: If RF and XGBoost agree, the model is confident. If they disagree significantly, it signals a high-uncertainty area that warrants further **Exploration**.

### B. Capturing "Synergy" (Non-linear effects)
The ensemble approach is particularly good at understanding that "1 + 1 > 2."
*   **Explanation**: Random Forest and XGBoost use tree-based logic to split the chemical space. This naturally captures interaction effects (e.g., "Trehalose only works effectively if DMSO is within a specific range").

### C. Heteroscedastic Awareness
While tree-based models don't use "Alpha" noise parameters in the same way as GP, our trainer is designed to prioritize data quality:
*   **Lab Data Priority**: Our lab results are the ground truth used to guide the final model weights.

### D. Prediction Clipping
To ensure scientific validity, the system includes a safety layer that **clips all outputs between 0% and 100%**, preventing unrealistic viability predictions.

### E. The Trainer & Orchestrator: `model_trainer.py`
The `ModelTrainer` manages the entire pipeline:
*   **Feature Engineering**: Converts text-based cooling rates into numerical "One-Hot" features.
*   **Cross-Validation**: Performs a 5-fold CV to ensure the Stacking model generalizes well to unseen data.
*   **Model Persistence**: Saves the trained ensemble as `viability_model.joblib`, acting as the system's "long-term memory."

## 3. The Output

After running `python3 run_training.py`, the model is saved to `trained_models/viability_model.joblib`. This file contains the "distilled wisdom" of the entire dataset.

---

**💡 Advice for the Team:**
The Stacking model is highly robust against outliers and noise common in literature data. Even a small number of high-quality lab results can significantly shift the "Final Estimator's" weights, pivoting the model toward accurate local optimization.
