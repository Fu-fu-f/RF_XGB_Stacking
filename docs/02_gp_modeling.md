# 🧠 Module 2: The Gaussian Process (GP) Model

For this project, we didn’t just pick a standard neutral network. Instead, we chose the **Gaussian Process (GP)** as our AI's "brain." 

## 1. Why GP? (Predicting Risk, not just Results)

Standard AI simply gives you a number: "I think this recipe will yield 80% viability."
A GP model gives you more context: "I predict 80% viability, but I only have 20% confidence because I’ve never explored this specific chemical combination before."

This **Uncertainty Prediction (Variance)** is the secret sauce for our Bayesian Optimization.

## 2. Key Technical Features

### A. Heteroscedastic Noise: The "Tiered Trust" System
This is a huge highlight of our project. Not all data is equal:
*   **Literature Data**: Might have experimental bias or reporting errors. We assign it a higher "noise" level (Alpha = 0.05).
*   **Our Lab Data**: This is our ground truth. We assign it extremely low noise (Alpha = 1e-10).
*   **The Result**: If literature and our lab data contradict each other, the model will always choose to believe our lab results.

### B. The Kernel: Simulating Chemical Logic
We use an `RBF` (Radial Basis Function) kernel.
*   **The Logic**: It represents "smoothness." In chemistry, if 10% DMSO works well, 10.1% DMSO should act similarly. The kernel helps the model learn the "wavelength" of how concentration changes affect survival.

### C. Transfer Learning
The model can take hyperparameters learned from one cell type (or old battery data) and apply them to a new task. This allows us to get a decent model even when we only have 20-30 new data points from the lab.

### D. Capturing "Synergy" (Non-linear effects)
Our GP model understands that "1 + 1 > 2."
*   **Explanation**: It doesn't just add scores for ingredients; it captures how they interact. For instance, it can learn that Trehalose only becomes powerful when a certain amount of DMSO is already present.

### E. Prediction Clipping
Because GP is based on probability, it can sometimes mathematically predict "-5%" or "110%" viability in unexplored "dark" areas.
*   **The Fix**: We’ve added a safety layer in the code that **clips all outputs between 0% and 100%**, keeping the results scientifically grounded.

### F. The Trainer & Orchestrator: `model_trainer.py`
If the GP model is the "Brain," then `model_trainer.py` is the **Teacher**. It manages the flow of information before the math even starts.
*   **Assigning Alpha (The Trust Knob)**: It scans the `source` column of our data. If it sees "Lab," it sets the noise parameter to near-zero; if it sees "Literature," it sets it higher. This is the script that actually *enforces* our expert trust strategy.
*   **Preparing the "Exam"**: It transforms the raw columns into the exact matrix format required for training, including converting the text-based cooling rates into numerical "One-Hot" features.
*   **Saving the Wisdom**: It orchestrates the entire training run and saves the final result as `viability_model.joblib`, which acts as the system's "long-term memory."

## 3. The Output

After running `python3 run_training.py`, the model is saved to `trained_models/viability_model.joblib`. This file contains the "distilled wisdom" of the entire dataset.

---

**💡 Advice for the Team:**
This model doesn't need thousands of data points. Because of our "High Trust" logic, even a small handful of high-quality lab results can pivot the model toward a much more accurate local optimization.
