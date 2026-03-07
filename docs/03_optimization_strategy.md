# 🎯 Module 3: Optimization Strategy & Constraints

If the Stacking Model is the "Brain," this module is the "Action Plan." Its mission: **Decide exactly which experiments we should run next based on current knowledge.**

## 1. The Core Idea: Expected Improvement (EI)

This is the heartbeat of Bayesian Optimization. In every round, the AI faces a dilemma:
*   **Exploitation**: Stick to known "high-score" areas to refine the recipe.
*   **Exploration**: Venture into "wild" areas where we haven't tested anything yet.
The **EI Algorithm** acts as a "Surprise Evaluator," picking the recipe that has the highest potential to break our current survival record. By using the disagreement between RF and XGBoost as our uncertainty measure, the EI algorithm can effectively identify regions of interest.

## 2. Real-world Constraints

### A. The "Top-8" Limit (Lab-Friendly Recipes)
Theoretically, an AI could suggest a recipe with 50 ingredients. In a real lab, that's impossible to pipette.
*   **The Filter**: During the search, our internal "Gating System" compares the contribution of every chemical and **automatically kills off the weakest links**, leaving us with exactly 8 ingredients.
*   **Normalized Ranking**: This filter is smart—it doesn't just look at weight. It looks at **"% of local max."** This ensures a tiny amount of a high-potency additive isn't "bullied" out of the Top-8 list by a large amount of a common sugar.

### B. Expert Sensitivity Bounds
We don't want the AI to suggest "100% DMSO." We’ve set specific "speed limits" based on chemical common sense:
*   `DMSO`: MAX 10%
*   `Sugars`: MAX 500mM

### C. The 60% Concentration "Red Line" (Penalty)
AI can be "greedy" and try to jam as many additives as possible into a solution to lower the freezing point.
*   **The Penalty**: If the total concentration (sum of all ingredients) exceeds **60%**, the AI receives a massive "score penalty" (a virtual slap on the wrist).
*   **Bioscience Reality Checks**:
    1.  **Toxicity**: 40%+ organic solvents will dissolve cell membranes before they even freeze.
    2.  **Osmotic Shock**: Extreme concentrations squeeze water out of the cell so fast it causes irreversible physical damage.
    3.  **Viscosity**: At 60%+, the solution becomes a syrup that you can't pipette accurately and won't conduct heat properly.

### D. Cooling Mode Lock: Slow Freeze
While the AI knows about other cooling methods, we have **locked its suggestions to "Slow Freeze" mode**.
*   **Why?**: This is our lab’s most robust and reproducible setup. We want the AI to find the best chemicals for *our* equipment, not some theoretical setup we don't have.

## 3. The Search Engine: Differential Evolution

Because of all these hard constraints (60% limit, Top-8 limit), the "landscape" of the math is very jagged. We use **Differential Evolution**.
*   **Analogy**: Imagine dropping a hundred tiny robots across a map. They communicate with each other to find the highest mountain peak (the best EI score), rather than just looking at the ground beneath their feet.

## 4. Cold Start: Latin Hypercube Sampling (LHS)

When the system is first set up, or when we have very little **Lab** data (less than 8 samples), model-based Bayesian optimization can be unreliable. In these cases, we use **LHS**.

*   **The Problem**: If we only have literature data, the AI might get "stuck" in a region that worked for another lab but doesn't work for our specific cells or equipment.
*   **The Solution (Space-Filling)**: LHS is a statistical method that generates a distribution of recipes that are elegantly spread out across the entire chemical space.
*   **The Logic**: Imagine a Sudoku grid; LHS ensures that for every ingredient, we test a wide range of concentrations, and no two suggested recipes are too similar.
*   **Transition**: Once we collect 8+ lab results, the system automatically switches from LHS (Pure Exploration) to Bayesian Optimization (Strategic Exploitation).

## 5. Batch Generation & Diversity

We generate 8 variants (#1 to #8) per run.
*   **Deterministic Diversity**: Every variant uses a unique seed. This ensures you get a "Portfolio" of recipes—some are safe improvements, others are bold, "out-of-the-box" ideas.

---

**💡 Advice for the Team:**
The AI isn't just picking what it thinks is "good." It’s picking what it thinks is **"knowledge-rich."** Even if a suggested recipe fails in the lab, that failure provides critical data that keeps the AI from making the same mistake twice.

---

## Development Meta: The Scientist's Instruction (Prompt)

The following instruction defines the boundaries and strategy for the optimization engine:

> "Build run_optimization.py and the Recommender logic to transition from exploration to exploitation.
> 1. Cold Start Strategy: Use Latin Hypercube Sampling (LHS) to evenly explore the chemical space if we have less than 8 Lab Samples. Switch to Bayesian Optimization via Expected Improvement (EI) once we have enough ground truth.
> 2. Search Logic: Use Differential Evolution to maximize the EI score.
> 3. Scientific Constraints (Hard Gating):
>    - The 60% Red-Line Penalty: If the total concentration of all ingredients exceeds 60%, apply a massive penalty. This prevents osmotic shock, membrane toxicity, and unpipettable viscosity.
>    - The Top-8 Lab Feasibility: Each suggested recipe must have exactly 8 ingredients. Implement a Gating System with Normalized Ranking to eliminate weak ingredients while preserving potent trace additives.
>    - Safety Bounds: Lock DMSO at a 10% max cap and Sugars at 500mM."
