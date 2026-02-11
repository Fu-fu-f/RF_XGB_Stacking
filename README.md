# CryoMN Optimization System

This project is a machine learning tool for optimizing cryopreservation recipes, specifically for cryomicroneedle (cryoMN) applications. It uses an **Ensemble Stacking model (Random Forest + XGBoost)** to predict cell viability and suggests the best ingredient combinations within expert-defined safety bounds.

### Quick Start: How to Run
The easiest way to use the system is the interactive loop. Open your terminal and run:

`python3 run_feedback_loop.py`

Once inside, you can choose from these options:
1. **View Recipes**: Check the latest 8 suggested recipes.
2. **Input Results**: Enter the viability results from your lab experiments. The system will automatically mark these as high-trust "Lab" data.
3. **Update Everything**: One-click to clean the data, retrain the model, and generate the next batch of optimized recipes.

If you prefer running steps manually:
- Clean data: `python3 clean_data_llm.py`
- Train model: `python3 run_training.py`
- Generate recipes: `python3 run_optimization.py`

---

### Data Handling & Unit Conversion

When we pull data from different research papers, one scientist might use "mM" while another uses "%". To train the AI, we have to make sure they are talking the same language.

#### 1. How we unify the units
- **Small Molecules** (like DMSO, Sugars, Salts): We standardize these to **mM**.
- **Large Molecules/Proteins** (like HSA, FBS, PEG): We standardize these to **%**.

#### 2. How the conversion works (Plain English)
We use the Molecular Weight (MW) of each chemical to bridge the gap between "weight" and "count".

- **To get mM from %**: 
  Take the percentage * 10,000 / Molecular Weight.
  *(Example: 10% DMSO becomes ~1280 mM)*

- **To get % from mM**: 
  Take the mM value * Molecular Weight / 10,000.
  *(Example: 100 mM Sucrose becomes ~3.42%)*

---

### Data Trust (The Source Tag)

How does the AI know who to trust more: a random paper from 2005 or your experiment from yesterday? 

#### 1. Lab Data vs. Literature Data
- Every row in your database has a **Source** column.
- Literature data is marked as `Literature`.
- Your results are automatically marked as `Lab`.

#### 2. Persistent High Weight
- Your lab data never "expires" or loses its importance. 
- Every time you retrain the model, it scans the whole file. Any row marked as `Lab` is given a **"Gold Standard"** status (extremely low noise).
- The AI will prioritize fitting its curve to your lab points above all else, while using literature only as a general background guide.

---

### Cooling Rate Logic

The system identifies how cells were frozen to ensure the AI doesn't mix up results from different methods.

#### 1. How the AI reads the cooling rate
The code scans the text descriptions for keywords:
- **Rapid Freeze**: Keywords like "Liquid Nitrogen", "Plunge", "Vitrification".
- **Mult Slow Freeze**: Mentions of multiple "Stages", "Steps", or "Holds".
- **Slow Freeze**: The standard -1°C/min approach.

#### 2. Standard Setting
Currently, all recipe recommendations are optimized for **Slow Freeze** (e.g., using a Mr. Frosty or controlled rate freezer) because it is the most reproducible method in a standard lab.

---

### Lab Preparation Guide: How to Mix Recipes

#### 1. Handling the Percentage (%) Ingredients
Percentage means how many grams to add per 100 mL of solution.
**Formula**: Grams to add = Total Volume (mL) * (Percentage / 100)
*Example: For 10 mL of solution with 11.40% PEG:*
10 * (11.40 / 100) = **1.14 grams of PEG**

#### 2. Handling the mM Ingredients
Calculate weight based on Molecular Weight (MW).
**Formula**: Grams to add = (mM / 1000) * MW * (Total Volume (mL) / 1000)
*Example: For 10 mL of solution with 355 mM Sucrose (MW 342.3):*
(355 / 1000) * 342.3 * (10 / 1000) = **1.215 grams of Sucrose**

#### 3. Best Lab Practice (Step-by-Step)
1. **Add Solids**: Weigh all powders (sugars, proteins) and add them to your tube.
2. **Add Base Media**: Add about 70% of your target volume (e.g., 7 mL DMEM) and vortex to dissolve.
3. **Add Liquids**: Use a pipette to add liquid components like FBS or DMSO.
4. **Top Up**: Finish by adding base media until you reach the exact target mark.

---

### Project Structure
- **src/**: Core math and optimization logic.
- **trained_models/**: The "brain" of the trained AI.
- **Cryopreservative Data 2026.csv**: Your main database.
- **latest_batch_recipes.csv**: Your current experimental plan.