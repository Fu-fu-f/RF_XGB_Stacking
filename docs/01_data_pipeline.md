# 🧪 Module 1: Data Cleaning & Feature Engineering

To help our AI learn the complex rules of cryopreservation, we first need to transform messy, human-recorded CSV data into a standardized numerical matrix. This process isn't just about "copy-pasting"—it’s a rigorous quality audit.

## 1. The "Entry Ticket": High-Purity Data Protocol

Not every line in the raw CSV makes it into the model. To ensure the AI doesn't learn from bad info, we enforce a strict **Rejection Protocol**. Any problematic data is shunted to `missing_ingredients.csv` for human review:

1.  **Incomplete Records (Zero Tolerance)**:
    *   If a record says "Add DMSO" but **doesn't specify a percentage or mM concentration**, the AI rejects it immediately.
    *   **The Rule**: No guessing in science. Data with missing concentrations is considered invalid.
2.  **Trace Ingredient Filtering ("Dust" Cleaning)**:
    *   If an ingredient's concentration (after conversion) is lower than **0.0001% (1e-4%)**, the entire row is rejected as "unreliable noise."
    *   **The Goal**: We want to exclude accidental impurities or weighing errors. This ensures the AI focuses on actual functional recipes rather than phantom correlations.
3.  **No Commercial Black Boxes**:
    *   Formulations using brand names like `CryoStor` or `Stem-CellBanker` are excluded.
    *   **Reason**: We don't know what's in them, so the AI can't learn anything useful from them.
4.  **Missing Viability**:
    *   Since we're doing supervised learning, rows without a clear `Viability` percentage are discarded.

## 2. Deciphering Cooling Rates

Cooling rates are critical, but human descriptions are chaotic. Our parser uses a "Three-Tier Classification" logic:

*   **Rapid Freeze**: Flagged by keywords like `rapid`, `direct`, `plunge`, `vitrification`, or `immediately`.
*   **Mult Slow Freeze**: Flagged by mentions of `stage`, `ramp`, `hold`, or `mult`.
*   **Slow Freeze**: The default classification (and the most common mode in our lab).

**💡 Tech Note**: These categories are converted into binary "One-Hot" columns (0 or 1), so the AI knows exactly under which thermal conditions a specific survival rate was achieved.

## 3. The Math of Unit Unification

To let percentages (%) and millimolar (mM) values coexist in the same model, we apply physical chemistry formulas:

*   **Large Molecules (Proteins/Polymers)**: Like FBS, HSA, or HES are unified to **Percentage (%)**.
*   **Small Molecules (Sugars/CPAs/Salts)**: Like DMSO, Trehalose, or Glucose are unified to **Millimolar (mM)**.

### The Conversion Engine (MW-based):
The code has an internal library of **Molecular Weights (MW)**:
*   **From % to mM**: `C(mM) = (C(%) * 10000) / MW`
*   **From mM to %**: `C(%) = (C(mM) * MW) / 10000`

*Examples: DMSO (78.13), Glycerol (92.09), Trehalose/Sucrose (342.3), HSA (66500).*

## 4. Cleaning the "Long Tail"

Out of hundreds of potential chemicals, many only appear once.
*   **The Top-15 Rule**: We only keep the 15 most frequently used ingredients.
*   **Preventing "Miracles"**: If a rare ingredient (e.g., "Gold Nanoparticles") appears in just one high-success experiment, the AI might mistake it for a miracle drug. By ignoring these "Long Tail" items, we force the AI to find patterns in reliable, common ingredients.

## 5. Buffer Auto-Expansion

This is a smart feature for handling sloppy historical data.
*   **The Problem**: Many old records just say `DPBS buffer` without listing the salts.
*   **The Solution**: When the parser finds a detailed definition (e.g., `DPBS (NaCl, KCl, ...)`), it caches it.
*   **The Result**: For future rows that only mention the shorthand name, the code automatically "teleports" the full ingredient list back in, saving us from losing valuable data.

---

**💡 Advice for the Team:**
If you notice a new experiment didn't get absorbed by the AI, check:
1. Did you fill in the Viability number?
2. Did you use a standard unit (% or mM)?
3. Is your cooling rate description too vague?
