# 🔄 Module 4: The Feedback Loop & Operations

Our system is alive. It follows an **Active Learning** logic: The AI suggests -> The Lab tests -> Data is recorded -> The AI gets smarter.

## 1. Step-by-Step Workflow

### Step 1: Launch the Control Center
Run the main loop script:
```bash
python3 run_feedback_loop.py
```

### Step 2: Run the Experiment
*   Choose Option **1** in the menu to see the 8 recipes the AI has generated.
*   The team runs these in the lab and records the results (Viability %).

### Step 3: Log the Results
*   Choose Option **2**. The script will ask you for the result of each Recipe ID.
*   **Crucial**: Once entered, the script automatically appends the result to our main database (`Cryopreservative Data 2026.csv`) and tags it as **`Source: Lab`**.

### Step 4: The One-Click Update
*   Choose Option **3**: "Retrain and Generate." This triggers:
    1.  `clean_data_llm.py`: Re-cleaning the database with your new results.
    2.  `run_training.py`: Feeding the new data into the GP model with "High Trust" weighting.
    3.  `run_optimization.py`: Outputting the next generation of superior recipes.

## 2. Core Mechanisms: "High Trust" & Flexibility

### A. The "High Trust" Tag
Results tagged as `Source: Lab` get a massive priority boost. The model treats them as 100% accurate, even if they contradict old data from a paper.

### B. Embracing Deviations (Recording the Truth)
*   **Human Error**: If the AI suggests 5.0% DMSO but you accidentally pipette 5.5%: **Log 5.5%**.
*   **The Principle**: The AI doesn't demand perfect obedience; it demands **the truth**. It learns much faster from an "accurate mistake" than from a "lied-about success."

### C. Judging "Convergence" (When are we done?)
How do you know if the AI has mastered the problem?
*   **The Check**: Compare the AI's `Predicted_Viability` with your actual lab result.
*   **The Goal**: Once your results are consistently within **±5%** of the AI's prediction, the model has "converged." It has fully understood the chemistry, and the recipes it outputs are at their theoretical peak.

## 3. Important: Backup Your Data

Since `run_feedback_loop.py` edits your main CSV database directly:
*   **Hard Rule**: Manually copy/backup `Cryopreservative Data 2026.csv` before running Option 3.
*   **Why?**: This prevents data pollution if you accidentally mistype a value during entry.

---

**💡 Advice for the Team:**
Don't be discouraged if the AI's predicted score drops during a round. It often happens when the AI is taking you through a "Low-Score Minefield" (Exploration) to make sure there isn't a better path hidden elsewhere. Stick with the process—the model will bounce back stronger in the next iteration.
