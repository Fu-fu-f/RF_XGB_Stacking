import pandas as pd
import os
import subprocess
import sys
import shutil
from datetime import datetime

# --- Config ---
RECIPE_FILE = 'latest_batch_recipes.csv'
RAW_DATA_FILE = 'Cryopreservative Data 2026.csv'

def run_cmd(cmd):
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing: {cmd}")
        return False
    return True

def create_backup():
    """Creates a timestamped backup of the raw data file in the backups/ directory."""
    if not os.path.exists(RAW_DATA_FILE):
        return
    
    backup_dir = 'backups'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"Cryopreservative_Data_2026_{timestamp}.csv.bak"
    backup_path = os.path.join(backup_dir, backup_name)
    
    shutil.copy2(RAW_DATA_FILE, backup_path)
    print(f"\n[BACKUP] Created: {backup_path}")
    
    # Keep only latest 10 backups
    backups = sorted([os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.endswith('.bak')])
    if len(backups) > 10:
        for old_backup in backups[:-10]:
            os.remove(old_backup)
            print(f"[BACKUP] Removed old: {old_backup}")

def main_loop():
    while True:
        print("\n" + "="*40)
        print("   CryoMN Active Learning Feedback Loop")
        print("="*40)
        print("1. View Current Batch Recipes (Last Generated)")
        print("2. Input New Experimental Results (Lab Validation)")
        print("3. Retrain and Generate Next Batch (Auto-Pipeline)")
        print("4. Generate Lab SOP (Convert to Grams/uL)")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            if os.path.exists(RECIPE_FILE):
                df = pd.read_csv(RECIPE_FILE)
                print("\n--- Current Optimal Recipes ---")
                print(df.to_string(index=False))
            else:
                print("\nNo recipe file found. Run option 3 first.")
                
        elif choice == '2':
            if not os.path.exists(RECIPE_FILE):
                print("\nNo recipe file found. Please generate recipes first.")
                continue
                
            create_backup()
            df = pd.read_csv(RECIPE_FILE)
            print("\n--- Inputting Lab Results ---")
            
            new_data_rows = []
            for idx, row in df.iterrows():
                print(f"\nRecipe ID: {row['Recipe_ID']}")
                print(f"Ingredients: {row['Ingredients']}")
                v = input("Enter Observed Viability % (Leave blank to skip this recipe): ").strip()
                
                if v:
                    try:
                        v_float = float(v)
                        new_row = {
                            'All ingredients in cryoprotective solution': row['Ingredients'],
                            'Viability': f"{v_float}%",
                            'Source': 'Lab',
                            'Cooling rate': 'slow freeze'
                        }
                        new_data_rows.append(new_row)
                    except ValueError:
                        print("Invalid input. Skipping.")
            
            if new_data_rows:
                try:
                    raw_df = pd.read_csv(RAW_DATA_FILE)
                    added_df = pd.DataFrame(new_data_rows)
                    updated_df = pd.concat([raw_df, added_df], ignore_index=True)
                    updated_df.to_csv(RAW_DATA_FILE, index=False)
                    print(f"\nSUCCESS: {len(new_data_rows)} new lab records saved.")
                except Exception as e:
                    print(f"Error saving data: {e}")
            else:
                print("\nNo new results entered.")
                
        elif choice == '3':
            print("\n--- Executing Integrated Pipeline ---")
            create_backup()
            if not run_cmd("python3 clean_data_llm.py"): continue
            if not run_cmd("python3 run_training.py"): continue
            if not run_cmd("python3 run_optimization.py"): continue
            print("\nSUCCESS: Model update and next-gen optimization complete.")
            
        elif choice == '4':
            vol = input("\nHow many mL of each recipe do you want to prepare? (Default 10): ").strip()
            if not vol: vol = "10.0"
            run_cmd(f"python3 src/generate_sop.py {vol}")
            
        elif choice == '5':
            print("Exiting. Good luck with your experiments!")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main_loop()
