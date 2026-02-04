#!/usr/bin/env python3
import pandas as pd
import re
import sys
import os

# --- Constants & Database ---
MW_MAP = {
    'dmso': 78.13, 'eg': 62.07, 'glycerol': 92.09,
    'trehalose': 342.3, 'sucrose': 342.3, 'glucose': 180.16,
    'mannitol': 182.17, 'proline': 115.13, 'hsa': 66500,
    'kh2po4': 136.09, 'k2hpo4': 174.2, 'mgcl2': 95.21,
    'glycine': 75.07, 'taurine': 125.15, 'sp2': 500000,
}

# Liquids that should be measured by volume (uL)
LIQUIDS = ['dmso', 'eg', 'glycerol', 'fbs', 'proh']

def parse_ingredient_line(line):
    """Parses '148.12 trehalose(mM) + 5.53 hsa(%)' into list of dicts"""
    parts = line.split(' + ')
    results = []
    for p in parts:
        m = re.search(r'([\d\.]+)\s+([^\(]+)\((%|mm)\)', p, re.IGNORECASE)
        if m:
            results.append({
                'val': float(m.group(1)),
                'name': m.group(2).strip().lower(),
                'unit': m.group(3).lower()
            })
    return results

def generate_sop(target_volume_ml=10.0):
    input_file = 'latest_batch_recipes.csv'
    output_file = 'lab_ready_sop.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    sop_rows = []

    for _, recipe in df.iterrows():
        rid = recipe['Recipe_ID']
        ingredients = parse_ingredient_line(recipe['Ingredients'])
        
        # Sort: Powders first, then Liquids (except DMSO), DMSO last
        powders = [i for i in ingredients if i['name'] not in LIQUIDS]
        liquids = [i for i in ingredients if i['name'] in LIQUIDS and i['name'] != 'dmso']
        dmso_list = [i for i in ingredients if i['name'] == 'dmso']

        step = 1
        
        def add_row(comp, state, action, amount, note):
            nonlocal step
            # Only show Recipe_ID on the very first step of the recipe
            display_id = rid if step == 1 else ""
            sop_rows.append({
                'Recipe_ID': display_id,
                'Step': step,
                'Component': comp,
                'State': state,
                'Action': action,
                'Amount': amount,
                'Note': note
            })
            step += 1

        # 1. Preparation Note
        add_row('START', '-', 'Prepare Tube', f'{target_volume_ml}mL scale', 'Clean environment')

        # 2. Add Powders
        for p in powders:
            if p['unit'] == 'mm':
                mw = MW_MAP.get(p['name'], 100.0)
                grams = (p['val'] / 1000.0) * mw * (target_volume_ml / 1000.0)
            else: # %
                grams = (p['val'] / 100.0) * target_volume_ml
            
            add_row(p['name'], 'Solid', 'WEIGH', f'{grams:.4f} g', f"Target {p['val']}{p['unit']}")

        # 3. Dissolve Step
        add_row('Base Media', 'Liquid', 'ADD & DISSOLVE', f'~{target_volume_ml*0.7:.1f} mL', 'Vortex until clear')

        # 4. Add Regular Liquids
        for l in liquids:
            if l['unit'] == 'mm':
                mw = MW_MAP.get(l['name'], 78.0)
                ml = (l['val'] * mw / 10000.0) * (target_volume_ml / 100.0)
            else: # %
                ml = (l['val'] / 100.0) * target_volume_ml
            
            add_row(l['name'], 'Liquid', 'ADD', f'{ml*1000:.1f} uL', 'Mix gently')

        # 5. Add DMSO
        for d in dmso_list:
            mw = 78.13
            ml = (d['val'] * mw / 10000.0) * (target_volume_ml / 100.0)
            add_row('DMSO', 'Liquid', 'DROPWISE', f'{ml*1000:.1f} uL', 'Slowly! Exothermic')

        # 6. Final Top-up
        add_row('Base Media', 'Liquid', 'TOP UP', f'To {target_volume_ml} mL', 'Line up the meniscus')

        # Add an empty row for visual separation between recipes
        sop_rows.append({k: "" for k in sop_rows[0].keys()})

    pd.DataFrame(sop_rows).to_csv(output_file, index=False)
    print(f"\n[SUCCESS] English SOP saved to '{output_file}' for {target_volume_ml}mL batch.")

if __name__ == "__main__":
    vol = 10.0
    if len(sys.argv) > 1:
        try: vol = float(sys.argv[1])
        except: pass
    generate_sop(vol)
