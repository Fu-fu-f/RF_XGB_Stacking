#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import os

# --- Configuration ---
INPUT_FILE = 'Cryopreservative Data 2026.csv'
OUTPUT_CLEAN_FILE = 'cleaned_data_2026.csv'
OUTPUT_COMMERCIAL_FILE = 'commercial_formulations.csv'
OUTPUT_MISSING_FILE = 'missing_ingredients.csv'
TOP_N_FEATURES = 15

COMMERCIAL_KEYWORDS = [
    'cryostor', 'stem-cellbanker', 'cellbanker', 'mesencult', 
    'synth-a-freeze', 'biofreeze', 'cellfreeze', 'bambanker', 
    'freezis', 'cryosofree', 'prime-xv', 'banker', 'reprocell'
]

# Media to ignore
BASE_MEDIA_LIST = [
    'dmem', 'pbs', 'phosphate buffer saline', 'saline', 'a-mem', 'culture', 
    'α-mem', 'ham f-12', 'f-12', 'medium', 'm-199', 'dpbs', 'imdm', 'rpm',
    'culture medium', 'basal medium', 'cocktail solution', 'solution',
    'electroporation buffer', 'lysis buffer', 'buffer', 'water', 'base'
]

# Molecular Weights (approximate for conversion)
# C(%) = (C(mM) * MW) / 10000
# C(mM) = (C(%) * 10000) / MW
MW_MAP = {
    'dmso': 78.13,
    'eg': 62.07,
    'glycerol': 92.09,
    'trehalose': 342.3,
    'sucrose': 342.3,
    'glucose': 180.16,
    'mannitol': 182.17,
    'proline': 115.13,
    'hsa': 66500, # Human Serum Albumin
    'kh2po4': 136.09,
    'k2hpo4': 174.2,
    'mgcl2': 95.21,
    'glycine': 75.07,
    'taurine': 125.15,
}

CHEMICAL_MAP = {
    'dmso': ['me2so', 'dimethyl sulfoxide', 'me 2 so', 'dimso'],
    'fbs': ['fcs', 'fetal bovine serum', 'fetal calf serum', 'serum', 'bovine serum'],
    'eg': ['ethylene glycol', 'ethyleneglycol'],
    'hes': ['hydroxyethyl starch', 'hes450', 'hydroxychyl starch'],
    'hsa': ['human albumin', 'human serum albumin', 'has', 'ha', 'albumin'],
    'trehalose': ['trehalose dihydrate', 'trehalose'],
    'sucrose': ['saccharose', 'cane sugar'],
    'glucose': ['d-glucose', 'dextrose'],
    'dextran': ['dextran-40', 'dextran40'],
    'glycerol': ['glycerin', 'glycerine'],
    'mc': ['methylcellulose', 'methyl cellulose'],
    'ectoin': ['ectoine'],
    'proline': ['l-proline'],
    'proh': ['propylene glycol', '1,2-propanediol'],
    'peg': ['polyethylene glycol', 'poly(ethylene glycol)', 'peg400', 'peg1000', 'peg 4000'],
    'mannitol': ['d-mannitol'],
    'mgcl2': ['magnesium chloride'],
    'kh2po4': ['potassium phosphate mono', 'kh2po4'],
    'k2hpo4': ['dipotassium phosphate', 'potassium phosphate dibasic', 'k2hpo4'],
    'taurine': [],
    'glycine': [],
    'sericin': [],
    'poly': ['polyampholyte', 'cooh-pll'],
}

BUFFER_LIBRARY = {}

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower().replace('\n', ' ')
    text = re.sub(r'\b[a-z0-9][\)\.]\s+', ' ', text)
    text = re.sub(r'doi:.*', '', text)
    return text.strip()

def unify_unit(canonical, val, unit):
    """
    Physically correct unit unification.
    Target:
    - Large molecules (Proteins/Polymers) -> %
    - Small molecules (Sugars/Salts/CPAs) -> mM
    """
    unit = unit.lower()
    mw = MW_MAP.get(canonical)

    # 1. Target: % (Large Molecules)
    if canonical in ['fbs', 'hsa', 'albumin', 'sericin', 'poly', 'hes', 'mc']:
        target_name = f"{canonical}(%)"
        if unit == 'present': return target_name, 1.0
        if unit == '%': return target_name, val
        if unit in ['mg/ml', 'mg/l', 'g/l', 'µg/ml']:
            # g/100ml = %
            if unit == 'mg/ml': return target_name, val / 10.0
            if unit == 'g/l': return target_name, val / 10.0
            if unit == 'mg/l': return target_name, val / 10000.0
            if unit == 'µg/ml': return target_name, val / 10000.0
        if mw and (unit == 'mm' or unit == 'm' or unit == 'mmol/l'):
            actual_mm = val * 1000 if unit == 'm' else val
            actual_perc = (actual_mm * mw) / 10000.0
            return target_name, actual_perc
        return target_name, val # Fallback

    # 2. Target: mM (Small Molecules)
    # Special case: DMSO, EG, Glycerol often used in % in literature. 
    # But for optimization consistency, we convert to target unless % is preferred for intuition.
    # We will pick mM for consistency with salts/sugars.
    target_name = f"{canonical}(mM)"
    
    # Unit conversion math
    if unit in ['m', 'mol/l']: return target_name, val * 1000.0
    if unit in ['mm', 'mmol/l']: return target_name, val
    if unit == 'present': return target_name, 10.0 # Heuristic placeholder
    
    if unit == '%':
        if mw: 
            return target_name, (val * 10000.0) / mw
        else:
            # If no MW, return % to avoid errors
            return f"{canonical}(%)", val

    if unit in ['mg/ml', 'g/l', 'mg/l']:
        if mw:
            gL = val if unit == 'g/l' or unit == 'mg/ml' else val / 1000.0
            return target_name, (gL * 1000.0) / mw
        else:
            return f"{canonical}({unit})", val

    return f"{canonical}({unit})", val

def process_single_ingredient(part, depth=0):
    if depth > 2: return [] 
    part = part.strip(' .(),[]-+;:')
    if not part: return []
    
    # 1. Buffer Handling
    for buf_name, buf_details in BUFFER_LIBRARY.items():
        if buf_name in part.lower():
            if '(' not in part or ')' not in part:
                recursive_parts = re.split(r'\s*\+\s*|\s*,\s*|\s+and\s+|\s*;\s*', buf_details)
                results = []
                for rp in recursive_parts:
                    results.extend(process_single_ingredient(rp, depth + 1))
                return results

    # 2. Extraction
    # Format A (Literature): 10% DMSO or 100 mM Trehalose
    pattern = re.search(r'([\d\.]+)\s*(%|mm|mmol/l|m|mol/l|µm|µg/ml|ng/ml|mg/ml|g/l)', part, re.IGNORECASE)
    
    # Format B (Generated): 10.00 dmso(mM)
    pattern_gen = re.search(r'([\d\.]+)\s+([^\(]+)\((%|mm|mmol/l|m|mol/l|µm|µg/ml|ng/ml|mg/ml|g/l)\)', part, re.IGNORECASE)

    if pattern_gen:
        try:
            val = float(pattern_gen.group(1))
            name = pattern_gen.group(2).strip(' .(),[]-+;:')
            unit = pattern_gen.group(3).lower()
        except:
            return "REJECT_MISSING_CONC"
    elif pattern:
        try:
            val = float(pattern.group(1))
            unit = pattern.group(2).lower()
            name = part.replace(pattern.group(0), '').strip(' .(),[]-+;:')
        except:
            return "REJECT_MISSING_CONC"
    else:
        return "REJECT_MISSING_CONC"
    
    # ... existing media cleaning ...
    for m in BASE_MEDIA_LIST:
        name = re.sub(r'\bin ' + re.escape(m) + r'\b.*', '', name, flags=re.IGNORECASE).strip(' .(),[]-+;:')
    
    name = re.sub(r'\(.*?\)', '', name).strip(' .(),[]-+;:')
    if not name or len(name) < 2: return []

    # 3. Canonical mapping
    for canonical, synonyms in CHEMICAL_MAP.items():
        search_list = [canonical] + synonyms
        found = False
        for s in search_list:
            if re.search(r'\b' + re.escape(s.lower()) + r'\b', name.lower()):
                found = True
                break
        if found:
            key, final_val = unify_unit(canonical, val, unit)
            
            # --- Dust Concentration Cleaning ---
            # Threshold: 1e-4 (0.0001% or equivalent). Anything below is considered noise.
            # Convert trace check to percentage for common ground
            trace_limit = 1e-4
            check_val = final_val
            if "(mm)" in key.lower():
                # Rough check: 0.1mM is typically > 1e-4% for small molecules
                if final_val < 0.01: return "REJECT_TRACE" 
            else:
                if final_val < trace_limit: return "REJECT_TRACE"
                
            return [(key, final_val)]
    
    # 4. Final exclusion
    low_name = name.lower()
    if any(m == low_name for m in BASE_MEDIA_LIST): return []
    return [(f"{name}({unit})", val)]

def get_cooling_rate(text):
    text = clean_text(text)
    if not text: return 'unknown'
    if any(kw in text for kw in ['rapid', 'direct', 'plunge', 'vitrif', 'immediately']):
        return 'rapid freeze'
    sc = text.count('stage') + text.count('ramp') + text.count('cool at')
    if sc >= 2 or 'mult' in text or 'hold' in text or 'wait' in text:
        return 'mult slow freeze'
    return 'slow freeze'

def main():
    print(f"Opening {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    # PASS 1: Buffer Library
    print("Pre-scanning for buffer definitions...")
    for _, row in df.iterrows():
        raw_ing = str(row.get('all ingredients in cryoprotective solution', '')).lower()
        match = re.search(r'([a-z0-9\s]+buffer)\s*\((.*?)\)', raw_ing)
        if match:
            buf_name = match.group(1).strip()
            details = match.group(2).strip()
            if len(details) > 5 and ('mm' in details or '%' in details or ',' in details):
                if buf_name not in BUFFER_LIBRARY:
                    BUFFER_LIBRARY[buf_name] = details
                    print(f"  Defined Buffer: '{buf_name}' -> '{details}'")

    clean_list = []
    commercial_list = []
    missing_list = []
    
    # PASS 2: Processing
    for _, row in df.iterrows():
        raw_ing = str(row.get('all ingredients in cryoprotective solution', '')).strip()
        source = str(row.get('source', 'Literature')).strip()
        
        if any(kw in raw_ing.lower() for kw in COMMERCIAL_KEYWORDS):
            commercial_list.append(row)
            continue
            
        text = clean_text(raw_ing)
        parts = re.split(r'\s*\+\s*|\s*,\s*|\s+and\s+|\s*;\s*', text)
        ingredients = {}
        found_buffer_termed = 'buffer' in raw_ing.lower()
        
        for p in parts:
            res = process_single_ingredient(p)
            if res in ["REJECT_MISSING_CONC", "REJECT_TRACE"]:
                ingredients = {} # Reset to trigger missing logic
                break
            for key, val in res:
                ingredients[key] = ingredients.get(key, 0) + val
        
        v_str = str(row.get('viability', ''))
        v = np.nan
        m = re.search(r'(\d+\.?\d*)', v_str)
        if m and 'not mention' not in v_str.lower():
            v = float(m.group(1))
            
        if (not ingredients and found_buffer_termed) or not ingredients or pd.isna(v):
            missing_list.append(row)
            continue
        
        entry = {
            'original_ingredients': raw_ing,
            'viability': v,
            'cooling_rate': get_cooling_rate(row.get('cooling rate', '')),
            'source': source
        }
        entry.update(ingredients)
        clean_list.append(entry)
        
    df_clean_all = pd.DataFrame(clean_list).fillna(0)
    meta = ['original_ingredients', 'viability', 'cooling_rate', 'source']
    feats_only = [c for c in df_clean_all.columns if c not in meta]
    usage = (df_clean_all[feats_only] > 0).sum().sort_values(ascending=False)
    
    top_keep = usage.head(TOP_N_FEATURES).index.tolist()
    final_cols = ['original_ingredients'] + top_keep + ['viability', 'cooling_rate', 'source']
    df_clean_final = df_clean_all[final_cols]
    
    df_clean_final.to_csv(OUTPUT_CLEAN_FILE, index=False)
    print(f"Saved {len(df_clean_final)} cleaned rows.")
    
    if commercial_list:
        pd.DataFrame(commercial_list).to_csv(OUTPUT_COMMERCIAL_FILE, index=False)
    if missing_list:
        pd.DataFrame(missing_list).to_csv(OUTPUT_MISSING_FILE, index=False)

if __name__ == "__main__":
    main()
