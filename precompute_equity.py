import os
import sys
import subprocess

# ==============================================================================
# 0. AUTOMATIC DEPENDENCY INSTALLATION
# ==============================================================================
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "geopandas": "geopandas",
    "openpyxl": "openpyxl",
    "shapely": "shapely",
    "fiona": "fiona"
}

def install_missing_dependencies():
    missing = [import_name for import_name in REQUIRED_PACKAGES if not __import__("importlib.util").util.find_spec(import_name)]
    if missing:
        print(f"Installing missing packages: {missing}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

install_missing_dependencies()

import pandas as pd
import geopandas as gpd
import numpy as np

# ==============================================================================
# 1. CONFIGURATION & SANITIZATION
# ==============================================================================
EQUITY_PROFILES_FILE = "neighbourhood-profiles-2021-158-model.xlsx"
EQUITY_BOUNDARIES_FILE = "Neighbourhoods - 4326.geojson"
OUTPUT_FILE = "equity_neighbourhoods.geojson"

def sanitize_key(val):
    """
    Strips parenthetical numbers (e.g. '(158)'), forces lowercase, 
    and removes all spaces/hyphens to guarantee a perfect join.
    'West Humber-Clairville (1)' -> 'westhumberclairville'
    """
    if pd.isna(val): return ""
    s = str(val).split('(')[0]
    return ''.join(filter(str.isalnum, s)).lower()

def main():
    if not os.path.exists(EQUITY_PROFILES_FILE) or not os.path.exists(EQUITY_BOUNDARIES_FILE):
        print("Error: Required raw data files not found.")
        sys.exit(1)

    print("Step 1: Loading spatial boundaries...")
    boundaries = gpd.read_file(EQUITY_BOUNDARIES_FILE)
    if boundaries.crs is None or boundaries.crs.to_epsg() != 4326:
        boundaries = boundaries.to_crs("EPSG:4326")

    # Apply aggressive sanitization to GeoJSON neighbourhood names
    boundaries['join_key'] = boundaries['AREA_NAME'].apply(sanitize_key)

    print("Step 2: Loading Census profiles spreadsheet (158-Model Layout)...")
    # Load without headers so we can explicitly target Column 0 for labels and Row 0 for names
    df = pd.read_excel(EQUITY_PROFILES_FILE, sheet_name=0, header=None)

    # Neighborhood names are in row 0, starting from column 1
    raw_nbh_names = df.iloc[0, 1:].values
    
    def extract_var(pattern):
        # Search Column 0 (A) using Regex
        mask = df[0].astype(str).str.contains(pattern, case=False, na=False, regex=True)
        if not mask.any():
            return pd.Series(np.nan, index=range(len(raw_nbh_names)))
        
        # Take the first matching row, skipping the label column
        row_idx = df[mask].index[0]
        vals = df.iloc[row_idx, 1:].copy()
        
        def clean_val(x):
            if pd.isna(x): return np.nan
            if isinstance(x, (int, float)): return float(x)
            x_str = str(x).replace(",", "").strip()
            if x_str in ("x", "-", "..", "", "null", "None", "F"): return np.nan
            try: return float(x_str)
            except ValueError: return np.nan
            
        return vals.apply(clean_val).reset_index(drop=True)

    print("Step 3: Extracting variables via Regex...")
    
    s_income = extract_var(r"Median total income.*household|Median total income.*2020")
    s_lowincome = extract_var(r"Low-income measure.*after tax|LIM-AT")
    
    # Use "Total - Age groups" as the baseline population denominator
    s_total_pop = extract_var(r"Population, 2021|Total - Age groups of the population")
    s_total_hh = extract_var(r"Total - Private households|Total - Occupied private dwellings")
    s_total_commuters = extract_var(r"Total - Main mode of commuting")

    # Safe divisions to prevent Infinity/NaN math errors
    s_zerocar = extract_var(r"No vehicles|No vehicle")
    s_zerocar_pct = np.where(s_total_hh > 0, (s_zerocar / s_total_hh) * 100, np.nan)

    s_transit = extract_var(r"^Public transit$")
    s_transit_pct = np.where(s_total_commuters > 0, (s_transit / s_total_commuters) * 100, np.nan)

    s_vismin = extract_var(r"Total visible minority population")
    s_vismin_pct = np.where(s_total_pop > 0, (s_vismin / s_total_pop) * 100, np.nan)

    s_immigrant = extract_var(r"Recent immigrants|2016 to 2021")
    s_immigrant_pct = np.where(s_total_pop > 0, (s_immigrant / s_total_pop) * 100, np.nan)

    s_seniors = extract_var(r"^65 years and over$")
    s_senior_pct = np.where(s_total_pop > 0, (s_seniors / s_total_pop) * 100, np.nan)

    print("Step 4: Compiling consolidated DataFrame...")
    equity_data = pd.DataFrame({
        'raw_key':              raw_nbh_names,
        'median_income':        s_income,
        'low_income_pct':       s_lowincome,
        'zero_car_pct':         s_zerocar_pct,
        'transit_commute_pct':  s_transit_pct,
        'visible_minority_pct': s_vismin_pct,
        'recent_immigrant_pct': s_immigrant_pct,
        'senior_pct':           s_senior_pct,
    })
    
    # Apply the same aggressive sanitization to the Excel names
    equity_data['join_key'] = equity_data['raw_key'].apply(sanitize_key)

    print("Step 5: Merging data tables...")
    merged = boundaries.merge(equity_data, on='join_key', how='left')
    merged['area_name'] = merged['AREA_NAME']

    final_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs="EPSG:4326")

    print(f"Step 6: Writing clean GeoJSON output to '{OUTPUT_FILE}'...")
    final_gdf.to_file(OUTPUT_FILE, driver="GeoJSON")
    print("Success! Precomputation pipeline complete.")

if __name__ == "__main__":
    main()
