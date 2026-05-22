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
    "fiona": "fiona"  # Required backend for geopandas file writing
}

def install_missing_dependencies():
    missing = []
    for import_name, install_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(install_name)
    
    if missing:
        print(f"Missing required packages: {missing}. Installing via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("All dependencies installed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)

install_missing_dependencies()

# Safe imports after installation check
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
EQUITY_PROFILES_FILE = "neighbourhood-profiles-2021-158-model.xlsx"
EQUITY_BOUNDARIES_FILE = "Neighbourhoods - 4326.geojson"
OUTPUT_FILE = "equity_neighbourhoods.geojson"

def main():
    # Verify file existence
    for f in [EQUITY_PROFILES_FILE, EQUITY_BOUNDARIES_FILE]:
        if not os.path.exists(f):
            print(f"Error: Required file '{f}' not found in the current directory.")
            print("Please ensure both the raw Excel profiles and raw GeoJSON boundaries are present.")
            sys.exit(1)

    print("Step 1: Loading spatial boundaries...")
    boundaries = gpd.read_file(EQUITY_BOUNDARIES_FILE)
    
    # Ensure correct projection (WGS84)
    if boundaries.crs is None or boundaries.crs.to_epsg() != 4326:
        print("Reprojecting boundaries to EPSG:4326...")
        boundaries = boundaries.to_crs("EPSG:4326")

    # Generate standardized join keys (e.g., "Moss Park (73)" -> "Moss Park")
    boundaries['join_key'] = boundaries['AREA_NAME'].str.extract(r'^(.*?)\s*\(\d+\)$')[0].str.strip()

    print("Step 2: Loading Census profiles spreadsheet...")
    df = pd.read_excel(EQUITY_PROFILES_FILE, sheet_name=0, header=0)

    # Detect variable columns and neighborhood columns
    char_col = "Characteristic" if "Characteristic" in df.columns else df.columns[4]
    char_idx = df.columns.get_loc(char_col)
    nbh_cols = list(df.columns[char_idx + 1:])

    print(f"Detected {len(nbh_cols)} neighbourhood columns.")

    def extract_var(keyword):
        """Helper to find matching census variable row and clean the neighborhood values."""
        # Force string conversion on characteristic column to ensure robust regex matching
        row_mask = df[char_col].astype(str).str.contains(keyword, case=False, na=False)
        if not row_mask.any():
            print(f"Warning: No match found for variable containing '{keyword}'")
            return pd.Series(np.nan, index=nbh_cols)
        
        matched_row = df[row_mask].iloc[0]
        extracted = matched_row[nbh_cols].copy()

        def clean_val(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            
            cleaned_str = str(val).replace(",", "").strip()
            if cleaned_str in ("x", "-", "..", "", "null", "None"):
                return np.nan
            try:
                return float(cleaned_str)
            except ValueError:
                return np.nan

        return extracted.apply(clean_val)

    print("Step 3: Extracting variables...")
    s_income = extract_var("Median total income of household")
    
    # Low-income fallback structure
    s_lowincome = extract_var("Low-income measure, after tax")
    if s_lowincome.isna().all():
        s_lowincome = extract_var("LIM-AT")

    s_total_pop = extract_var("Population, 2021")
    s_total_hh = extract_var("Total - Private households")
    s_total_commuters = extract_var("Total - Main mode of commuting")

    # Zero-car percentage computation
    s_zerocar = extract_var("No vehicles")
    s_zerocar_pct = (s_zerocar / s_total_hh * 100) if not s_total_hh.isna().all() else s_zerocar

    # Transit commuting percentage computation
    s_transit = extract_var("Public transit")
    s_transit_pct = (s_transit / s_total_commuters * 100) if not s_total_commuters.isna().all() else s_transit

    # Visible minority percentage computation
    s_vismin = extract_var("Total visible minority population")
    s_vismin_pct = (s_vismin / s_total_pop * 100) if not s_total_pop.isna().all() else s_vismin

    # Recent immigrant percentage computation
    s_immigrant = extract_var("Recent immigrants")
    s_immigrant_pct = (s_immigrant / s_total_pop * 100) if not s_total_pop.isna().all() else s_immigrant

    # Seniors sum and percentage computation
    s_65_79 = extract_var("65 to 79 years")
    s_80plus = extract_var("80 years and over")
    s_seniors = s_65_79.add(s_80plus, fill_value=0)
    s_senior_pct = (s_seniors / s_total_pop * 100) if not s_total_pop.isna().all() else s_seniors

    print("Step 4: Compiling consolidated DataFrame...")
    equity_data = pd.DataFrame({
        'median_income':        s_income,
        'low_income_pct':       s_lowincome,
        'zero_car_pct':         s_zerocar_pct,
        'transit_commute_pct':  s_transit_pct,
        'visible_minority_pct': s_vismin_pct,
        'recent_immigrant_pct': s_immigrant_pct,
        'senior_pct':           s_senior_pct,
    })
    
    equity_data.index.name = 'join_key'
    equity_data = equity_data.reset_index()

    # Clean the index join keys to strip parenthetical numbers if present
    equity_data['join_key'] = equity_data['join_key'].str.extract(r'^(.*?)\s*(?:\(\d+\))?$')[0].str.strip()

    print("Step 5: Merging data tables...")
    merged = boundaries.merge(equity_data, on='join_key', how='left')
    
    # Establish standard area name and drop helper fields
    merged['area_name'] = merged['AREA_NAME']
    
    # NIA binary fallback mapping (0 if missing/not matching)
    if 'is_nia' not in merged.columns:
        merged['is_nia'] = 0

    # Ensure output is a clean GeoDataFrame
    final_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs="EPSG:4326")

    print(f"Step 6: Writing clean GeoJSON output to '{OUTPUT_FILE}'...")
    final_gdf.to_file(OUTPUT_FILE, driver="GeoJSON")
    print("Success! Precomputation pipeline complete.")

if __name__ == "__main__":
    main()
