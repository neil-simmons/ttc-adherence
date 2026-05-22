import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np

EQUITY_PROFILES_FILE = "neighbourhood-profiles-2021-158-model.xlsx"
EQUITY_BOUNDARIES_FILE = "Neighbourhoods - 4326.geojson"
OUTPUT_FILE = "equity_neighbourhoods.geojson"

def sanitize_key(val):
    if pd.isna(val):
        return ""
    # Strip parenthetical IDs, force lowercase, remove spaces/punctuation for a clean join
    s = str(val).split('(')[0]
    s = s.lower().replace("and", "").replace("&", "")
    return "".join(filter(str.isalnum, s))

def main():
    if not os.path.exists(EQUITY_PROFILES_FILE) or not os.path.exists(EQUITY_BOUNDARIES_FILE):
        print("Required input files are missing.")
        sys.exit(1)

    print("Loading spatial boundaries...")
    boundaries = gpd.read_file(EQUITY_BOUNDARIES_FILE)
    if boundaries.crs is None or boundaries.crs.to_epsg() != 4326:
        boundaries = boundaries.to_crs("EPSG:4326")

    # Clean join keys
    boundaries['join_key'] = boundaries['AREA_NAME'].apply(sanitize_key)

    print("Loading Census profiles spreadsheet...")
    df = pd.read_excel(EQUITY_PROFILES_FILE, sheet_name=0, header=0)

    char_col = "Characteristic" if "Characteristic" in df.columns else df.columns[4]
    char_idx = df.columns.get_loc(char_col)
    nbh_cols = list(df.columns[char_idx + 1:])

    def extract_var(keyword):
        row_mask = df[char_col].astype(str).str.contains(keyword, case=False, na=False)
        if not row_mask.any():
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

    print("Extracting variables...")
    s_income = extract_var("Median total income of household")
    
    s_lowincome = extract_var("Low-income measure, after tax")
    if s_lowincome.isna().all():
        s_lowincome = extract_var("LIM-AT")

    s_total_pop = extract_var("Population, 2021")
    s_total_hh = extract_var("Total - Private households")
    s_total_commuters = extract_var("Total - Main mode of commuting")

    # Safe divisions to prevent inf or NaN errors
    s_zerocar = extract_var("No vehicles")
    s_zerocar_pct = np.where(s_total_hh > 0, (s_zerocar / s_total_hh) * 100, np.nan)

    s_transit = extract_var("Public transit")
    s_transit_pct = np.where(s_total_commuters > 0, (s_transit / s_total_commuters) * 100, np.nan)

    s_vismin = extract_var("Total visible minority population")
    s_vismin_pct = np.where(s_total_pop > 0, (s_vismin / s_total_pop) * 100, np.nan)

    s_immigrant = extract_var("Recent immigrants")
    s_immigrant_pct = np.where(s_total_pop > 0, (s_immigrant / s_total_pop) * 100, np.nan)

    # Use the more reliable standard "65 years and over" category
    s_seniors = extract_var("65 years and over")
    s_senior_pct = np.where(s_total_pop > 0, (s_seniors / s_total_pop) * 100, np.nan)

    equity_data = pd.DataFrame({
        'raw_key':              nbh_cols,
        'median_income':        s_income,
        'low_income_pct':       s_lowincome,
        'zero_car_pct':         s_zerocar_pct,
        'transit_commute_pct':  s_transit_pct,
        'visible_minority_pct': s_vismin_pct,
        'recent_immigrant_pct': s_immigrant_pct,
        'senior_pct':           s_senior_pct,
    })
    
    equity_data['join_key'] = equity_data['raw_key'].apply(sanitize_key)

    print("Merging data tables...")
    merged = boundaries.merge(equity_data, on='join_key', how='left')
    merged['area_name'] = merged['AREA_NAME']

    final_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs="EPSG:4326")
    final_gdf.to_file(OUTPUT_FILE, driver="GeoJSON")
    print("Precomputation pipeline complete.")

if __name__ == "__main__":
    main()
