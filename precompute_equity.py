print("Step 3: Extracting variables via Regex...")
    
    s_income = extract_var(r"Median total income.*household|Median total income.*2020")
    # Explicitly grab the prevalence percentage row, not the absolute count
    s_lowincome = extract_var(r"Prevalence of low income based on the Low-income measure.*%|LIM-AT.*%")
    
    # Use "Total - Age groups" as the baseline population denominator
    s_total_pop = extract_var(r"Population, 2021|Total - Age groups of the population")
    s_total_hh = extract_var(r"Total - Private households by household size|Total - Occupied private dwellings")
    s_total_commuters = extract_var(r"Total - Main mode of commuting")

    # Safe divisions to prevent Infinity/NaN math errors
    s_zerocar = extract_var(r"No vehicles|No vehicle")
    s_zerocar_pct = np.where(s_total_hh > 0, (s_zerocar / s_total_hh) * 100, np.nan)

    # Added \s* to ignore Excel indentation spaces
    s_transit = extract_var(r"^\s*Public transit\s*$")
    s_transit_pct = np.where(s_total_commuters > 0, (s_transit / s_total_commuters) * 100, np.nan)

    s_vismin = extract_var(r"^\s*Total visible minority population\s*$")
    s_vismin_pct = np.where(s_total_pop > 0, (s_vismin / s_total_pop) * 100, np.nan)

    s_immigrant = extract_var(r"^\s*Recent immigrants\s*$|^\s*2016 to 2021\s*$")
    s_immigrant_pct = np.where(s_total_pop > 0, (s_immigrant / s_total_pop) * 100, np.nan)

    s_seniors = extract_var(r"^\s*65 years and over\s*$")
    s_senior_pct = np.where(s_total_pop > 0, (s_seniors / s_total_pop) * 100, np.nan)
