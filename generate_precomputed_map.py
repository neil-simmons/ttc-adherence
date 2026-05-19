import sys
import subprocess
import importlib

# --- AUTO-INSTALLER ---
def ensure_packages():
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'geopandas': 'geopandas',
        'shapely': 'shapely',
        'pyarrow': 'pyarrow',
        'huggingface_hub': 'huggingface-hub'
    }
    for module_name, pip_name in packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"📦 Missing {module_name}. Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ Successfully installed {pip_name}.")

ensure_packages()

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow as pa
from huggingface_hub import hf_hub_download
import json
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
HF_REPO = "neil-simmons/ttc-avl-data"
HF_REPO_TYPE = "dataset"
PARQUET_HISTORY = "ttc_all_streetcars_history.parquet"
START_DATE = '2026-03-15'
END_DATE = '2026-05-02 23:59:59'
STAT_HOLIDAYS = ['2026-04-03']
MAX_TRACK_DEVIATION_M = 150
MAX_ALLOWED_PING_GAP_SEC = 120
UTM_PROJ = "EPSG:32617"
LATLON_PROJ = "EPSG:4326"

# --- ANALYSIS PARAMETERS (ENTIRE DAY) ---
DAY_TYPE = "Weekdays"
TIME_MODE = "Overlap Mode"
FORCE_T0 = False
WINDOW_EARLY = -15
WINDOW_LATE = 120
FILTER_START_SEC = 0
FILTER_END_SEC = 28 * 3600  # Covers the complete 24-hour operating day (including past midnight)

def _hf(filename):
    print(f"Downloading {filename}...")
    return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type=HF_REPO_TYPE)

def parse_gtfs_time(time_str):
    if pd.isna(time_str): return np.nan
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def load_data():
    stops = pd.read_csv(_hf("stops.txt"), usecols=['stop_id', 'stop_name', 'stop_lat', 'stop_lon'], dtype={'stop_id': 'string[pyarrow]', 'stop_name': 'string[pyarrow]'})
    stops['stop_id'] = stops['stop_id'].astype('category')
    stops['stop_lat'] = pd.to_numeric(stops['stop_lat'])
    stops['stop_lon'] = pd.to_numeric(stops['stop_lon'])

    trips = pd.read_csv(_hf("trips.txt"), usecols=['route_id', 'trip_id', 'shape_id', 'trip_headsign'], dtype='string[pyarrow]')
    trips['trip_id'] = trips['trip_id'].str.replace(r'\.0$', '', regex=True).str.strip().astype('category')

    stop_times = pd.read_csv(_hf("stop_times.txt"), usecols=['trip_id', 'stop_id', 'arrival_time', 'stop_sequence', 'shape_dist_traveled'], dtype='string[pyarrow]')
    stop_times['trip_id'] = stop_times['trip_id'].str.replace(r'\.0$', '', regex=True).str.strip().astype('category')
    stop_times['stop_id'] = stop_times['stop_id'].astype('category')
    stop_times['shape_dist_traveled'] = pd.to_numeric(stop_times['shape_dist_traveled'], downcast='float')
    stop_times['stop_sequence'] = pd.to_numeric(stop_times['stop_sequence'], downcast='integer')

    shapes = pd.read_csv(_hf("shapes.txt"), usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'], dtype={'shape_id': 'string[pyarrow]'})
    shapes['shape_pt_lat'] = pd.to_numeric(shapes['shape_pt_lat'], downcast='float')
    shapes['shape_pt_lon'] = pd.to_numeric(shapes['shape_pt_lon'], downcast='float')
    shapes['shape_pt_sequence'] = pd.to_numeric(shapes['shape_pt_sequence'], downcast='integer')
    
    return stops, trips, stop_times, shapes

def load_route_data(path, selected_route):
    schema = pq.read_schema(path)
    route_id_type = schema.field('route_id').type
    if pa.types.is_integer(route_id_type): filter_val = [int(selected_route)]
    elif pa.types.is_floating(route_id_type): filter_val = [float(selected_route)]
    else: filter_val = [str(selected_route), f"{selected_route}.0"]

    table = ds.dataset(path, format="parquet").to_table(columns=['trip_id', 'system_time', 'latitude', 'longitude'], filter=ds.field('route_id').isin(filter_val))
    df = table.to_pandas()
    df['trip_id'] = df['trip_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().astype('category')
    df['latitude'] = df['latitude'].astype(np.float32)
    df['longitude'] = df['longitude'].astype(np.float32)
    df['system_time'] = df['system_time'].astype(np.int32)

    local_time = pd.to_datetime(df['system_time'], unit='s', utc=True).dt.tz_convert('America/Toronto')
    mask = ((local_time.dt.tz_localize(None) >= pd.to_datetime(START_DATE)) & (local_time.dt.tz_localize(None) <= pd.to_datetime(END_DATE)))
    df = df[mask].copy()
    local_time = local_time[mask]
    
    hour = local_time.dt.hour.astype(np.int32) 
    sec_since_midnight = (hour * 3600 + local_time.dt.minute.astype(np.int32) * 60 + local_time.dt.second.astype(np.int32)).astype(np.int32)

    df['op_seconds'] = np.where(hour < 4, sec_since_midnight + 86400, sec_since_midnight).astype(np.int32)
    op_date = np.where(hour < 4, (local_time - pd.Timedelta(days=1)).dt.date, local_time.dt.date)
    df['op_date'] = pd.Series(op_date).astype(str).astype('category')
    df['day_of_week'] = pd.to_datetime(df['op_date']).dt.dayofweek.astype(np.int8)
    df['is_holiday'] = df['op_date'].astype(str).isin(STAT_HOLIDAYS)
    return df

def apply_route_offset(geom, route_index, total_routes):
    """Applies a visual offset to geometries that share the same physical tracks."""
    if total_routes <= 1 or geom is None: return geom
    offset_step = 0.00008 
    offset_val = (route_index - (total_routes - 1) / 2.0) * offset_step
    
    if offset_val == 0: return geom
    
    try:
        if hasattr(geom, 'offset_curve'): return geom.offset_curve(offset_val)
        else: return geom.parallel_offset(abs(offset_val), 'left' if offset_val > 0 else 'right')
    except:
        return geom

def run_precompute():
    print("--- TTC Network Precompute Script ---")
    parquet_path = _hf(PARQUET_HISTORY)
    stops, trips, stop_times, shapes = load_data()
    
    table = pq.read_table(parquet_path, columns=['route_id'])
    available_routes = sorted(list(set(str(r).replace('.0', '').strip() for r in pc.unique(table.column('route_id')).to_pylist() if pd.notna(r) and str(r) != 'nan')))
    
    all_stops, all_segments = [], []

    for route in available_routes:
        print(f"\nProcessing Route {route}...")
        df_hist_raw = load_route_data(parquet_path, route)
        
        if DAY_TYPE == "Saturdays": day_mask = (df_hist_raw['day_of_week'] == 5) & (~df_hist_raw['is_holiday'])
        elif DAY_TYPE == "Sundays & Holidays": day_mask = (df_hist_raw['day_of_week'] == 6) | (df_hist_raw['is_holiday'])
        else: day_mask = (df_hist_raw['day_of_week'] <= 4) & (~df_hist_raw['is_holiday'])
        df_hist = df_hist_raw[day_mask]
        
        # Case-insensitive filtration excludes Short, SHORT, or short variations automatically
        directions = [d for d in trips[trips['route_id'] == route]['trip_headsign'].dropna().unique() if "short" not in str(d).lower()]
        
        for direction in directions:
            print(f"  -> Direction: {direction}")
            gtfs_route_trips = trips[(trips['route_id'] == route) & (trips['trip_headsign'] == direction)]
            valid_trips = gtfs_route_trips[gtfs_route_trips['trip_id'].isin(df_hist['trip_id'].unique())]
            if valid_trips.empty: continue

            valid_st = stop_times[stop_times['trip_id'].isin(valid_trips['trip_id'])].copy()
            valid_st = valid_st.merge(stops, on='stop_id', how='left')
            valid_st['arrival_sec'] = valid_st['arrival_time'].apply(parse_gtfs_time)
            valid_st['relative_sec'] = valid_st['arrival_sec'] - valid_st.groupby('trip_id')['arrival_sec'].transform('min')
            valid_st = valid_st.sort_values(['trip_id', 'stop_sequence'])

            sig_dict = {}
            for t_id, df_group in valid_st.groupby('trip_id'):
                sig = tuple(zip(df_group['stop_id'], df_group['relative_sec']))
                if sig: sig_dict.setdefault(sig, []).append(t_id)

            trip_start_dict = dict(zip(valid_st.groupby('trip_id').first().reset_index()['trip_id'], valid_st.groupby('trip_id').first().reset_index()['arrival_sec']))
            trip_hist_counts = df_hist.groupby('trip_id')['op_date'].nunique().to_dict()

            sig_list = []
            for sig, t_ids in sig_dict.items():
                if sum(trip_hist_counts.get(tid, 0) for tid in t_ids) == 0: continue
                starts = [trip_start_dict[tid] for tid in t_ids]
                if max(starts) < FILTER_START_SEC or min(starts) > FILTER_END_SEC: continue
                sig_list.append({'t_ids': t_ids})

            for sig in sig_list:
                trip_h = df_hist[df_hist['trip_id'].isin(sig['t_ids'])].copy()
                st_filtered = valid_st[valid_st['trip_id'] == sig['t_ids'][0]].copy().sort_values('stop_sequence')
                if st_filtered['shape_dist_traveled'].max() > 500: st_filtered['shape_dist_traveled'] /= 1000.0
                if len(st_filtered) < 2: continue

                shp_id = gtfs_route_trips[gtfs_route_trips['trip_id'] == sig['t_ids'][0]]['shape_id'].iloc[0]
                shp_pts = shapes[shapes['shape_id'] == shp_id].copy().sort_values('shape_pt_sequence')
                target_line_utm = gpd.GeoDataFrame(index=[0], crs=LATLON_PROJ, geometry=[LineString(list(zip(shp_pts['shape_pt_lon'].astype(float), shp_pts['shape_pt_lat'].astype(float))))]).to_crs(UTM_PROJ).geometry.iloc[0]

                trip_h_gdf = gpd.GeoDataFrame(trip_h, crs=LATLON_PROJ, geometry=gpd.points_from_xy(trip_h.longitude, trip_h.latitude)).to_crs(UTM_PROJ)
                trip_h = trip_h[trip_h_gdf.distance(target_line_utm) <= MAX_TRACK_DEVIATION_M].copy()
                if trip_h.empty: continue
                trip_h['official_dist_km'] = trip_h_gdf[trip_h_gdf.distance(target_line_utm) <= MAX_TRACK_DEVIATION_M].geometry.apply(lambda pt: target_line_utm.project(pt)) / 1000.0

                actual_times = {sid: [] for sid in st_filtered['stop_id']}
                for (op_date, t_id), group in trip_h.groupby(['op_date', 'trip_id']):
                    group = group.sort_values('system_time').reset_index(drop=True)
                    if len(group) < 3 or group['official_dist_km'].isna().all(): continue
                    group = group.loc[:group['official_dist_km'].idxmax()].copy()
                    group['official_dist_km'] = group['official_dist_km'].cummax()
                    group = group.drop_duplicates(subset=['official_dist_km'], keep='first')
                    if len(group) < 2: continue

                    interp = {sid: t for sid, t in zip(st_filtered['stop_id'], np.interp(st_filtered['shape_dist_traveled'].values, group['official_dist_km'].values, group['op_seconds'].values, left=np.nan, right=np.nan)) if not np.isnan(t)}
                    if not interp: continue

                    anchor_dist = st_filtered.iloc[1]['shape_dist_traveled'] if len(st_filtered) > 1 else st_filtered.iloc[0]['shape_dist_traveled']
                    if group['official_dist_km'].iloc[0] > anchor_dist: continue
                    anchor_sec = trip_start_dict.get(t_id)

                    if (TIME_MODE == "Trip Start Mode" and FILTER_START_SEC <= anchor_sec <= FILTER_END_SEC) or (TIME_MODE != "Trip Start Mode" and any(FILTER_START_SEC <= t <= FILTER_END_SEC for t in interp.values())):
                        for s_id, t in interp.items(): actual_times[s_id].append(t - anchor_sec)

                # Geometry Mapping Updates
                rel_dict, rel_vals, sample_sizes = {}, {}, {}
                for stop in st_filtered.itertuples():
                    arr = actual_times[stop.stop_id]
                    sample_sizes[stop.stop_id] = len(arr)
                    if not arr:
                        rel_vals[stop.stop_id] = 0.0
                        continue
                    hits = sum(1 for t in arr if WINDOW_EARLY <= (t - stop.relative_sec) <= WINDOW_LATE)
                    rel_vals[stop.stop_id] = (hits / len(arr)) * 100

                stops_df = st_filtered[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].copy()
                stops_df['reliability'] = stops_df['stop_id'].map(rel_vals)
                stops_df['sample_size'] = stops_df['stop_id'].map(sample_sizes)
                all_stops.append(stops_df)

                # shapely Path Tracing
                shape_coords = list(zip(shp_pts['shape_pt_lon'].astype(float), shp_pts['shape_pt_lat'].astype(float)))
                full_route_line = LineString(shape_coords) if len(shape_coords) > 1 else None

                segs = []
                for i in range(len(st_filtered) - 1):
                    s1, s2 = st_filtered.iloc[i], st_filtered.iloc[i + 1]
                    if s1.stop_lon == s2.stop_lon and s1.stop_lat == s2.stop_lat: continue
                    
                    lon1, lat1 = float(s1.stop_lon), float(s1.stop_lat)
                    lon2, lat2 = float(s2.stop_lon), float(s2.stop_lat)
                    
                    geom = None
                    if full_route_line:
                        d1 = full_route_line.project(Point(lon1, lat1))
                        d2 = full_route_line.project(Point(lon2, lat2))
                        start_d, end_d = min(d1, d2), max(d1, d2)
                        if end_d > start_d:
                            geom = substring(full_route_line, start_d, end_d)
                            
                    if geom is None or geom.is_empty:
                        geom = LineString([(lon1, lat1), (lon2, lat2)])
                        
                    segs.append({
                        'route_id': route,
                        'segment': f"{s1.stop_name} to {s2.stop_name}",
                        'avg_reliability': (rel_vals[s1.stop_id] + rel_vals[s2.stop_id]) / 2.0,
                        'geometry': geom
                    })
                if segs: all_segments.append(gpd.GeoDataFrame(segs, geometry='geometry', crs=LATLON_PROJ))

    print("\nAggregating final network...")
    master_stops = pd.concat(all_stops, ignore_index=True)
    master_stops['rel_w'] = master_stops['reliability'] * master_stops['sample_size']
    master_stops = master_stops.groupby(['stop_id', 'stop_name', 'stop_lat', 'stop_lon'], as_index=False).agg({'rel_w': 'sum', 'sample_size': 'sum'})
    master_stops['reliability'] = np.where(master_stops['sample_size'] > 0, master_stops['rel_w'] / master_stops['sample_size'], 0)
    master_stops.drop(columns=['rel_w'], inplace=True)

    master_segments = gpd.GeoDataFrame()
    if all_segments:
        master_segments = pd.concat(all_segments, ignore_index=True)
        # Group by route_id AND segment to ensure overlapping short-turns merge seamlessly, but separate routes stay distinct
        master_segments = master_segments.groupby(['route_id', 'segment'], as_index=False).agg({'avg_reliability': 'mean', 'geometry': 'first'})
        
        # APPLY OFFSET FOR PARALLEL ROUTES
        unique_routes = sorted(master_segments['route_id'].unique())
        total_routes = len(unique_routes)
        route_idx_map = {r: i for i, r in enumerate(unique_routes)}
        master_segments['geometry'] = master_segments.apply(lambda row: apply_route_offset(row['geometry'], route_idx_map[row['route_id']], total_routes), axis=1)
        master_segments = gpd.GeoDataFrame(master_segments, geometry='geometry', crs=LATLON_PROJ)

    output = {
        "stops": master_stops.to_dict(orient='records'),
        "segments": json.loads(master_segments.to_json()) if not master_segments.empty else {},
        "config": {} # Left empty to ensure app.py overrides it with fresh styling
    }
    
    with open('precomputed_network.json', 'w') as f:
        json.dump(output, f)
        
    print("Done! Saved to precomputed_network.json. Please upload this file to your Hugging Face repository.")

if __name__ == "__main__":
    run_precompute()
