import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring, linemerge
import plotly.graph_objects as go
import gc
import threading
import json
import datetime
import re
from huggingface_hub import hf_hub_download
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.dataset as ds
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 0. CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="TTC-ScheduleWatch",
    page_icon="🚊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# MOBILE WARNING CSS INJECTION
st.markdown("""
<style>
.mobile-warning { display: none; background-color: #ffcccc; color: #900; padding: 12px; border-left: 6px solid #DA251D; margin-bottom: 15px; font-size: 14px; border-radius: 4px; }
@media (max-width: 768px) { .mobile-warning { display: block; } }
</style>
<div class="mobile-warning">⚠️ <b>Mobile Device Detected:</b> This dashboard includes extremely dense data visualizations. Please open on a desktop computer for the best experience with the Time-Distance and Density charts.</div>
""", unsafe_allow_html=True)


HF_REPO      = "neil-simmons/ttc-avl-data"
HF_REPO_TYPE = "dataset"

PARQUET_HISTORY = "ttc_all_streetcars_history.parquet"
GTFS_STOPS      = "stops.txt"
GTFS_TRIPS      = "trips.txt"
GTFS_STOP_TIMES = "stop_times.txt"
GTFS_SHAPES     = "shapes.txt"
PRECOMPUTED_MAP = "precomputed_network.json"

START_DATE    = '2026-03-15'
END_DATE      = '2026-05-02 23:59:59'
STAT_HOLIDAYS = ['2026-04-03']

MAX_TRACK_DEVIATION_M   = 150
MAX_ALLOWED_PING_GAP_SEC = 120
UTM_PROJ   = "EPSG:32617"
LATLON_PROJ = "EPSG:4326"

TTC_RED = "#DA251D"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}

@st.cache_resource
def get_network_lock():
    return threading.Lock()

# Ensure session state defaults
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'raw_pipeline_data' not in st.session_state: st.session_state.raw_pipeline_data = None
if 'show_settings' not in st.session_state: st.session_state.show_settings = False
if 'signatures_loaded' not in st.session_state: st.session_state.signatures_loaded = False
if 'signature_list' not in st.session_state: st.session_state.signature_list = []
if 'trip_start_dict' not in st.session_state: st.session_state.trip_start_dict = {}

# ==============================================================================
# 2. DATA LOADERS
# ==============================================================================
def _hf(filename):
    return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type=HF_REPO_TYPE)

@st.cache_resource(show_spinner="Connecting to AVL data source...")
def get_parquet_path(): return _hf(PARQUET_HISTORY)

@st.cache_resource(show_spinner="Downloading GTFS stops...")
def get_stops_path(): return _hf(GTFS_STOPS)

@st.cache_resource(show_spinner="Downloading GTFS trips...")
def get_trips_path(): return _hf(GTFS_TRIPS)

@st.cache_resource(show_spinner="Downloading GTFS stop times...")
def get_stop_times_path(): return _hf(GTFS_STOP_TIMES)

@st.cache_resource(show_spinner="Downloading GTFS shapes...")
def get_shapes_path(): return _hf(GTFS_SHAPES)

@st.cache_data(show_spinner="Indexing available routes...")
def get_available_routes(path):
    table = pq.read_table(path, columns=['route_id'])
    unique_arr = pc.unique(table.column('route_id')).to_pylist()
    cleaned = set()
    for r in unique_arr:
        if pd.isna(r): continue
        r_str = str(r).replace('.0', '').strip()
        if r_str and r_str != 'nan': cleaned.add(r_str)
    return sorted(list(cleaned))

@st.cache_data(show_spinner="Generating Global Route Dictionary...")
def get_all_route_directions(_trips, available_routes):
    options = []
    for r in available_routes:
        dirs = _trips[_trips['route_id'] == r]['trip_headsign'].dropna().unique()
        for d in dirs:
            if "short" in str(d).lower():
                continue
            options.append(f"{r} | {d}")
    return options

@st.cache_data(max_entries=1, show_spinner="Extracting route data...")
def load_route_data(path, selected_route):
    schema = pq.read_schema(path)
    route_id_type = schema.field('route_id').type

    if pa.types.is_integer(route_id_type): filter_val = [int(selected_route)]
    elif pa.types.is_floating(route_id_type): filter_val = [float(selected_route)]
    else: filter_val = [str(selected_route), f"{selected_route}.0"]

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=['trip_id', 'system_time', 'latitude', 'longitude'],
        filter=ds.field('route_id').isin(filter_val)
    )
    df = table.to_pandas()

    df['trip_id']     = df['trip_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().astype('category')
    df['latitude']    = df['latitude'].astype(np.float32)
    df['longitude']   = df['longitude'].astype(np.float32)
    df['system_time'] = df['system_time'].astype(np.int32)

    local_time = pd.to_datetime(df['system_time'], unit='s', utc=True).dt.tz_convert('America/Toronto')
    mask = (
        (local_time.dt.tz_localize(None) >= pd.to_datetime(START_DATE)) &
        (local_time.dt.tz_localize(None) <= pd.to_datetime(END_DATE))
    )
    df         = df[mask].copy()
    local_time = local_time[mask]

    hour             = local_time.dt.hour.astype(np.int32)
    sec_since_midnight = (hour * 3600 + local_time.dt.minute * 60 + local_time.dt.second).astype(np.int32)

    df['op_seconds']  = np.where(hour < 4, sec_since_midnight + 86400, sec_since_midnight).astype(np.int32)
    op_date           = np.where(hour < 4, (local_time - pd.Timedelta(days=1)).dt.date, local_time.dt.date)
    df['op_date']     = pd.Series(op_date).astype(str).astype('category')
    df['day_of_week'] = pd.to_datetime(df['op_date']).dt.dayofweek.astype(np.int8)
    df['is_holiday']  = df['op_date'].astype(str).isin(STAT_HOLIDAYS)

    gc.collect()
    return df

@st.cache_data(show_spinner="Loading static GTFS data...")
def load_gtfs():
    str_dtype = 'string[pyarrow]'
    stops = pd.read_csv(get_stops_path(), usecols=['stop_id', 'stop_name', 'stop_lat', 'stop_lon'], 
                        dtype={'stop_id': str_dtype, 'stop_name': str_dtype})
    stops['stop_id'] = stops['stop_id'].astype('category')
    stops['stop_lat'] = pd.to_numeric(stops['stop_lat'], downcast='float')
    stops['stop_lon'] = pd.to_numeric(stops['stop_lon'], downcast='float')

    trips = pd.read_csv(get_trips_path(), usecols=['route_id', 'trip_id', 'shape_id', 'trip_headsign'],
                        dtype={'route_id': str_dtype, 'trip_id': str_dtype, 'shape_id': str_dtype, 'trip_headsign': 'category'})
    trips['trip_id'] = trips['trip_id'].str.replace(r'\.0$', '', regex=True).str.strip().astype('category')

    stop_times = pd.read_csv(get_stop_times_path(), usecols=['trip_id', 'stop_id', 'arrival_time', 'stop_sequence', 'shape_dist_traveled'],
                             dtype={'trip_id': str_dtype, 'stop_id': str_dtype, 'arrival_time': str_dtype})
    stop_times['trip_id']            = stop_times['trip_id'].str.replace(r'\.0$', '', regex=True).str.strip().astype('category')
    stop_times['stop_id']            = stop_times['stop_id'].astype('category')
    stop_times['shape_dist_traveled'] = pd.to_numeric(stop_times['shape_dist_traveled'], downcast='float')
    stop_times['stop_sequence']       = pd.to_numeric(stop_times['stop_sequence'], downcast='integer')

    shapes = pd.read_csv(get_shapes_path(), usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'], dtype={'shape_id': str_dtype})
    shapes['shape_pt_lat']      = pd.to_numeric(shapes['shape_pt_lat'], downcast='float')
    shapes['shape_pt_lon']      = pd.to_numeric(shapes['shape_pt_lon'], downcast='float')
    shapes['shape_pt_sequence'] = pd.to_numeric(shapes['shape_pt_sequence'], downcast='integer')

    gc.collect()
    return stops, trips, stop_times, shapes

# ==============================================================================
# 3. HELPER FUNCTIONS & KEPLER CONFIG
# ==============================================================================
def parse_gtfs_time(time_str):
    if pd.isna(time_str): return np.nan
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def format_seconds_to_time(seconds):
    if pd.isna(seconds) or seconds < 0: return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    display_h = h % 24
    ampm = "AM" if display_h < 12 else "PM"
    display_h = 12 if display_h in (0, 12) else display_h % 12
    return f"{display_h:02d}:{m:02d} {ampm}"

def clean_stop_name(name):
    name = re.sub(r'(?i)\s+(East|West|North|South)\s+Side', '', name)
    name = name.split(' - ')[0].strip()
    name = name.split(' at ')[-1].strip() if ' at ' in name and len(name) > 35 else name
    if len(name) > 45:
        name = name[:42] + "..."
    return name

def load_precomputed_network():
    try:
        path = _hf(PRECOMPUTED_MAP)
        with open(path, 'r') as f: return json.load(f)
    except Exception: return None 

def inject_legend_anchors(stops_df, segments_df):
    if stops_df.empty or segments_df.empty: return stops_df, segments_df
    
    # Clone stop anchors and set coordinates to NaN so they are ignored by the renderer
    d_stop_0, d_stop_100 = stops_df.iloc[0].copy(), stops_df.iloc[0].copy()
    d_stop_0['reliability'], d_stop_0['sample_size'], d_stop_0['stop_lat'], d_stop_0['stop_lon'] = 0.0, 0, np.nan, np.nan
    d_stop_100['reliability'], d_stop_100['sample_size'], d_stop_100['stop_lat'], d_stop_100['stop_lon'] = 100.0, 0, np.nan, np.nan
    s_df = pd.concat([stops_df, pd.DataFrame([d_stop_0, d_stop_100])], ignore_index=True)
    
    # Clone segment anchors and set geometry to None (null GeoJSON geometry)
    d_seg_0, d_seg_100 = segments_df.iloc[0].copy(), segments_df.iloc[0].copy()
    d_seg_0['avg_reliability'] = 0.0
    d_seg_0['geometry'] = None
    
    d_seg_100['avg_reliability'] = 100.0
    d_seg_100['geometry'] = None
    
    seg_df = pd.concat([segments_df, gpd.GeoDataFrame([d_seg_0, d_seg_100], geometry='geometry', crs=LATLON_PROJ)], ignore_index=True)
    return s_df, seg_df

def generate_kepler_config():
    custom_20_colors = [
        "#DA251D", "#E03920", "#E54E23", "#EB6326", "#F07729", 
        "#F58C2C", "#FBA02F", "#FFB532", "#FFCA35", "#FFDE38", 
        "#F2E43B", "#D8DB3D", "#BED240", "#A3C942", "#89C045", 
        "#6FB747", "#54AE4A", "#3AA54C", "#209C4F", "#1A9641"
    ]
    color_scale_config = {"name": "TTC_Scale", "type": "custom", "category": "Custom", "colors": custom_20_colors}
    return {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                        "id": "segments", "type": "geojson",
                        "config": {
                            "dataId": "segments", "label": "Route Segments", "columns": {"geojson": "geometry"}, "isVisible": True,
                            "visConfig": {"opacity": 1.0, "strokeOpacity": 1.0, "thickness": 1.0, "strokeColor": None, "colorRange": color_scale_config, "strokeColorRange": color_scale_config}
                        },
                        "visualChannels": {"colorField": {"name": "avg_reliability", "type": "real"}, "colorScale": "quantize", "strokeColorField": {"name": "avg_reliability", "type": "real"}, "strokeColorScale": "quantize"}
                    },
                    {
                        "id": "stops", "type": "point",
                        "config": {
                            "dataId": "stops", "label": "Stops", "columns": {"lat": "stop_lat", "lng": "stop_lon"}, "isVisible": True,
                            "visConfig": {"radiusRange": [3, 9], "opacity": 1.0, "filled": True, "outline": True, "thickness": 1.5, "strokeColor": [255, 255, 255], "colorRange": color_scale_config}
                        },
                        "visualChannels": {"colorField": {"name": "reliability", "type": "real"}, "colorScale": "quantize", "sizeField": {"name": "sample_size", "type": "integer"}, "sizeScale": "linear"}
                    }
                ],
                "layerOrder": ["segments", "stops"],  # Stops render on top of segments
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "segments": [{"name": "segment", "format": None}, {"name": "route_id", "format": None}, {"name": "avg_reliability", "format": ".1f"}],
                            "stops": [{"name": "stop_name", "format": None}, {"name": "route_id", "format": None}, {"name": "reliability", "format": ".1f"}]
                        },
                        "enabled": True
                    }
                }
            },
            "mapStyle": {"styleType": "muted_night"}
        }
    }

# ==============================================================================
# 4. MODULARIZED PIPELINE FUNCTIONS
# ==============================================================================
def apply_day_filters(df, days_selected):
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    selected_dow = [day_mapping[d] for d in days_selected if d != "Holiday"]
    
    day_mask = df['day_of_week'].isin(selected_dow)
    
    if "Holiday" in days_selected:
        day_mask = day_mask | df['is_holiday']
    else:
        day_mask = day_mask & (~df['is_holiday'])
        
    return df[day_mask]

def get_route_signatures(df_hist_raw, valid_trips, stop_times, stops, filter_start_sec, filter_end_sec, days_selected):
    df_hist_filtered = apply_day_filters(df_hist_raw, days_selected)
    
    valid_st = stop_times[stop_times['trip_id'].isin(valid_trips['trip_id'])].copy()
    valid_st = valid_st.merge(stops, on='stop_id', how='left')
    valid_st['arrival_sec'] = valid_st['arrival_time'].apply(parse_gtfs_time)
    start_times_series = valid_st.groupby('trip_id')['arrival_sec'].transform('min')
    valid_st['relative_sec'] = valid_st['arrival_sec'] - start_times_series
    valid_st = valid_st.sort_values(['trip_id', 'stop_sequence'])

    signatures_dict = {}
    for t_id, df_group in valid_st.groupby('trip_id', observed=True):
        sig = tuple(zip(df_group['stop_id'], df_group['relative_sec']))
        if not sig: continue
        if sig not in signatures_dict: signatures_dict[sig] = []
        signatures_dict[sig].append(t_id)

    first_stops = valid_st.groupby('trip_id', observed=True).first().reset_index()
    last_stops = valid_st.groupby('trip_id', observed=True).last().reset_index()
    trip_start_dict = dict(zip(first_stops['trip_id'], first_stops['arrival_sec']))
    trip_orig_dict = dict(zip(first_stops['trip_id'], first_stops['stop_name']))
    trip_dest_dict = dict(zip(last_stops['trip_id'], last_stops['stop_name']))
    
    trip_hist_counts = df_hist_filtered.groupby('trip_id', observed=True)['op_date'].nunique().to_dict()

    sig_ui_list = []
    for sig, t_ids in signatures_dict.items():
        hist_run_count = sum(trip_hist_counts.get(tid, 0) for tid in t_ids)
        if hist_run_count == 0: continue
        start_secs = [trip_start_dict[tid] for tid in t_ids]
        min_s, max_s = min(start_secs), max(start_secs)
        if max_s < filter_start_sec or min_s > filter_end_sec: continue
        
        sig_ui_list.append({
            'signature': sig, 't_ids': t_ids, 'orig': trip_orig_dict[t_ids[0]], 
            'dest': trip_dest_dict[t_ids[0]], 'stops': len(sig), 
            'min_sec': min_s, 'max_sec': max_s, 'runs': hist_run_count
        })

    return sorted(sig_ui_list, key=lambda x: x['min_sec']), trip_start_dict

def run_tracking(df_hist_raw, matching_trip_ids, s2_vars, stop_times, stops, gtfs_route_trips, shapes):
    df_hist_filtered = apply_day_filters(df_hist_raw, s2_vars['days_selected'])
    trip_hist        = df_hist_filtered[df_hist_filtered['trip_id'].isin(matching_trip_ids)].copy()

    valid_st_stage2 = stop_times[stop_times['trip_id'].isin(matching_trip_ids)].copy()
    valid_st_stage2 = valid_st_stage2.merge(stops, on='stop_id', how='left')
    valid_st_stage2['arrival_sec']  = valid_st_stage2['arrival_time'].apply(parse_gtfs_time)
    start_times_s2                  = valid_st_stage2.groupby('trip_id', observed=True)['arrival_sec'].transform('min')
    valid_st_stage2['relative_sec'] = valid_st_stage2['arrival_sec'] - start_times_s2

    sample_trip = matching_trip_ids[0]
    st_filtered = valid_st_stage2[valid_st_stage2['trip_id'] == sample_trip].copy().sort_values('stop_sequence')
    if st_filtered['shape_dist_traveled'].max() > 500: st_filtered['shape_dist_traveled'] /= 1000.0

    if s2_vars['stop_filter_ids']:
        st_filtered = st_filtered[st_filtered['stop_id'].isin(s2_vars['stop_filter_ids'])]

    if len(st_filtered) < 2: return None

    # Track corridor minimum and maximum bound values
    min_dist = st_filtered['shape_dist_traveled'].min()
    max_dist = st_filtered['shape_dist_traveled'].max()

    sample_shape_id = gtfs_route_trips[gtfs_route_trips['trip_id'] == sample_trip]['shape_id'].iloc[0]
    shp_pts         = shapes[shapes['shape_id'] == sample_shape_id].copy().sort_values('shape_pt_sequence')
    line_coords     = list(zip(shp_pts['shape_pt_lon'].astype(float), shp_pts['shape_pt_lat'].astype(float)))
    target_line_utm = gpd.GeoDataFrame(index=[0], crs=LATLON_PROJ, geometry=[LineString(line_coords)]).to_crs(UTM_PROJ).geometry.iloc[0]

    trip_hist_gdf = gpd.GeoDataFrame(trip_hist, crs=LATLON_PROJ, geometry=gpd.points_from_xy(trip_hist.longitude, trip_hist.latitude)).to_crs(UTM_PROJ)
    trip_hist['dist_to_track_m'] = trip_hist_gdf.distance(target_line_utm)
    valid_mask = trip_hist['dist_to_track_m'] <= MAX_TRACK_DEVIATION_M
    trip_hist  = trip_hist[valid_mask].copy()

    if trip_hist.empty: return None

    trip_hist_gdf = trip_hist_gdf[valid_mask].copy()
    trip_hist['official_dist_km'] = trip_hist_gdf.geometry.apply(lambda pt: target_line_utm.project(pt)) / 1000.0

    actual_relative_times = {stop_id: [] for stop_id in st_filtered['stop_id']}
    mode_b_lines          = []

    for (op_date, t_id), group in trip_hist.groupby(['op_date', 'trip_id'], observed=True):
        group = group.sort_values('system_time').reset_index(drop=True)
        if len(group) < 3: continue

        gtfs_start_sec = s2_vars['trip_start_dict'].get(t_id)
        if gtfs_start_sec is None or group['official_dist_km'].isna().all(): continue

        max_dist_idx = group['official_dist_km'].idxmax()
        group = group.loc[:max_dist_idx].copy()
        group['official_dist_km'] = group['official_dist_km'].cummax()
        group = group.drop_duplicates(subset=['official_dist_km'], keep='first')

        # Filter points specifically to the selected corridor range
        group = group[(group['official_dist_km'] >= min_dist) & (group['official_dist_km'] <= max_dist)].copy()
        if len(group) < 2: continue

        interpolated_times = np.interp(st_filtered['shape_dist_traveled'].values, group['official_dist_km'].values, group['op_seconds'].values, left=np.nan, right=np.nan)
        run_interpolations = {sid: t for sid, t in zip(st_filtered['stop_id'], interpolated_times) if not np.isnan(t)}
        if not run_interpolations: continue

        anchor_stop = st_filtered.iloc[1] if len(st_filtered) > 1 else st_filtered.iloc[0]
        anchor_stop_dist = anchor_stop['shape_dist_traveled']

        if group['official_dist_km'].iloc[0] > anchor_stop_dist: continue

        if s2_vars['force_t0']:
            anchor_stop_id = anchor_stop['stop_id']
            if anchor_stop_id not in run_interpolations: continue
            idx_after = np.searchsorted(group['official_dist_km'].values, anchor_stop_dist)
            if idx_after == 0 or idx_after >= len(group): continue
            time_gap = group['op_seconds'].iloc[idx_after] - group['op_seconds'].iloc[idx_after - 1]
            if time_gap > MAX_ALLOWED_PING_GAP_SEC: continue
            anchor_sec = run_interpolations[anchor_stop_id] - anchor_stop['relative_sec']
        else:
            anchor_sec = gtfs_start_sec

        f_start, f_end = s2_vars['filter_start_sec'], s2_vars['filter_end_sec']
        if "Trip Start Mode" in s2_vars['time_mode']: is_valid = f_start <= anchor_sec <= f_end
        else: is_valid = any(f_start <= t <= f_end for t in run_interpolations.values())

        if is_valid:
            for s_id, t in run_interpolations.items(): actual_relative_times[s_id].append(t - anchor_sec)

            dist_diff = group['official_dist_km'].diff()
            time_diff = group['system_time'].diff()
            group['prev_speed_kmh'] = np.where(time_diff > 0, (dist_diff / time_diff) * 3600, 0).clip(min=0)
            group['relative_min'] = (group['op_seconds'] - anchor_sec) / 60.0

            # Trace sequence and inject None element where sequential pings break tracking bounds
            raw_x = group['relative_min'].tolist()
            raw_y = group['official_dist_km'].tolist()
            raw_lat = group['latitude'].tolist()
            raw_lon = group['longitude'].tolist()
            raw_speed = group['prev_speed_kmh'].tolist()
            raw_systime = group['system_time'].tolist()

            raw_abs = (pd.to_datetime(group['system_time'], unit='s', utc=True)
                       .dt.tz_convert('America/Toronto')
                       .dt.strftime('%I:%M:%S %p').tolist())

            x_gaps, y_gaps, abs_gaps, lat_gaps, lon_gaps, speed_gaps = [], [], [], [], [], []

            for i in range(len(raw_x)):
                if i > 0 and (raw_systime[i] - raw_systime[i-1]) > MAX_ALLOWED_PING_GAP_SEC:
                    # Injected None breaks the Plotly line visualization when gaps are too large
                    x_gaps.append(None)
                    y_gaps.append(None)
                    abs_gaps.append(None)
                    lat_gaps.append(None)
                    lon_gaps.append(None)
                    speed_gaps.append(None)

                x_gaps.append(raw_x[i])
                y_gaps.append(raw_y[i])
                abs_gaps.append(raw_abs[i])
                lat_gaps.append(raw_lat[i])
                lon_gaps.append(raw_lon[i])
                speed_gaps.append(raw_speed[i])

            mode_b_lines.append({
                'name': f"{op_date} | {t_id}", 'op_date': str(op_date), 'start_time': format_seconds_to_time(list(run_interpolations.values())[0]),
                't_id': str(t_id), 'x': x_gaps, 'y': y_gaps,
                'abs_time': abs_gaps, 'lat': lat_gaps, 'lon': lon_gaps, 'speed': speed_gaps
            })
            
    if not mode_b_lines: return None
    return {'st_filtered': st_filtered, 'actual_relative_times': actual_relative_times, 'mode_b_lines': mode_b_lines, 'shape_id': sample_shape_id}

def build_spatial_data(st_filtered, actual_relative_times, window_early, window_late, shapes, shape_id, route_id, route_idx):
    reliability_dict, reliability_vals, sample_sizes = {}, {}, {}
    for stop in st_filtered.itertuples():
        arr = actual_relative_times[stop.stop_id]
        sample_sizes[stop.stop_id] = len(arr)
        if not arr:
            reliability_dict[stop.stop_id], reliability_vals[stop.stop_id] = "N/A", 0.0
            continue
        sched_sec = stop.relative_sec
        hits = sum(1 for t in arr if window_early <= (t - sched_sec) <= window_late)
        pct  = (hits / len(arr)) * 100
        reliability_dict[stop.stop_id], reliability_vals[stop.stop_id] = f"{pct:.1f}%", pct

    shp_pts = shapes[shapes['shape_id'] == shape_id].sort_values('shape_pt_sequence')
    shape_coords = list(zip(shp_pts['shape_pt_lon'].astype(float), shp_pts['shape_pt_lat'].astype(float)))
    full_route_line = LineString(shape_coords) if len(shape_coords) > 1 else None

    if full_route_line:
        offset_meters = 5.0 + (route_idx * 8.0)
        try:
            gs = gpd.GeoSeries([full_route_line], crs=LATLON_PROJ).to_crs(UTM_PROJ)
            gs_offset = gs.geometry.apply(lambda geom: geom.parallel_offset(offset_meters, 'right', join_style=2))
            if not gs_offset.empty and not gs_offset.iloc[0].is_empty:
                offset_geom = gs_offset.to_crs(LATLON_PROJ).iloc[0]
                if offset_geom.geom_type == 'MultiLineString':
                    offset_geom = linemerge(offset_geom)
                    if offset_geom.geom_type == 'MultiLineString': offset_geom = max(offset_geom.geoms, key=lambda x: x.length)
                full_route_line = offset_geom
        except Exception: pass

    offset_stops = []
    for stop in st_filtered.itertuples():
        orig_pt = Point(float(stop.stop_lon), float(stop.stop_lat))
        if full_route_line and not full_route_line.is_empty:
            proj_dist = full_route_line.project(orig_pt)
            new_pt = full_route_line.interpolate(proj_dist)
            new_lon, new_lat = new_pt.x, new_pt.y
        else: new_lon, new_lat = orig_pt.x, orig_pt.y
            
        offset_stops.append({'route_id': route_id, 'stop_id': stop.stop_id, 'stop_name': stop.stop_name, 'stop_lat': new_lat, 'stop_lon': new_lon, 'reliability': reliability_vals[stop.stop_id], 'sample_size': sample_sizes[stop.stop_id]})
    stops_df = pd.DataFrame(offset_stops)

    segments = []
    for i in range(len(st_filtered) - 1):
        s1, s2 = st_filtered.iloc[i], st_filtered.iloc[i + 1]
        if s1.stop_lon == s2.stop_lon and s1.stop_lat == s2.stop_lat: continue
        geom = None
        if full_route_line and not full_route_line.is_empty:
            d1 = full_route_line.project(Point(float(s1.stop_lon), float(s1.stop_lat)))
            d2 = full_route_line.project(Point(float(s2.stop_lon), float(s2.stop_lat)))
            start_d, end_d = min(d1, d2), max(d1, d2)
            if end_d > start_d: geom = substring(full_route_line, start_d, end_d)
        if geom is None or geom.is_empty: geom = LineString([(float(s1.stop_lon), float(s1.stop_lat)), (float(s2.stop_lon), float(s2.stop_lat))])
        segments.append({'route_id': route_id, 'segment': f"{s1.stop_name} to {s2.stop_name}", 'avg_reliability': (reliability_vals[s1.stop_id] + reliability_vals[s2.stop_id]) / 2.0, 'geometry': geom})
        
    segments_df = gpd.GeoDataFrame(segments, geometry='geometry', crs=LATLON_PROJ) if segments else gpd.GeoDataFrame()
    return stops_df, segments_df, reliability_dict, reliability_vals

# ==============================================================================
# 5. EXECUTION PIPELINES
# ==============================================================================
def execute_single_route_pipeline(parquet_path, selected_route, selected_dir, s2_vars, gtfs_route_trips, stop_times, stops, shapes):
    df_hist_raw = load_route_data(parquet_path, selected_route)
    
    base_trips = s2_vars['signature_t_ids']
    if s2_vars['isolated_trips']: 
        trip_list = [t for t in base_trips if t in s2_vars['isolated_trips']]
    else: 
        trip_list = base_trips
        
    if not trip_list:
        st.error("No valid trips remain after applying filters.")
        return False
    
    raw_data = run_tracking(df_hist_raw, trip_list, s2_vars, stop_times, stops, gtfs_route_trips, shapes)
    
    if not raw_data:
        st.error("Tracking failed or no historical data found for these scheduled trips.")
        return False

    t_str = f"Route {selected_route} {selected_dir} | {s2_vars['sig_desc']} | {s2_vars['days_summary']} | {s2_vars['time_range_str']} | Mode: {s2_vars['time_mode']} | Window: {s2_vars['window_early']}s to +{s2_vars['window_late']}s"
    if s2_vars['force_t0']: t_str += " | t=0 Aligned"
    
    if s2_vars.get('stop_filter_ids') and len(s2_vars['stop_filter_ids']) < s2_vars.get('total_route_stops', 999):
        t_str += f" | Corridor: {len(s2_vars['stop_filter_ids'])}/{s2_vars['total_route_stops']} Stops"
        
    if s2_vars.get('isolated_trips'): 
        t_str += f" | Trip IDs: {', '.join(s2_vars['isolated_trips'])}" if len(s2_vars['isolated_trips']) <= 4 else f" | Filtered to {len(s2_vars['isolated_trips'])} Specific Trips"
            
    raw_data['title_info'] = t_str
    st.session_state.raw_pipeline_data = raw_data
    
    stops_df, segments_df, reliability_dict, reliability_vals = build_spatial_data(
        raw_data['st_filtered'], raw_data['actual_relative_times'], s2_vars['window_early'], s2_vars['window_late'], shapes, raw_data['shape_id'], selected_route, 0
    )
    stops_df, segments_df = inject_legend_anchors(stops_df, segments_df)
    
    # -------------------------------------------------------------
    # SMART Y-TICK TRUNCATION & DENSITY CHART GENERATION
    # -------------------------------------------------------------
    
    # Calculate zero-overlap limits dynamically based on corridor stop spacing
    dists = raw_data['st_filtered']['shape_dist_traveled'].values
    if len(dists) > 1:
        diffs = np.diff(np.sort(dists))
        valid_diffs = diffs[diffs > 0.01] # Ignore extremely tight GPS overlaps
        global_min_gap = np.min(valid_diffs) if len(valid_diffs) > 0 else 0.2
    else:
        global_min_gap = 0.2
        
    box_offset = global_min_gap * 0.25
    violin_max_height = global_min_gap * 0.65
    violin_plotly_width = violin_max_height * 2 # Plotly's internal width representation for 1-sided violins
    
    y_tick_texts = []
    for _, row in raw_data['st_filtered'].iterrows():
        clean_name = clean_stop_name(row['stop_name'])
        y_tick_texts.append(f"{clean_name} ({row['shape_dist_traveled']:.1f}km) | Rel: {reliability_dict[row['stop_id']]}")
        
    fig_A = go.Figure()
    for stop in raw_data['st_filtered'].itertuples():
        offsets_arr = raw_data['actual_relative_times'][stop.stop_id]
        N = len(offsets_arr)
        if N == 0: continue
        times_min = [round(t / 60.0, 1) for t in offsets_arr]
        c_base, c_fill, c_box = (TTC_RED, 'rgba(218,37,29,0.4)', 'rgba(218,37,29,0.1)') if N < 10 else ('goldenrod', 'rgba(218,165,32,0.4)', 'rgba(218,165,32,0.1)') if N < 25 else ('#1f77b4', 'rgba(31,119,180,0.4)', 'rgba(31,119,180,0.1)')
        
        clean_name = clean_stop_name(stop.stop_name)
        
        # Explicitly formatted hover template to fix the "Trace X" confusion
        density_hover_template = (
            f"<b>{stop.stop_name}</b><br>"
            f"Sample Size: {N} runs<br>"
            "Rel Time: %{x:.1f} mins<extra></extra>"
        )
        
        fig_A.add_trace(go.Violin(
            x=times_min, 
            y=np.repeat(stop.shape_dist_traveled, N), 
            name=clean_name, 
            orientation='h', 
            side='positive', 
            scalemode='count', 
            spanmode='hard', # 'hard' strictly bounds the curve to the min and max data points (no overhang)
            width=violin_plotly_width, # Applies Global Ceiling zero-overlap logic
            line_color=c_base, fillcolor=c_fill, showlegend=False, points=False, box_visible=False,
            hovertemplate=density_hover_template
        ))
        fig_A.add_trace(go.Box(
            x=times_min, 
            y=np.repeat(stop.shape_dist_traveled - box_offset, N), # Applies Box dynamic offset lane
            name=clean_name, 
            orientation='h', width=(global_min_gap * 0.15),
            line_color=c_base, fillcolor=c_box, boxpoints='outliers', showlegend=False,
            hovertemplate=density_hover_template
        ))

    # -------------------------------------------------------------
    # BUILD FIG B (TIME-DISTANCE) WITH EXPLICIT HOVER TEMPLATES
    # -------------------------------------------------------------
    fig_B = go.Figure()
    
    hover_tmpl = (
        "<b>Date:</b> %{customdata[0]}<br>"
        "<b>Trip ID:</b> %{customdata[2]}<br>"
        "<b>Abs Time:</b> %{customdata[3]}<br>"
        "<b>Distance:</b> %{y:.2f} km<br>"
        "<b>Rel Time:</b> %{x:.1f} mins<br>"
        "<i>(Click point to get Google Maps link)</i>"
        "<extra></extra>" 
    )

    for line_data in raw_data['mode_b_lines']:
        cd = np.empty((len(line_data['x']), 6), dtype=object)
        cd[:, 0], cd[:, 1], cd[:, 2], cd[:, 3], cd[:, 4], cd[:, 5] = line_data['op_date'], line_data['start_time'], line_data['t_id'], line_data['abs_time'], line_data['lat'], line_data['lon']
        fig_B.add_trace(go.Scattergl(
            x=line_data['x'], 
            y=line_data['y'], 
            mode='lines+markers', 
            line=dict(width=0.3), 
            marker=dict(size=1.5), 
            opacity=1.0, 
            connectgaps=False, 
            name=line_data['name'], 
            customdata=cd,
            hovertemplate=hover_tmpl
        ))

    sched_sample_sizes = [len(raw_data['actual_relative_times'][stop_id]) for stop_id in raw_data['st_filtered']['stop_id']]

    sched_trace = go.Scattergl(
        x=raw_data['st_filtered']['relative_sec'] / 60.0, 
        y=raw_data['st_filtered']['shape_dist_traveled'], 
        mode='lines+markers', 
        line=dict(color='#000000', width=1.4), 
        marker=dict(symbol='circle', size=4.5, color='#000000'), 
        name="Scheduled Baseline",
        customdata=sched_sample_sizes,
        hovertemplate="<b>Scheduled Baseline</b><br>Distance: %{y:.2f} km<br>Rel Time: %{x:.1f} mins<br>Sample Size: %{customdata} runs<extra></extra>"
    )
    # Add baseline trace LAST so it renders on top of the density curves
    fig_A.add_trace(sched_trace)
    fig_B.add_trace(sched_trace)
    
    common_layout = dict(
        height=900, 
        yaxis_title="Official Track Distance (km) & Stops [On-Time Reliability %]",
        xaxis_title="Relative Time (Minutes)",
        template="plotly_white", 
        margin=dict(r=260, t=70, b=50), 
        xaxis=dict(automargin=True),
        yaxis=dict(
            automargin=True, 
            tickmode='array', 
            tickvals=raw_data['st_filtered']['shape_dist_traveled'], 
            ticktext=y_tick_texts
        ),
        legend=dict(
            title="Trips (Double-Click to Isolate)",
            x=1.0, 
            xanchor="left",
            y=1.0,
            yanchor="top"
        ),
        dragmode="zoom"  # Standard zoom-box selected initially
    )
    
    fig_A.update_layout(**common_layout, title=f"{t_str} — Density", violinmode='overlay', boxmode='overlay')
    fig_B.update_layout(**common_layout, title=f"{t_str} — Time-Distance", hovermode='closest')

    st.session_state.analysis_results = {'is_multi': False, 'fig_A': fig_A, 'fig_B': fig_B, 'stops_df': stops_df, 'segments_df': segments_df, 'kepler_config': generate_kepler_config()}
    return True

def execute_multi_route_pipeline(selected_combos, parquet_path, trips, stop_times, stops, shapes, s2_vars):
    lock = get_network_lock()
    if not lock.acquire(blocking=False):
        st.error("⚠️ The server is currently processing a heavy network-wide calculation for another user. Please wait a moment and try again.")
        return False
        
    try:
        all_stops, all_segments = [], []
        progress_bar = st.progress(0)
        unique_routes = sorted(list(set([c.split(' | ')[0] for c in selected_combos])))
        route_idx_map = {r: i for i, r in enumerate(unique_routes)}
        
        for i, selection in enumerate(selected_combos):
            progress_bar.progress((i) / len(selected_combos), text=f"Processing {selection}...")
            route, direction = selection.split(" | ")
            route_idx = route_idx_map[route]
            
            gtfs_route_trips = trips[(trips['route_id'] == route) & (trips['trip_headsign'] == direction)]
            df_hist = load_route_data(parquet_path, route)
            
            sig_list, trip_start_dict = get_route_signatures(
                df_hist, gtfs_route_trips, stop_times, stops, s2_vars['filter_start_sec'], s2_vars['filter_end_sec'], s2_vars['days_selected']
            )
            if not sig_list: continue
            s2_vars['trip_start_dict'] = trip_start_dict
            
            for sig in sig_list:
                raw_data = run_tracking(df_hist, sig['t_ids'], s2_vars, stop_times, stops, gtfs_route_trips, shapes)
                if not raw_data: continue
                
                stops_df, segments_df, _, _ = build_spatial_data(
                    raw_data['st_filtered'], raw_data['actual_relative_times'], s2_vars['window_early'], s2_vars['window_late'], shapes, raw_data['shape_id'], route, route_idx
                )
                all_stops.append(stops_df)
                all_segments.append(segments_df)
            
        progress_bar.empty()
        
        if not all_stops:
            st.error("Could not generate multi-route map for selected criteria.")
            return False
            
        master_stops = pd.concat(all_stops, ignore_index=True)
        master_stops['rel_weighted'] = master_stops['reliability'] * master_stops['sample_size']
        master_stops = master_stops.groupby(['route_id', 'stop_id', 'stop_name', 'stop_lat', 'stop_lon'], as_index=False, observed=True).agg({'rel_weighted': 'sum', 'sample_size': 'sum'})
        master_stops['reliability'] = np.where(master_stops['sample_size'] > 0, master_stops['rel_weighted'] / master_stops['sample_size'], 0)
        master_stops.drop(columns=['rel_weighted'], inplace=True)

        if all_segments:
            master_segments = pd.concat(all_segments, ignore_index=True)
            master_segments = master_segments.groupby(['route_id', 'segment'], as_index=False).agg({'avg_reliability': 'mean', 'geometry': 'first'})
            master_segments = gpd.GeoDataFrame(master_segments, geometry='geometry', crs=LATLON_PROJ)
        else:
            master_segments = gpd.GeoDataFrame()
            
        master_stops, master_segments = inject_legend_anchors(master_stops, master_segments)
        
        t_str = f"Multi-Route Analysis | {s2_vars['days_summary']} | {s2_vars['time_range_str']} | Mode: {s2_vars['time_mode']} | Window: {s2_vars['window_early']}s to +{s2_vars['window_late']}s"
        st.session_state.raw_pipeline_data = {'title_info': t_str}
        st.session_state.analysis_results = {'is_multi': True, 'stops_df': master_stops, 'segments_df': master_segments, 'kepler_config': generate_kepler_config()}
        return True
    finally:
        lock.release()

def _clear_isolated_trips():
    if 'isolated_trips' in st.session_state:
        del st.session_state['isolated_trips']
    if 'end_stop_idx' in st.session_state:
        del st.session_state['end_stop_idx']

# Smart callback validation for corridor selectors
def validate_corridor_selection():
    new_start = st.session_state.get("start_stop_idx", 0)
    curr_end = st.session_state.get("end_stop_idx", 0)
    # If the user sets the start stop past or equal to the end stop, reset end stop
    if curr_end <= new_start:
        if 'end_stop_idx' in st.session_state:
            del st.session_state['end_stop_idx']

# ==============================================================================
# 6. FILTER SETTINGS PANEL
# ==============================================================================
def render_filter_panel(available_routes, parquet_path, trips, stop_times, stops, shapes):
    st.markdown("Configure your parameters below. Click **Apply & Run Analysis** to calculate metrics and update charts.")
    
    col_a, col_b = st.columns(2)
    
    # Initialize defaults cleanly in session state to avoid Widget API conflict errors
    if "days_selected" not in st.session_state: st.session_state.days_selected = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    if "time_slider" not in st.session_state: st.session_state.time_slider = (datetime.time(7, 0), datetime.time(9, 0))
    if "window_slider" not in st.session_state: st.session_state.window_slider = (-15, 120)
    if "time_mode" not in st.session_state: st.session_state.time_mode = "Overlap Mode"
    if "force_t0" not in st.session_state: st.session_state.force_t0 = False
    if "isolated_trips" not in st.session_state: st.session_state.isolated_trips = []
    if "route_selection" not in st.session_state: st.session_state.route_selection = available_routes[0]
    
    # Callback to reset signatures state when dependent query filters are changed
    def reset_signatures():
        st.session_state.signatures_loaded = False
        st.session_state.signature_list = []
        st.session_state.trip_start_dict = {}
        st.session_state.isolated_trips = []
        if 'start_stop_idx' in st.session_state:
            del st.session_state.start_stop_idx
        if 'end_stop_idx' in st.session_state:
            del st.session_state.end_stop_idx

    with col_b:
        st.subheader("2. Time & Date Configuration")
        days_opts = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Holiday"]
        st.multiselect("Days to Include", days_opts, key="days_selected", on_change=reset_signatures)
        st.slider("Time Window", min_value=datetime.time(0,0), max_value=datetime.time(23,59), format="HH:mm", key="time_slider", on_change=reset_signatures)

    with col_a:
        st.subheader("1. Route Selection")
        adv_mode = st.toggle("Advanced: Multi-Route Analysis", key="adv_mode", on_change=reset_signatures)
        
        if adv_mode:
            all_options = get_all_route_directions(trips, available_routes)
            if "multi_routes" not in st.session_state: st.session_state.multi_routes = all_options[:2]
            
            selected_combos = st.multiselect("Select Routes & Directions", options=all_options, key="multi_routes")
            st.info("⚠️ Calculating the entire network may a while. Detailed charts will be disabled.")
            
        else:
            selected_route = st.selectbox("Route", available_routes, key="route_selection", on_change=reset_signatures)
            gtfs_route_trips = trips[trips['route_id'] == selected_route].copy()
            headsigns = gtfs_route_trips['trip_headsign'].dropna().unique()
            # Clean single route selection choices by excluding short turns
            filtered_headsigns = [h for h in headsigns if "short" not in str(h).lower()]
            selected_dir = st.selectbox("Direction (Headsign)", filtered_headsigns if len(filtered_headsigns)>0 else ["No Data"], key="dir_selection", on_change=reset_signatures)
            
            if len(filtered_headsigns) > 0:
                gtfs_route_trips = gtfs_route_trips[gtfs_route_trips['trip_headsign'] == selected_dir]
            
            st.info("ℹ️ **What is a Schedule Signature?** A Schedule Signature groups trips that share the exact same stop sequence and relative scheduled timing. This ensures mathematical integrity by measuring every trip against an identical baseline.")
            
            if st.button("🔍 1. Find Available Schedule Signatures (Required)", use_container_width=True):
                with st.spinner("Scanning historical database for matching trip patterns..."):
                    df_hist = load_route_data(parquet_path, selected_route)
                    f_start = st.session_state.time_slider[0].hour * 3600 + st.session_state.time_slider[0].minute * 60
                    f_end = st.session_state.time_slider[1].hour * 3600 + st.session_state.time_slider[1].minute * 60
                    
                    sig_list, trip_start_dict = get_route_signatures(
                        df_hist, gtfs_route_trips, stop_times, stops, f_start, f_end, st.session_state.days_selected
                    )
                    
                    st.session_state.signature_list = sig_list
                    st.session_state.trip_start_dict = trip_start_dict
                    st.session_state.signatures_loaded = True
                    
            if st.session_state.signatures_loaded:
                if not st.session_state.signature_list:
                    st.warning("No GTFS Schedule Signatures scheduled to run within your current Time Window.")
                else:
                    sig_opts = {i: f"({s['runs']} recorded runs) | {s['orig']} → {s['dest']} | Scheduled: {format_seconds_to_time(s['min_sec'])}–{format_seconds_to_time(s['max_sec'])}" for i, s in enumerate(st.session_state.signature_list)}
                    selected_sig_idx = st.selectbox(
                        "Select Schedule Signature", 
                        options=list(sig_opts.keys()), 
                        format_func=lambda x: sig_opts[x], 
                        key="sig_selection", 
                        on_change=_clear_isolated_trips
                    )
                    
                    st.markdown("---")
                    st.markdown("**Corridor Selection** (Optional)")
                    
                    selected_sig = st.session_state.signature_list[selected_sig_idx]
                    sample_t = selected_sig['t_ids'][0]
                    
                    sample_stops = stop_times[stop_times['trip_id'] == sample_t].sort_values('stop_sequence')
                    sample_stops = sample_stops.merge(stops, on='stop_id', how='left')
                    if sample_stops['shape_dist_traveled'].max() > 500: 
                        sample_stops['shape_dist_traveled'] /= 1000.0
                        
                    stop_opts = sample_stops.to_dict('records')
                    
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_opts = list(range(len(stop_opts) - 1)) if len(stop_opts) > 1 else list(range(len(stop_opts)))
                        
                        start_stop_idx = st.selectbox(
                            "Start Stop", 
                            options=start_opts, 
                            format_func=lambda i: f"{stop_opts[i]['stop_name']} ({stop_opts[i]['shape_dist_traveled']:.1f} km)",
                            key="start_stop_idx",
                            on_change=validate_corridor_selection
                        )
                        
                    with col_end:
                        end_opts = list(range(start_stop_idx + 1, len(stop_opts))) if len(stop_opts) > 1 else list(range(len(stop_opts)))
                        
                        # Only set default to the last option if there is NO memory of a previous selection
                        if "end_stop_idx" not in st.session_state or st.session_state["end_stop_idx"] not in end_opts:
                            st.session_state["end_stop_idx"] = end_opts[-1] if end_opts else 0
                            
                        end_stop_idx = st.selectbox(
                            "End Stop", 
                            options=end_opts, 
                            format_func=lambda i: f"{stop_opts[i]['stop_name']} ({stop_opts[i]['shape_dist_traveled']:.1f} km)",
                            key="end_stop_idx"
                        )
                        
                    selected_stop_ids = [s['stop_id'] for s in stop_opts[start_stop_idx : end_stop_idx + 1]]

    with st.expander("Advanced Configuration"):
        st.slider("On-Time Reliability Window (Seconds)", min_value=-300, max_value=300, step=5, key="window_slider", help="Negative values allow early arrivals. Positive values allow late arrivals.")
        st.radio("Time Application Mode", ["Overlap Mode", "Trip Start Mode"], key="time_mode", on_change=reset_signatures, help="Overlap: Triggers if ANY part of the trip touches your Time Window. Trip Start: Only triggers if the trip explicitly originates within your Time Window.")
        
        f_disabled = False
        if not adv_mode and len(filtered_headsigns) > 0 and st.session_state.signatures_loaded and st.session_state.signature_list:
            if 'start_stop_idx' in locals() and start_stop_idx > 0:
                f_disabled = True
                
        st.checkbox("Align to First Observed Stop (Override GTFS Start)", key="force_t0", disabled=f_disabled, help="Calculates relative delays by anchoring t=0 at the first physical GPS ping at the origin stop, instead of the official GTFS scheduled departure. Disabled if Start Stop is not the true route origin.")
        
        if not adv_mode and len(filtered_headsigns) > 0 and st.session_state.signatures_loaded and st.session_state.signature_list:
            available_tids = st.session_state.signature_list[selected_sig_idx]['t_ids']
            st.multiselect("Isolate Specific Trip IDs", options=available_tids, key="isolated_trips", help="Explicitly filter the analysis to only process these scheduled trips.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_run, col_reset = st.columns([3, 1])
    
    with col_reset:
        if st.button("🔄 Reset Filters", use_container_width=True):
            keys_to_clear = [
                'days_selected', 'time_slider', 'adv_mode', 'multi_routes',
                'route_selection', 'dir_selection', 'sig_selection',
                'start_stop_idx', 'end_stop_idx', 'window_slider',
                'time_mode', 'force_t0', 'isolated_trips', 'saved_ui_state',
                'signatures_loaded', 'signature_list', 'trip_start_dict',
                'analysis_results', 'raw_pipeline_data'
            ]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    with col_run:
        run_btn = st.button("🚀 2. Apply & Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        if not adv_mode and not st.session_state.signatures_loaded:
            st.error("Please click 'Find Available Schedule Signatures' first before running a single route analysis.")
            return

        f_start = st.session_state.time_slider[0].hour * 3600 + st.session_state.time_slider[0].minute * 60
        f_end = st.session_state.time_slider[1].hour * 3600 + st.session_state.time_slider[1].minute * 60
        days_lbl = ",".join([d[:3] for d in st.session_state.days_selected]) if len(st.session_state.days_selected) < 7 else "All Days"
        
        # Override force_t0 if Corridor starts mid-route
        force_t0_val = st.session_state.force_t0
        if not adv_mode and 'start_stop_idx' in locals() and start_stop_idx > 0:
            force_t0_val = False
        
        s2_vars = {
            'filter_start_sec': f_start,
            'filter_end_sec': f_end,
            'time_mode': st.session_state.time_mode,
            'force_t0': force_t0_val,
            'days_selected': st.session_state.days_selected,
            'days_summary': days_lbl,
            'time_range_str': f"{st.session_state.time_slider[0].strftime('%H:%M')}-{st.session_state.time_slider[1].strftime('%H:%M')}",
            'window_early': st.session_state.window_slider[0],
            'window_late': st.session_state.window_slider[1],
            'stop_filter_ids': selected_stop_ids if not adv_mode and 'selected_stop_ids' in locals() else None,
            'total_route_stops': len(stop_opts) if not adv_mode and 'stop_opts' in locals() else 0,
            'isolated_trips': st.session_state.isolated_trips if (not adv_mode and 'isolated_trips' in st.session_state) else []
        }
        
        if not adv_mode:
            selected_sig = st.session_state.signature_list[selected_sig_idx]
            s2_vars['signature_t_ids'] = selected_sig['t_ids']
            s2_vars['trip_start_dict'] = st.session_state.trip_start_dict
            s2_vars['sig_desc'] = f"{selected_sig['orig']} → {selected_sig['dest']}"
        
        with st.spinner("Processing analysis pipeline..."):
            if adv_mode:
                success = execute_multi_route_pipeline(st.session_state.multi_routes, parquet_path, trips, stop_times, stops, shapes, s2_vars)
            else:
                success = execute_single_route_pipeline(parquet_path, st.session_state.route_selection, st.session_state.dir_selection, s2_vars, gtfs_route_trips, stop_times, stops, shapes)
                
            if success:
                # -------------------------------------------------------------------
                # Shadow State Persistence Logic (Saves exact UI state before hiding)
                # -------------------------------------------------------------------
                keys_to_save = [
                    'days_selected', 'time_slider', 'adv_mode', 'multi_routes',
                    'route_selection', 'dir_selection', 'sig_selection',
                    'start_stop_idx', 'end_stop_idx', 'window_slider',
                    'time_mode', 'force_t0', 'isolated_trips'
                ]
                st.session_state.saved_ui_state = {k: st.session_state[k] for k in keys_to_save if k in st.session_state}
                st.session_state.show_settings = False
                st.rerun()

def render_insights_panel(raw_pipeline_data, analysis_results):
    real_stops = analysis_results['stops_df'][analysis_results['stops_df']['stop_lat'].notna()]
    st_filt = raw_pipeline_data['st_filtered']
    art = raw_pipeline_data['actual_relative_times']
    art_str_keys = {str(k): v for k, v in art.items()}
    rel_sec_map = {str(row.stop_id): row.relative_sec for row in st_filt.itertuples()}
    trips_analyzed = len(raw_pipeline_data['mode_b_lines'])
    operating_days = len(set(line['op_date'] for line in raw_pipeline_data['mode_b_lines']))
    weighted_reliability = np.average(real_stops['reliability'], weights=real_stops['sample_size'].clip(lower=1))

    # Insight 3 - Worst Stop
    worst_row = real_stops.loc[real_stops['reliability'].idxmin()]
    worst_stop_id = str(worst_row['stop_id'])
    worst_stop_name = worst_row['stop_name']
    worst_reliability = worst_row['reliability']
    if worst_stop_id in art_str_keys and len(art_str_keys[worst_stop_id]) > 0 and worst_stop_id in rel_sec_map:
        delays_sec = [t - rel_sec_map[worst_stop_id] for t in art_str_keys[worst_stop_id]]
        median_delay_min = np.median(delays_sec) / 60.0
        delay_str = f"Typically {median_delay_min:.1f} min late" if median_delay_min > 0.5 else f"Typically {abs(median_delay_min):.1f} min early" if median_delay_min < -0.5 else "Typically on-time"
    else:
        delay_str = "No timing data"
        
    worst_short = worst_stop_name[:27] + "..." if len(worst_stop_name) > 30 else worst_stop_name

    # Insight 4 - Best Stop
    candidates = real_stops[real_stops['sample_size'] >= 5]
    if candidates.empty:
        candidates = real_stops
    best_row = candidates.loc[candidates['reliability'].idxmax()]
    best_stop_name = best_row['stop_name']
    best_short = best_stop_name[:27] + "..." if len(best_stop_name) > 30 else best_stop_name

    # Insight 5 - Biggest Reliability Cliff
    rs_df = real_stops.copy()
    rs_df['stop_id_str'] = rs_df['stop_id'].astype(str)
    st_df = st_filt.copy()
    st_df['stop_id_str'] = st_df['stop_id'].astype(str)
    merged = pd.merge(rs_df, st_df, on='stop_id_str', how='inner', suffixes=('_rs', '_st'))
    merged = merged.sort_values('shape_dist_traveled').reset_index(drop=True)
    
    diffs = merged['reliability'].diff()
    min_diff_idx = diffs.idxmin() if len(diffs) > 1 else None
    if min_diff_idx is not None and not pd.isna(min_diff_idx):
        drop = diffs.loc[min_diff_idx]
        if drop < -8:
            stop_a_name = merged.loc[min_diff_idx - 1, 'stop_name_rs']
            stop_b_name = merged.loc[min_diff_idx, 'stop_name_rs']
            cliff_text = f"Sharpest reliability drop: {stop_a_name} → {stop_b_name} ({abs(drop):.0f} pp decline)"
        else:
            cliff_text = "No single stop-to-stop reliability cliff exceeds 8 percentage points — degradation is gradual."
    else:
        cliff_text = "Not enough data to calculate reliability cliff."

    # Insight 6 - Delay Accumulation
    positions = np.arange(len(merged))
    reliabilities = merged['reliability'].values
    if len(positions) > 1 and np.std(reliabilities) > 0:
        r = np.corrcoef(positions, reliabilities)[0, 1]
        if r < -0.30:
            pattern_text = f"Delays accumulate along the route (r = {r:.2f}) — earlier stops are more reliable than later ones."
        elif r > 0.30:
            pattern_text = f"Reliability improves along the route (r = {r:.2f}) — vehicles recover schedule as the trip progresses."
        else:
            pattern_text = f"No clear spatial trend in reliability (r = {r:.2f}) — delays are distributed unevenly rather than building progressively."
    else:
        pattern_text = "Insufficient variation to determine spatial trends."

    # Insight 7 - Systematic Timing Bias
    all_offsets = []
    for sid, times in art_str_keys.items():
        if sid in rel_sec_map and len(times) > 0:
            mean_offset = np.mean(times) - rel_sec_map[sid]
            all_offsets.append(mean_offset)
            
    if not all_offsets:
        bias_text = "Insufficient data to assess timing bias."
    else:
        overall_mean = np.mean(all_offsets)
        if overall_mean > 90:
            bias_text = f"Service runs systematically late (avg {overall_mean/60:.1f} min behind schedule across all stops)."
        elif overall_mean < -90:
            bias_text = f"Service runs systematically early (avg {abs(overall_mean)/60:.1f} min ahead of schedule)."
        else:
            bias_text = f"Timing is well-centered around the schedule (avg offset: {overall_mean:+.0f}s)."

    # Layout
    with st.expander("📋 Key Findings from This Analysis", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trips Analyzed", trips_analyzed)
        col2.metric("Operating Days", operating_days)
        col3.metric("Network On-Time Rate", f"{weighted_reliability:.1f}%")
        
        c2_1, c2_2, c2_3, c2_4 = st.columns(4)
        c2_1.metric(
            label=f"Worst: {worst_short}", 
            value=f"{worst_reliability:.0f}% on-time", 
            help=f"Full Name: {worst_stop_name}\n{delay_str}"
        )
        c2_2.metric(
            label=f"Best: {best_short}", 
            value=f"{best_row['reliability']:.0f}% on-time", 
            help=f"Full Name: {best_stop_name}"
        )
        
        st.markdown("---")
        
        c3_1, c3_2, c3_3 = st.columns(3)
        c3_1.info(cliff_text)
        c3_2.info(pattern_text)
        c3_3.info(bias_text)

# ==============================================================================
# 7. MAIN UI & TAB LAYOUT
# ==============================================================================
st.title("TTC ScheduleWatch")
st.caption("Open-data analysis tool of TTC streetcar performance versus published schedules (GTFS). Visualizes streetcar locations between March 15 - May 2 2026")

parquet_path = get_parquet_path()
available_routes = get_available_routes(parquet_path)
stops, trips, stop_times, shapes = load_gtfs()

if not st.session_state.show_settings:
    if st.button("⚙️ Open Filter & Analysis Settings", type="primary"):
        st.session_state.show_settings = True
        # -------------------------------------------------------------------
        # Shadow State Restoration Logic (Pushes saved values back to widgets)
        # -------------------------------------------------------------------
        if 'saved_ui_state' in st.session_state:
            for k, v in st.session_state.saved_ui_state.items():
                st.session_state[k] = v
        st.rerun()

if st.session_state.show_settings:
    with st.container():
        st.markdown("### ⚙️ Analysis & Filter Settings")
        render_filter_panel(available_routes, parquet_path, trips, stop_times, stops, shapes)
        st.markdown("---")

if st.session_state.analysis_results is not None and not st.session_state.analysis_results.get('is_multi', False):
    render_insights_panel(st.session_state.raw_pipeline_data, st.session_state.analysis_results)

tab_map, tab_spaghetti, tab_stats = st.tabs([
    "🗺️ Route Reliability Map", 
    "🍝 Time-Distance Chart", 
    "📊 Density Chart"
])

with tab_map:
    if not st.session_state.analysis_results:
        precomputed = load_precomputed_network()
        if precomputed:
            st.info("🗺️ **Showing Default Network Reliability View.** All-Day Weekdays, All Routes. Click the **⚙️ Open Filter & Analysis Settings** button above to run a custom analysis.")
            stops_df = pd.DataFrame(precomputed['stops'])
            segments_df = gpd.GeoDataFrame.from_features(precomputed['segments']['features'])
            
            # Inject anchors so the landing page colors lock to 0% - 100%
            stops_df, segments_df = inject_legend_anchors(stops_df, segments_df)
            
            map_instance = KeplerGl(height=600, data={"stops": stops_df, "segments": segments_df}, config=generate_kepler_config())
            keplergl_static(map_instance, center_map=True)
        else:
            st.info("🗺️ **Map View is Empty.** Please click the **⚙️ Open Filter & Analysis Settings** button above to run an analysis.")
    else:
        st.markdown(f"**Configuration:** {st.session_state.raw_pipeline_data['title_info']}")
        results = st.session_state.analysis_results
        if 'segments_df' in results and not results['segments_df'].empty:
            map_instance = KeplerGl(height=600, data={"stops": results['stops_df'], "segments": results['segments_df']}, config=results['kepler_config'])
            keplergl_static(map_instance, center_map=True)
        else:
            st.warning("Spatial geometry could not be built for this route.")

with tab_spaghetti:
    if not st.session_state.analysis_results:
        st.info("🍝 **Time-Distance Chart is Empty.** Please click the **⚙️ Open Filter & Analysis Settings** button above to run an analysis.")
    elif st.session_state.analysis_results.get('is_multi', False):
        st.warning("⚠️ **Charts Disabled.** Detailed trip visualizations are only available when analyzing a single route.")
    else:
        st.markdown(f"**Configuration:** {st.session_state.raw_pipeline_data['title_info']}")
        
        gmaps_link_container = st.empty()
        
        event = st.plotly_chart(
            st.session_state.analysis_results['fig_B'],
            use_container_width=True,
            height=900,
            config=PLOTLY_CONFIG,
            on_select="rerun",
            selection_mode=["points"]
        )
        
        if event and event.selection.get("points"):
            pt = event.selection["points"][0]
            if "customdata" in pt and len(pt["customdata"]) >= 6:
                op_date = pt["customdata"][0]
                t_id = pt["customdata"][2]
                lat = pt["customdata"][4]
                lon = pt["customdata"][5]
                
                gmaps_link_container.success(
                    f"📍 **Selected {op_date} | Trip {t_id}:** "
                    f"[**Click here to open this location in Google Maps**]"
                    f"(https://www.google.com/maps/search/?api=1&query={lat},{lon})"
                )
            else:
                gmaps_link_container.info("Click a specific coordinate point on a trip line to get a Google Maps link.")
        else:
            gmaps_link_container.caption("👉 Click any data point on the chart to generate a Google Maps link for that exact location.")

with tab_stats:
    if not st.session_state.analysis_results:
        st.info("📊 **Density Chart is Empty.** Please click the **⚙️ Open Filter & Analysis Settings** button above to run an analysis.")
    elif st.session_state.analysis_results.get('is_multi', False):
        st.warning("⚠️ **Charts Disabled.** Detailed density plots are only available when analyzing a single route.")
    else:
        st.markdown(f"**Configuration:** {st.session_state.raw_pipeline_data['title_info']}")
        st.plotly_chart(st.session_state.analysis_results['fig_A'], use_container_width=True, height=900, config=PLOTLY_CONFIG)

st.markdown("---")
st.caption("**Data Privacy Statement:** All data is open public data sourced from the City of Toronto Open Data Portal. © 2026 Neil Simmons. All rights reserved.")
