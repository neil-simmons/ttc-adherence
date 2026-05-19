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
    page_title="TTC Streetcar Reliability",
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
    if len(name) > 30:
        name = name[:27] + "..."
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
        if len(group) < 2: continue

        interpolated_times = np.interp(st_filtered['shape_dist_traveled'].values, group['official_dist_km'].values, group['op_seconds'].values, left=np.nan, right=np.nan)
        run_interpolations = {sid: t for sid, t in zip
