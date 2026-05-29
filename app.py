import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring, linemerge
import plotly.graph_objects as go
import plotly.io as pio
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

# ==============================================================================
# 0. CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="TTC-ScheduleWatch",
    page_icon="🚊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# MOBILE WARNING CSS INJECTION & ACCESSIBILITY SCREEN-READER CLASSES
st.markdown("""
<style>
.mobile-warning { display: none; background-color: #ffcccc; color: #900; padding: 12px; border-left: 6px solid #DA251D; margin-bottom: 15px; font-size: 14px; border-radius: 4px; }
@media (max-width: 768px) { .mobile-warning { display: block; } }
/* Standard off-screen styling to provide structural details strictly to screen readers without modifying visual layout */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}
</style>
<div class="mobile-warning" role="alert" aria-live="polite">⚠️ <b>Mobile Device Detected:</b> This dashboard includes extremely dense data visualizations. Please open on a desktop computer for the best experience with the Time-Distance and Density charts.</div>
""", unsafe_allow_html=True)


HF_REPO      = "neil-simmons/ttc-avl-data"
HF_REPO_TYPE = "dataset"

PARQUET_HISTORY = "ttc_all_streetcars_history.parquet"
GTFS_STOPS      = "stops.txt"
GTFS_TRIPS      = "trips.txt"
GTFS_STOP_TIMES = "stop_times.txt"
GTFS_SHAPES     = "shapes.txt"
PRECOMPUTED_MAP = "precomputed_network.json"
EQUITY_NBH_FILE = "equity_neighbourhoods.geojson"

START_DATE    = '2026-03-15'
END_DATE      = '2026-05-02 23:59:59'
STAT_HOLIDAYS = ['2026-04-03']

MAX_TRACK_DEVIATION_M   = 150
MAX_ALLOWED_PING_GAP_SEC = 60
UTM_PROJ   = "EPSG:32617"
LATLON_PROJ = "EPSG:4326"

TTC_RED = "#DA251D"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}

TIME_BUCKETS = [
    ("Early AM",    0,  7),
    ("AM Peak",     7,  10),
    ("Midday",     10,  14),
    ("PM Peak",    14,  18),
    ("Evening",    18,  21),
    ("Late Night", 21,  28),
]

DOW_LABELS = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday',
    'Friday', 'Saturday', 'Sunday'
]

WCAG_ROUTE_COLORS = [
    "#0072B2",  # Blue        4.71:1
    "#D55E00",  # Vermillion  4.65:1
    "#009E73",  # Teal        4.52:1
    "#B56900",  # Dark amber  5.01:1
    "#7B2D8B",  # Purple      7.02:1
    "#1E7340",  # Dark green  7.48:1
    "#C0392B",  # Dark red    5.52:1
    "#2C3E50",  # Slate       13.1:1
]


# Additive Screen-Reader Announcement Utility
def announce_sr(text):
    st.markdown(f'<div class="sr-only" role="status" aria-live="polite">{text}</div>', unsafe_allow_html=True)

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

# Set color theme defaults in session state
if 'color_theme' not in st.session_state: st.session_state.color_theme = "Default (Classic Red-Green)"

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

@st.cache_resource(show_spinner="Loading neighbourhood equity data...")
def load_equity_data():
    path = _hf(EQUITY_NBH_FILE)
    return gpd.read_file(path)

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
    df = df[mask].copy()

    # Sort chronological to ensure the gap detector works properly
    df = df.sort_values(['trip_id', 'system_time'])

    # 1. Identify distinct daily runs. A gap of > 4 hours (14400s) safely splits yesterday's run from today's.
    time_diff = df.groupby('trip_id', observed=True)['system_time'].diff()
    df['run_id'] = (time_diff.isna() | (time_diff > 14400)).cumsum()

    # 2. Get the exact starting timestamp of each isolated run
    run_starts = df.groupby('run_id')['system_time'].transform('min')
    run_starts_local = pd.to_datetime(run_starts, unit='s', utc=True).dt.tz_convert('America/Toronto')
    
    # 3. Determine the operating date based on the run's start time
    # If the run STARTS before 4 AM, the ENTIRE continuous run belongs to yesterday
    start_hour = run_starts_local.dt.hour
    op_date = np.where(
        start_hour < 4,
        (run_starts_local - pd.Timedelta(days=1)).dt.date,
        run_starts_local.dt.date
    )
    df['op_date'] = pd.Series(op_date, index=df.index).astype(str).astype('category')

    # 4. Calculate continuous op_seconds relative to that single op_date's midnight
    # Using unix timestamp subtraction keeps the time perfectly continuous past 4:00 AM
    op_midnight_local = pd.to_datetime(df['op_date'].astype(str)).dt.tz_localize('America/Toronto')
    op_midnight_epoch = op_midnight_local.astype('int64') // 10**9
    
    df['op_seconds']  = (df['system_time'] - op_midnight_epoch).astype(np.int32)
    df['day_of_week'] = pd.to_datetime(df['op_date']).dt.dayofweek.astype(np.int8)
    df['is_holiday']  = df['op_date'].astype(str).isin(STAT_HOLIDAYS)

    df.drop(columns=['run_id'], inplace=True)

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
    # Dynamic toggle logic for color scheme
    if st.session_state.get('color_theme', 'Default (Classic Red-Green)') == "Accessible":
        # diverging sequential blue-to-yellow palette
        custom_20_colors = [
            "#053061", "#1e538d", "#3676b9", "#5298c8", "#78b9d6",
            "#a4dae8", "#cef1f5", "#ebf7f5", "#fdfae5", "#fef1be",
            "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02",
            "#993404", "#662506", "#4d1a04", "#331102", "#1a0801"
        ]
    else:
        # Original classic TTC red-to-green palette
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
                        "id": "stops", "type": "point",
                        "config": {
                            "dataId": "stops", "label": "Stops", "columns": {"lat": "stop_lat", "lng": "stop_lon"}, "isVisible": True,
                            "visConfig": {
                                "radius": 4.5, # Small, crisp screen pixel radius (stays sharp at all zoom levels)
                                "radiusUnit": "pixels", 
                                "opacity": 0.95, 
                                "filled": True, 
                                "outline": True, 
                                "thickness": 0.8, # Thin outline matches the pinhead scale
                                "strokeColor": [220, 220, 220], 
                                "colorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "reliability", "type": "real"}, "colorScale": "quantize"}
                    },
                    {
                        "id": "segments", "type": "geojson",
                        "config": {
                            "dataId": "segments", "label": "Route Segments", "columns": {"geojson": "geometry"}, "isVisible": True,
                            "visConfig": {
                                "opacity": 0.5, 
                                "strokeOpacity": 0.5, 
                                "thickness": 0.6, 
                                "strokeColor": None, 
                                "colorRange": color_scale_config, 
                                "strokeColorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "avg_reliability", "type": "real"}, "colorScale": "quantize", "strokeColorField": {"name": "avg_reliability", "type": "real"}, "strokeColorScale": "quantize"}
                    }
                ],
                "layerOrder": ["stops", "segments"],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "segments": [{"name": "segment", "format": None}, {"name": "route_id", "format": None}, {"name": "direction", "format": None}, {"name": "avg_reliability", "format": ".1f"}],
                            "stops": [{"name": "stop_name", "format": None}, {"name": "route_id", "format": None}, {"name": "reliability", "format": ".1f"}]
                        },
                        "enabled": True
                    }
                }
            },
            "mapStyle": {"styleType": "muted_night"}
        }
    }

def generate_equity_kepler_config():
    if st.session_state.get('color_theme', 'Default (Classic Red-Green)') == "Accessible":
        custom_20_colors = [
            "#053061", "#1e538d", "#3676b9", "#5298c8", "#78b9d6",
            "#a4dae8", "#cef1f5", "#ebf7f5", "#fdfae5", "#fef1be",
            "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02",
            "#993404", "#662506", "#4d1a04", "#331102", "#1a0801"
        ]
    else:
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
                        "id": "stops", "type": "point",
                        "config": {
                            "dataId": "stops", "label": "Stops", "columns": {"lat": "stop_lat", "lng": "stop_lon"}, "isVisible": True,
                            "visConfig": {
                                "radius": 4.5, # Small, crisp screen pixel radius (stays sharp at all zoom levels)
                                "radiusUnit": "pixels", 
                                "opacity": 0.95, 
                                "filled": True, 
                                "outline": True, 
                                "thickness": 1.0, # Dark border provides contrast on top of the census shapes
                                "strokeColor": [255, 255, 255], 
                                "colorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "reliability", "type": "real"}, "colorScale": "quantize"}
                    },
                    {
                        "id": "segments", "type": "geojson",
                        "config": {
                            "dataId": "segments", "label": "Route Segments", "columns": {"geojson": "geometry"}, "isVisible": True,
                            "visConfig": {
                                "opacity": 0.9, 
                                "strokeOpacity": 0.9, 
                                "thickness": 1.4, 
                                "strokeColor": None, 
                                "colorRange": color_scale_config, 
                                "strokeColorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "avg_reliability", "type": "real"}, "colorScale": "quantize", "strokeColorField": {"name": "avg_reliability", "type": "real"}, "strokeColorScale": "quantize"}
                    },
                    {
                        "id": "eq_income", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Median Household Income ($)", "columns": {"geojson": "geometry"}, "isVisible": True,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Income_Blues", "type": "custom", "category": "Custom", "colors": ["#eff3ff","#c6dbef","#9ecae1","#6baed6","#3182bd","#08519c"]}}
                        },
                        "visualChannels": {"colorField": {"name": "median_income", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_lowincome", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Low-Income Households (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "LowIncome_Purples", "type": "custom", "category": "Custom", "colors": ["#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#756bb1","#54278f"]}}
                        },
                        "visualChannels": {"colorField": {"name": "low_income_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_transit", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Transit Commuters (%) — Transit Dependence", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Transit_Pinks", "type": "custom", "category": "Custom", "colors": ["#fde0dd","#fcc5c0","#fa9fb5","#f768a1","#c51b8a","#7a0177"]}}
                        },
                        "visualChannels": {"colorField": {"name": "transit_commute_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_vismin", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Visible Minority Population (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "VisMin_Greys", "type": "custom", "category": "Custom", "colors": ["#f7f7f7","#d9d9d9","#bdbdbd","#969696","#636363","#252525"]}}
                        },
                        "visualChannels": {"colorField": {"name": "visible_minority_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_immigrant", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Recent Immigrants — Last 5 Years (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Immigrant_Indigo", "type": "custom", "category": "Custom", "colors": ["#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#810f7c"]}}
                        },
                        "visualChannels": {"colorField": {"name": "recent_immigrant_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_seniors", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Seniors 65+ (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Senior_Browns", "type": "custom", "category": "Custom", "colors": ["#f6e8c3","#dfc27d","#bf812d","#8c510a","#543005","#331A00"]}}
                        },
                        "visualChannels": {"colorField": {"name": "senior_pct", "type": "real"}, "colorScale": "quantile"}
                    }
                ],
                "layerOrder": ["stops", "segments", "eq_income", "eq_lowincome", "eq_transit", "eq_vismin", "eq_immigrant", "eq_seniors"],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "equity": [
                                {"name": "area_name", "format": None},
                                {"name": "median_income", "format": None},
                                {"name": "low_income_pct", "format": None},
                                {"name": "transit_commute_pct", "format": None},
                                {"name": "visible_minority_pct", "format": None},
                                {"name": "recent_immigrant_pct", "format": None},
                                {"name": "senior_pct", "format": None}
                            ],
                            "segments": [{"name": "segment", "format": None}, {"name": "route_id", "format": None}, {"name": "direction", "format": None}, {"name": "avg_reliability", "format": ".1f"}],
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

    return sorted(sig_ui_list, key=lambda x: (-x['runs'], x['min_sec'])), trip_start_dict

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

        # =====================================================================
        # 1. ACCURATE TERMINAL LAYOVER TRIMMING (BY INDEX SLICING)
        # =====================================================================
        # Identify the extreme physical start and end coordinates of this run
        min_dist_val = group['official_dist_km'].min()
        max_dist_val = group['official_dist_km'].max()
        
        # Last ping at the origin terminal (the exact departure moment)
        start_idx = group[group['official_dist_km'] == min_dist_val].index[-1]
        # First ping at the destination terminal (the exact arrival moment)
        end_idx = group[group['official_dist_km'] == max_dist_val].index[0]
        
        # Slice to preserve all intermediate pings, but drop dead-time at terminals
        group = group.loc[start_idx:end_idx].copy()
        
        if len(group) < 2: continue

        # =====================================================================
        # 2. GPS COORDINATE SMOOTHING & RATCHET (WOBBLE FILTER)
        # =====================================================================
        # To eliminate GPS coordinates jitter/creep without dropping timestamps, 
        # we apply a rolling median filter (window of 3) to smooth the distances,
        # followed by cummax() to prevent backward GPS drift.
        # This preserves 100% of the dwell-time timestamps and keeps the math & visuals coupled.
        group['official_dist_km'] = (
            group['official_dist_km']
            .rolling(window=3, min_periods=1, center=True)
            .median()
        )
        group['official_dist_km'] = group['official_dist_km'].cummax()

        if group['official_dist_km'].isna().all(): continue

        # =====================================================================
        # 3. VECTORIZED DETERMINISTIC INTERPOLATION (TRUE ARRIVALS)
        # =====================================================================
        stop_dists = st_filtered['shape_dist_traveled'].values
        ping_dists = group['official_dist_km'].values
        ping_times = group['op_seconds'].values
        
        # Search with side='left' to target the exact Arrival Edge of stops
        idxs = np.searchsorted(ping_dists, stop_dists, side='left')
        idxs = np.clip(idxs, 1, len(ping_dists) - 1)
        
        # Extract the bounding coordinates and timestamps
        d1 = ping_dists[idxs - 1]
        d2 = ping_dists[idxs]
        t1 = ping_times[idxs - 1]
        t2 = ping_times[idxs]
        
        # Perform single-step linear interpolation (avoiding division-by-zero)
        denom = d2 - d1
        denom_safe = np.where(denom == 0, 1.0, denom)
        
        interpolated_times = t1 + (stop_dists - d1) * (t2 - t1) / denom_safe
        
        # Nullify if the stop falls outside the actual tracked range of the trip
        interpolated_times = np.where(
            (stop_dists < ping_dists[0]) | (stop_dists > ping_dists[-1]), 
            np.nan, 
            interpolated_times
        )

        # =====================================================================
        # 4. GEOFENCED GAP FILTERING (PREVENTS FALSE DWELL TIMEOUTS)
        # =====================================================================
        # Check the gap of the segment immediately leading into the arrival.
        # This prevents long dwell times from triggering the missing-data filter.
        gaps = t2 - t1
        interpolated_times = np.where(gaps > MAX_ALLOWED_PING_GAP_SEC, np.nan, interpolated_times)
        # =====================================================================

        run_interpolations = {
            sid: t for sid, t in zip(st_filtered['stop_id'], interpolated_times) if not np.isnan(t)
        }
        if not run_interpolations: continue

        anchor_stop = st_filtered.iloc[1] if len(st_filtered) > 1 else st_filtered.iloc[0]
        anchor_stop_dist = anchor_stop['shape_dist_traveled']

        if group['official_dist_km'].iloc[0] > anchor_stop_dist: continue

        if s2_vars['force_t0']:
            anchor_stop_id = anchor_stop['stop_id']
            if anchor_stop_id not in run_interpolations: continue
            
            # Find the time index of the anchor stop using our smoothed pings
            idx_after = np.searchsorted(group['official_dist_km'].values, anchor_stop_dist)
            if idx_after == 0 or idx_after >= len(group): continue
            
            time_gap = group['op_seconds'].iloc[idx_after] - group['op_seconds'].iloc[idx_after - 1]
            if time_gap > MAX_ALLOWED_PING_GAP_SEC: continue
            anchor_sec = run_interpolations[anchor_stop_id] - anchor_stop['relative_sec']
        else:
            anchor_sec = gtfs_start_sec

        f_start, f_end = s2_vars['filter_start_sec'], s2_vars['filter_end_sec']
        if "Trip Start Mode" in s2_vars['time_mode']: 
            is_valid = f_start <= anchor_sec <= f_end
        else: 
            is_valid = any(f_start <= t <= f_end for t in run_interpolations.values())

        if is_valid:
            stop_delays_this_trip = {
                str(s_id): (t - anchor_sec)
                for s_id, t in run_interpolations.items()
            }
            for s_id, t in run_interpolations.items(): 
                actual_relative_times[s_id].append(t - anchor_sec)

            dist_diff = group['official_dist_km'].diff()
            time_diff = group['system_time'].diff()
            group['prev_speed_kmh'] = np.where(time_diff > 0, (dist_diff / time_diff) * 3600, 0).clip(min=0)
            group['relative_min'] = (group['op_seconds'] - anchor_sec) / 60.0

            # Filter points specifically to the selected corridor range for visual plotting
            plot_group = group[(group['official_dist_km'] >= min_dist - 0.1) & (group['official_dist_km'] <= max_dist + 0.1)].copy()
            if len(plot_group) < 2: continue

            raw_x = plot_group['relative_min'].tolist()
            raw_y = plot_group['official_dist_km'].tolist()
            raw_lat = plot_group['latitude'].tolist()
            raw_lon = plot_group['longitude'].tolist()
            raw_speed = plot_group['prev_speed_kmh'].tolist()
            raw_systime = plot_group['system_time'].tolist()

            raw_abs = (pd.to_datetime(plot_group['system_time'], unit='s', utc=True)
                       .dt.tz_convert('America/Toronto')
                       .dt.strftime('%I:%M:%S %p').tolist())

            x_gaps, y_gaps, abs_gaps, lat_gaps, lon_gaps, speed_gaps = [], [], [], [], [], []

            for i in range(len(raw_x)):
                if i > 0 and (raw_systime[i] - raw_systime[i-1]) > MAX_ALLOWED_PING_GAP_SEC:
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
                'abs_time': abs_gaps, 'lat': lat_gaps, 'lon': lon_gaps, 'speed': speed_gaps,
                'anchor_sec': anchor_sec,
                'stop_delays': stop_delays_this_trip
            })
            
    if not mode_b_lines: return None
    return {'st_filtered': st_filtered, 'actual_relative_times': actual_relative_times, 'mode_b_lines': mode_b_lines, 'shape_id': sample_shape_id}

def build_spatial_data(st_filtered, actual_relative_times, window_early, window_late, shapes, shape_id, route_id, route_idx, direction):
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
        offset_stops.append({
        'route_id': route_id, 
        'stop_id': stop.stop_id, 
        'stop_name': stop.stop_name, 
        'stop_lat': new_lat, 
        'stop_lon': new_lon,
        'true_lat': float(stop.stop_lat), # Added
        'true_lon': float(stop.stop_lon), # Added
        'reliability': reliability_vals[stop.stop_id], 
        'sample_size': sample_sizes[stop.stop_id]
    })
        
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
        
        seg_samples = (sample_sizes[s1.stop_id] + sample_sizes[s2.stop_id]) / 2.0
        segments.append({
            'route_id': route_id, 
            'direction': direction, # Store the direction headsign 
            'segment': f"{s1.stop_name} to {s2.stop_name}", 
            'avg_reliability': (reliability_vals[s1.stop_id] + reliability_vals[s2.stop_id]) / 2.0, 
            'sample_size': seg_samples,
            'geometry': geom
        })
        
    segments_df = gpd.GeoDataFrame(segments, geometry='geometry', crs=LATLON_PROJ) if segments else gpd.GeoDataFrame()
    return stops_df, segments_df, reliability_dict, reliability_vals

def compute_trip_stats(raw_pipeline_data):
    """
    Derives per-trip and per-stop aggregated statistics from the enriched
    mode_b_lines produced by run_tracking(). Returns a dict stored under
    raw_pipeline_data['trip_stats']. Only valid for single-signature analyses.
    """
    mode_b = raw_pipeline_data['mode_b_lines']
    art    = raw_pipeline_data['actual_relative_times']
    st_filt = raw_pipeline_data['st_filtered']

    # Build ordered stop list (route order, str keys for consistency)
    ordered = st_filt.sort_values('shape_dist_traveled')
    stop_order   = [str(r.stop_id) for r in ordered.itertuples()]
    stop_names   = {str(r.stop_id): r.stop_name  for r in ordered.itertuples()}
    rel_sec_map  = {str(r.stop_id): r.relative_sec for r in ordered.itertuples()}

    n_trips = len(mode_b)

    # Per-trip arrays (parallel to mode_b)
    per_trip_mean_delay = []   # float, seconds; NaN if no stops resolved
    per_trip_dow        = []   # int 0=Mon..6=Sun; None on parse failure
    per_trip_date       = []   # 'YYYY-MM-DD' string
    per_trip_hour       = []   # float, 0–24; None if anchor_sec missing

    for line in mode_b:
        # Mean delay across all successfully interpolated stops for this trip
        sd = line.get('stop_delays', {})
        if sd and rel_sec_map:
            delays = [
                sd[sid] - rel_sec_map[sid]
                for sid in sd
                if sid in rel_sec_map
            ]
            per_trip_mean_delay.append(float(np.mean(delays)) if delays else np.nan)
        else:
            per_trip_mean_delay.append(np.nan)

        # Day of week
        date_str = line.get('op_date', '')
        per_trip_date.append(date_str)
        try:
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            per_trip_dow.append(dt.weekday())
        except Exception:
            per_trip_dow.append(None)

        # Time of day from anchor_sec
        anc = line.get('anchor_sec')
        if anc is not None:
            per_trip_hour.append((float(anc) % 86400.0) / 3600.0)
        else:
            per_trip_hour.append(None)

    # Per-stop delay stats from actual_relative_times
    # (uses the full sample, not trip-indexed, for robust per-stop stats)
    art_str = {str(k): v for k, v in art.items()}
    per_stop_delays    = {}   # stop_id_str -> list of delays in seconds
    per_stop_mean      = {}
    per_stop_std       = {}
    for sid in stop_order:
        if sid in art_str and sid in rel_sec_map and art_str[sid]:
            delays = [t - rel_sec_map[sid] for t in art_str[sid]]
            per_stop_delays[sid] = delays
            per_stop_mean[sid]   = float(np.mean(delays))
            per_stop_std[sid]    = float(np.std(delays))

    # Summary scalars for condition checks
    valid_hours = [h for h in per_trip_hour if h is not None]
    valid_dow   = [d for d in per_trip_dow  if d is not None]
    valid_dates = [d for d in per_trip_date if d]

    hour_range = (max(valid_hours) - min(valid_hours)) if len(valid_hours) >= 2 else 0.0

    return {
        'stop_order':          stop_order,
        'stop_names':          stop_names,
        'rel_sec_map':         rel_sec_map,
        'n_trips':             n_trips,
        'per_trip_mean_delay': per_trip_mean_delay,
        'per_trip_dow':        per_trip_dow,
        'per_trip_date':       per_trip_date,
        'per_trip_hour':       per_trip_hour,
        'per_stop_delays':     per_stop_delays,
        'per_stop_mean':       per_stop_mean,
        'per_stop_std':        per_stop_std,
        'n_unique_dates':      len(set(valid_dates)),
        'n_unique_dow':        len(set(valid_dow)),
        'hour_range':          hour_range,
        'mode_b_lines':        mode_b
    }

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
        st.error("After filtering for valid trips, none were found. This may occur from construction, deadheading or other variables. Tracking may have also failed.")
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
        raw_data['st_filtered'], raw_data['actual_relative_times'], s2_vars['window_early'], s2_vars['window_late'], shapes, raw_data['shape_id'], selected_route, 0, selected_dir
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
        y_tick_texts.append(f"{clean_name} ({row['shape_dist_traveled']:.1f}km) | Rel: {reliability_dict[row['stop_id']]} ")
        
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

    # -----------------------------------------------------------------
    # BUILD FIG B (TIME-DISTANCE) WITH EXPLICIT HOVER TEMPLATES & SPEED
    # -----------------------------------------------------------------
    fig_B = go.Figure()
    
    hover_tmpl = (
        "<b>Date:</b> %{customdata[0]}<br>"
        "<b>Trip ID:</b> %{customdata[2]}<br>"
        "<b>Abs Time:</b> %{customdata[3]}<br>"
        "<b>Distance:</b> %{y:.2f} km<br>"
        "<b>Rel Time:</b> %{x:.1f} mins<br>"
        "<b>Avg Speed (Last Segment):</b> %{customdata[6]:.1f} km/h<br>"  # Added speed to hover
        "<i>(Click point to get Google Maps link)</i>"
        "<extra></extra>" 
    )

    for line_data in raw_data['mode_b_lines']:
        # Expanded customdata from 6 dimensions to 7 to house segment speed
        cd = np.empty((len(line_data['x']), 7), dtype=object)
        cd[:, 0] = line_data['op_date']
        cd[:, 1] = line_data['start_time']
        cd[:, 2] = line_data['t_id']
        cd[:, 3] = line_data['abs_time']
        cd[:, 4] = line_data['lat']
        cd[:, 5] = line_data['lon']
        cd[:, 6] = line_data['speed']  # Passed calculated segment speed array to index 6
        
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

    st.session_state.analysis_results = {
        'is_multi': False, 
        'fig_A': fig_A, 
        'fig_B': fig_B, 
        'stops_df': stops_df, 
        'segments_df': segments_df, 
        'kepler_config': generate_kepler_config()
    }
    announce_sr(f"Analysis completed successfully for Route {selected_route}. Visual charts and geographic maps are updated.")
    
    raw_data['trip_stats'] = compute_trip_stats(raw_data)
    st.session_state.show_temporal = False
    st.session_state.show_spacetime = False
    
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
                    raw_data['st_filtered'], raw_data['actual_relative_times'], s2_vars['window_early'], s2_vars['window_late'], shapes, raw_data['shape_id'], route, route_idx, direction
                )
                all_stops.append(stops_df)
                all_segments.append(segments_df)
            
        progress_bar.empty()
        
        if not all_stops:
            st.error("Could not generate multi-route map for selected criteria.")
            return False
            
        master_stops = pd.concat(all_stops, ignore_index=True)
        master_stops['rel_weighted'] = master_stops['reliability'] * master_stops['sample_size']
        
        # Ensure route_id is handled as a string for clean list concatenation
        master_stops['route_id'] = master_stops['route_id'].astype(str)
        
        # Group strictly by physical stop metadata to eliminate overlapping concentric rings
        master_stops = master_stops.groupby(['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'true_lat', 'true_lon'], as_index=False, observed=True).agg({
            'route_id': lambda x: ", ".join(sorted(list(set(x)))),  # Merges route labels: "501, 504"
            'rel_weighted': 'sum',
            'sample_size': 'sum'
        })
        
        # Compute the statistically rigorous weighted Micro-Average across pooled transit runs
        master_stops['reliability'] = np.where(master_stops['sample_size'] > 0, master_stops['rel_weighted'] / master_stops['sample_size'], 0)
        master_stops.drop(columns=['rel_weighted'], inplace=True)

        if all_segments:
            master_segments = pd.concat(all_segments, ignore_index=True)
            master_segments['seg_rel_weighted'] = master_segments['avg_reliability'] * master_segments['sample_size']
            
            # Group by route_id and direction to keep different travel directions separate
            master_segments = master_segments.groupby(['route_id', 'direction', 'segment'], as_index=False).agg({
                'seg_rel_weighted': 'sum',
                'sample_size': 'sum',
                'geometry': 'first'
            })
            master_segments['avg_reliability'] = np.where(
                master_segments['sample_size'] > 0, 
                master_segments['seg_rel_weighted'] / master_segments['sample_size'], 
                0
            )
            master_segments.drop(columns=['seg_rel_weighted'], inplace=True)
            master_segments = gpd.GeoDataFrame(master_segments, geometry='geometry', crs=LATLON_PROJ)
        else:
            master_segments = gpd.GeoDataFrame()
            
        master_stops, master_segments = inject_legend_anchors(master_stops, master_segments)
        
        t_str = f"Multi-Route Analysis | {s2_vars['days_summary']} | {s2_vars['time_range_str']} | Mode: {s2_vars['time_mode']} | Window: {s2_vars['window_early']}s to +{s2_vars['window_late']}s"
        st.session_state.raw_pipeline_data = {'title_info': t_str}
        st.session_state.analysis_results = {'is_multi': True, 'stops_df': master_stops, 'segments_df': master_segments, 'kepler_config': generate_kepler_config()}
        announce_sr("Route-Level calculations complete. Network map refreshed with collective performance metrics.")
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
    if "window_slider" not in st.session_state: st.session_state.window_slider = (-60, 120)
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
        adv_mode = st.toggle("Advanced: Route-Level Analysis", key="adv_mode", on_change=reset_signatures)
        #Advanced: Route-Level Analysis used to be called Multi-Level Analysis
        
        if adv_mode:
            all_options = get_all_route_directions(trips, available_routes)
            if "multi_routes" not in st.session_state: st.session_state.multi_routes = all_options[:2]
            
            selected_combos = st.multiselect("Select Routes & Directions", options=all_options, key="multi_routes")
            st.info("⚠️ Calculating the entire network may take a while. Detailed charts will be disabled.")
            
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
                    announce_sr(f"Found {len(sig_list)} available schedule signatures matching your parameters.")
                    
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
                'analysis_results', 'raw_pipeline_data', 'show_temporal',
                'show_spacetime'
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
                    'time_mode', 'force_t0', 'isolated_trips', 'color_theme'
                ]
                st.session_state.saved_ui_state = {k: st.session_state[k] for k in keys_to_save if k in st.session_state}
                st.session_state.show_settings = False
                st.rerun()  # <--- PUT THIS BACK

def render_insights_panel(raw_pipeline_data, analysis_results):
    is_multi = analysis_results.get('is_multi', False)
    real_stops = analysis_results['stops_df'][analysis_results['stops_df']['stop_lat'].notna()]
    weighted_reliability = np.average(real_stops['reliability'], weights=real_stops['sample_size'].clip(lower=1))

    # Gracefully handle missing granular data in multi-route mode
    if is_multi:
        trips_analyzed = "N/A"
        operating_days = "N/A"
    else:
        st_filt = raw_pipeline_data['st_filtered']
        art = raw_pipeline_data['actual_relative_times']
        art_str_keys = {str(k): v for k, v in art.items()}
        rel_sec_map = {str(row.stop_id): row.relative_sec for row in st_filt.itertuples()}
        trips_analyzed = len(raw_pipeline_data['mode_b_lines'])
        operating_days = len(set(line['op_date'] for line in raw_pipeline_data['mode_b_lines']))

    # Insight 3 - Worst Stop
    worst_row = real_stops.loc[real_stops['reliability'].idxmin()]
    worst_stop_id = str(worst_row['stop_id'])
    worst_stop_name = worst_row['stop_name']
    worst_reliability = worst_row['reliability']
    
    if not is_multi and worst_stop_id in art_str_keys and len(art_str_keys[worst_stop_id]) > 0 and worst_stop_id in rel_sec_map:
        delays_sec = [t - rel_sec_map[worst_stop_id] for t in art_str_keys[worst_stop_id]]
        median_delay_min = np.median(delays_sec) / 60.0
        delay_str = f"Typically {median_delay_min:.1f} min late" if median_delay_min > 0.5 else f"Typically {abs(median_delay_min):.1f} min early" if median_delay_min < -0.5 else "Typically on-time"
    else:
        delay_str = "Specific timing variance unavailable in multi-route view"
        
    worst_short = worst_stop_name[:27] + "..." if len(worst_stop_name) > 30 else worst_stop_name

    # Insight 4 - Best Stop
    candidates = real_stops[real_stops['sample_size'] >= 5]
    if candidates.empty:
        candidates = real_stops
    best_row = candidates.loc[candidates['reliability'].idxmax()]
    best_stop_name = best_row['stop_name']
    best_short = best_stop_name[:27] + "..." if len(best_stop_name) > 30 else best_stop_name

    # Insight 5 - Biggest Reliability Cliff (Disabled for multi-route due to intersecting geometries)
    if is_multi:
        cliff_text = "Stop-to-stop reliability cliff calculation is disabled in network mode due to intersecting route geometries."
    else:
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
            if drop < -10:
                stop_a_name = merged.loc[min_diff_idx - 1, 'stop_name_rs']
                stop_b_name = merged.loc[min_diff_idx, 'stop_name_rs']
                cliff_text = f"Sharpest reliability drop: {stop_a_name} → {stop_b_name} ({abs(drop):.0f} pp decline)"
            else:
                cliff_text = "No single stop-to-stop reliability cliff exceeds 10 percentage points — degradation is gradual."
        else:
            cliff_text = "Not enough data to calculate reliability cliff."

    # Layout
    with st.expander("📋 Key Findings from This Analysis", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Trips Analyzed", trips_analyzed)
        col2.metric("Operating Days", operating_days)
        col3.metric("Network On-Time Rate", f"{weighted_reliability:.1f}%")
        
        c2_1, c2_2 = st.columns(2)
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
        st.info(cliff_text)

# ==============================================================================
# 6.4. ANALYTICS CHART BUILDERS
# ==============================================================================
def build_dow_chart(trip_stats):
    fig = go.Figure()
    valid_dows = sorted(list(set(d for d in trip_stats['per_trip_dow'] if d is not None)))
    for dow in valid_dows:
        delays = [trip_stats['per_trip_mean_delay'][i]/60 for i, d in enumerate(trip_stats['per_trip_dow']) if d == dow and not np.isnan(trip_stats['per_trip_mean_delay'][i])]
        if len(delays) < 3: continue
        fig.add_trace(go.Box(
            y=delays, name=DOW_LABELS[dow], boxpoints='outliers',
            marker=dict(size=6, opacity=0.7), line=dict(color=TTC_RED),
            fillcolor='rgba(218,37,29,0.15)', showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=[DOW_LABELS[d] for d in sorted(set(valid_dows))], y=[0]*len(valid_dows), mode='lines',
        line=dict(dash='dash', color='#555555', width=1.2),
        showlegend=False, hoverinfo='skip'
    ))
    fig.update_layout(
        title="Per-Trip Mean Delay by Day of Week",
        xaxis_title="Day of Week", yaxis_title="Mean Trip Delay (min)",
        template="plotly_white", yaxis_zeroline=True,
        yaxis=dict(zerolinecolor='#AAAAAA', zerolinewidth=1)
    )
    return fig

def build_date_trend_chart(trip_stats):
    date_dict = {}
    for d, delay in zip(trip_stats['per_trip_date'], trip_stats['per_trip_mean_delay']):
        if d and not np.isnan(delay):
            date_dict.setdefault(d, []).append(delay/60)
    sorted_dates = sorted(date_dict.keys())
    if len(sorted_dates) < 2: return go.Figure()
    
    daily_medians = [np.median(date_dict[d]) for d in sorted_dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_dates, y=daily_medians, mode='lines+markers',
        marker=dict(size=8, color=TTC_RED, symbol='circle'),
        line=dict(color=TTC_RED, width=1.8), name='Daily Median Deviation',
        hovertemplate="Date: %{x}<br>Median Deviation: %{y:.1f} min<extra></extra>"
    ))
    if len(sorted_dates) >= 7:
        rolling = pd.Series(daily_medians).rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=sorted_dates, y=rolling, mode='lines',
            line=dict(dash='dot', color='#0072B2', width=1.5),
            name='7-Day Rolling Median'
        ))
    fig.add_trace(go.Scatter(
        x=sorted_dates, y=[0]*len(sorted_dates), mode='lines',
        line=dict(dash='dash', color='#555555', width=1.2),
        showlegend=False, hoverinfo='skip'
    ))
    fig.update_layout(
        title="Daily Median Deviation Over Analysis Period",
        xaxis_title="Operating Date", yaxis_title="Median Deviation (min)",
        template="plotly_white"
    )
    return fig

def build_departure_scatter(trip_stats):
    hours, delays, dates = [], [], []
    for h, d, dt in zip(trip_stats['per_trip_hour'], trip_stats['per_trip_mean_delay'], trip_stats['per_trip_date']):
        if h is not None and not np.isnan(d):
            hours.append(h)
            delays.append(d/60)
            dates.append(dt)
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=delays, mode='markers',
        marker=dict(size=10, color=TTC_RED, symbol='circle', opacity=0.65, line=dict(width=0.8, color='#FFFFFF')),
        name='Individual Trips', customdata=dates,
        hovertemplate="Departure: %{x:.1f}h<br>Mean Delay: %{y:.1f} min<br>Date: %{customdata}<extra></extra>"
    ))
    if len(hours) >= 5:
        coeffs = np.polyfit(hours, delays, 1)
        x_line = np.linspace(min(hours), max(hours), 200)
        fig.add_trace(go.Scatter(
            x=x_line, y=np.polyval(coeffs, x_line), mode='lines',
            line=dict(dash='dot', color='#0072B2', width=1.8),
            name='Linear Trend', showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=[min(hours), max(hours)] if hours else [0, 24], y=[0, 0], mode='lines',
        line=dict(dash='dash', color='#555555', width=1.2),
        showlegend=False, hoverinfo='skip'
    ))
    fig.update_layout(
        title="Per-Trip Mean Delay vs Departure Hour",
        xaxis_title="Trip Departure Hour (0–24)", yaxis_title="Mean Delay (min)",
        template="plotly_white"
    )
    return fig

def build_stop_time_heatmap(trip_stats):
    buckets = {b[0]: [] for b in TIME_BUCKETS}
    for line in trip_stats['mode_b_lines']:
        anc = line.get('anchor_sec')
        if anc is None: continue
        hour = (float(anc) % 86400.0) / 3600.0
        for name, s, e in TIME_BUCKETS:
            if s <= hour < e:
                buckets[name].append(line)
                break

    used_buckets = []
    bucket_labels = []
    for name, _s, _e in TIME_BUCKETS:
        if len(buckets[name]) >= 3:
            used_buckets.append(name)
            bucket_labels.append(f"{name}\n(n={len(buckets[name])})")

    matrix = []
    stop_order = trip_stats['stop_order']
    stop_names = trip_stats['stop_names']
    rel_sec_map = trip_stats['rel_sec_map']

    for b_name in used_buckets:
        row = []
        lines = buckets[b_name]
        for sid in stop_order:
            delays = []
            for line in lines:
                sd = line.get('stop_delays', {})
                if sid in sd and sid in rel_sec_map:
                    delays.append((sd[sid] - rel_sec_map[sid])/60)
            if len(delays) >= 2:
                row.append(np.median(delays))
            else:
                row.append(np.nan)
        matrix.append(row)

    stop_labels = [clean_stop_name(stop_names[sid]) for sid in stop_order]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=stop_labels, y=bucket_labels,
        colorscale='RdBu_r', zmid=0,
        colorbar=dict(title="Deviation (min)", ticksuffix=" min"),
        hovertemplate="Stop: %{x}<br>Period: %{y}<br>Median Deviation: %{z:.1f} min<extra></extra>"
    ))
    fig.update_layout(
        title="Median Deviation by Stop and Time of Day",
        xaxis=dict(tickangle=45, automargin=True),
        height=max(380, 120 + len(used_buckets) * 70),
        template="plotly_white"
    )
    return fig

def build_stop_dow_heatmap(trip_stats):
    dow_dict = {d: [] for d in range(7)}
    for i, d in enumerate(trip_stats['per_trip_dow']):
        if d is not None:
            dow_dict[d].append(trip_stats['mode_b_lines'][i])

    used_dows = []
    row_labels = []
    for d in range(7):
        if len(dow_dict[d]) >= 3:
            used_dows.append(d)
            row_labels.append(f"{DOW_LABELS[d]}\n(n={len(dow_dict[d])})")

    matrix = []
    stop_order = trip_stats['stop_order']
    stop_names = trip_stats['stop_names']
    rel_sec_map = trip_stats['rel_sec_map']

    for d in used_dows:
        row = []
        lines = dow_dict[d]
        for sid in stop_order:
            delays = []
            for line in lines:
                sd = line.get('stop_delays', {})
                if sid in sd and sid in rel_sec_map:
                    delays.append((sd[sid] - rel_sec_map[sid])/60)
            if len(delays) >= 2:
                row.append(np.median(delays))
            else:
                row.append(np.nan)
        matrix.append(row)

    stop_labels = [clean_stop_name(stop_names[sid]) for sid in stop_order]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=stop_labels, y=row_labels,
        colorscale='RdBu_r', zmid=0,
        colorbar=dict(title="Deviation (min)", ticksuffix=" min"),
        hovertemplate="Stop: %{x}<br>Day: %{y}<br>Median Deviation: %{z:.1f} min<extra></extra>"
    ))
    fig.update_layout(
        title="Median Deviation by Stop and Day of Week",
        xaxis=dict(tickangle=45, automargin=True),
        height=max(380, 120 + len(used_dows) * 70),
        template="plotly_white"
    )
    return fig
    
def build_delay_variance_chart(trip_stats):
    fig = go.Figure()
    stop_order = trip_stats['stop_order']
    stop_names = trip_stats['stop_names']
    per_stop_delays = trip_stats['per_stop_delays']
    used_stops = []
    for sid in stop_order:
        if sid in per_stop_delays and len(per_stop_delays[sid]) >= 2:
            y_vals = [d/60 for d in per_stop_delays[sid]]
            label = clean_stop_name(stop_names[sid])
            full_name = stop_names[sid]
            used_stops.append(label)
            fig.add_trace(go.Box(
                y=y_vals, name=label, boxpoints='outliers',
                marker=dict(size=5, color=TTC_RED, opacity=0.6),
                line=dict(color=TTC_RED), fillcolor='rgba(218,37,29,0.15)',
                showlegend=False,
                hovertemplate=f"<b>{full_name}</b><br>Delay: %{{y:.1f}} min<extra></extra>"
            ))
    fig.add_hline(y=0, line=dict(dash='dash', color='#AAAAAA', width=1))
    fig.update_layout(
        title="Delay Distribution by Stop (Box-and-Whisker)",
        xaxis=dict(tickangle=45, automargin=True),
        yaxis_title="Delay from Schedule (min)",
        yaxis_zeroline=True, yaxis=dict(zerolinecolor='#AAAAAA', zerolinewidth=1),
        template="plotly_white",
        height=max(450, 350 + len(used_stops) * 8)
    )
    return fig

def build_equity_scatter(stops_df, equity_gdf, equity_field, metric_label):
    stops_gdf = gpd.GeoDataFrame(
        stops_df.copy(),
        geometry=gpd.points_from_xy(stops_df['true_lon'], stops_df['true_lat']),
        crs=LATLON_PROJ
    )
    joined = gpd.sjoin(
        stops_gdf[['stop_name','route_id','reliability','sample_size','geometry']],
        equity_gdf[['area_name', equity_field, 'geometry']],
        how='left', predicate='within'
    )
    joined = joined.dropna(subset=[equity_field, 'reliability'])
    n_total = len(stops_df[stops_df['stop_lat'].notna()])
    n_joined = len(joined)

    unique_routes = sorted(joined['route_id'].astype(str).unique())
    fig = go.Figure()

    for i, route in enumerate(unique_routes):
        color = WCAG_ROUTE_COLORS[i % len(WCAG_ROUTE_COLORS)]
        route_data = joined[joined['route_id'].astype(str) == route]
        
        # Format legend nicely (avoiding "Route Streetcar Network" redundancy)
        legend_name = route if "Network" in str(route) else f"Route {route}"
        
        fig.add_trace(go.Scatter(
            x=route_data[equity_field], y=route_data['reliability'],
            mode='markers', name=legend_name,
            marker=dict(
                size=12, 
                color=color, 
                symbol="circle", 
                opacity=0.80, 
                line=dict(width=1.0, color='#FFFFFF')
            ),
            text=route_data['stop_name'] + ' — ' + route_data['area_name'].fillna('Outside boundary'),
            hovertemplate="%{text}<br>" + metric_label + ": %{x:.1f}<br>Reliability: %{y:.1f}%<extra></extra>"
        ))

    x_all = joined[equity_field].values.astype(float)
    y_all = joined['reliability'].values.astype(float)
    mask = ~(np.isnan(x_all) | np.isnan(y_all))
    if mask.sum() >= 3:
        coeffs = np.polyfit(x_all[mask], y_all[mask], 1)
        x_line = np.linspace(x_all[mask].min(), x_all[mask].max(), 200)
        fig.add_trace(go.Scatter(
            x=x_line, y=np.polyval(coeffs, x_line), mode='lines',
            line=dict(dash='dot', color='#2C3E50', width=1.5),
            name='Overall Trend', showlegend=True, hoverinfo='skip'
        ))

    fig.update_layout(
        title=f"Stop Reliability vs {metric_label}  —  N = {n_joined} of {n_total} stops",
        xaxis_title=metric_label, yaxis_title="On-Time Reliability (%)",
        yaxis=dict(range=[0, 100]), template="plotly_white",
        legend=dict(title="Route", x=1.02, xanchor='left')
    )
    return fig

# ==============================================================================
# 6.5. RECALIBRATION LOGIC
# ==============================================================================
def compute_recalibration(st_filtered, actual_relative_times, target_percentile):
    data = []
    for row in st_filtered.itertuples():
        stop_id_str = str(row.stop_id)
        if stop_id_str not in actual_relative_times or len(actual_relative_times[stop_id_str]) < 3:
            continue
        delays_sec = [t - row.relative_sec for t in actual_relative_times[stop_id_str]]
        adjustment_sec = np.percentile(delays_sec, target_percentile)
        new_arrival_sec = row.arrival_sec + adjustment_sec
        
        total_seconds = int(round(new_arrival_sec))
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        gtfs_time_str = f"{h:02d}:{m:02d}:{s:02d}"
        
        data.append({
            'stop_name': row.stop_name,
            'stop_id': stop_id_str,
            'current_schedule': row.arrival_time,
            'suggested_schedule': gtfs_time_str,
            'adjustment_sec': round(adjustment_sec),
            'adjustment_min': round(adjustment_sec / 60, 1),
            'sample_size': len(actual_relative_times[stop_id_str]),
            'shape_dist_traveled': row.shape_dist_traveled
        })
        
    if not data:
        return None
        
    return pd.DataFrame(data).sort_values('shape_dist_traveled')

def generate_gtfs_stop_times_content(recal_df, raw_pipeline_data):
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    title_info = raw_pipeline_data.get('title_info', '')
    
    lines = [
        "# TTC ScheduleWatch — Suggested Schedule Adjustment",
        f"# Analysis: {title_info}",
        f"# Generated: {now_str}",
        "# NOTE: One adjusted schedule sequence for the analyzed signature only.",
        "# Not a complete GTFS feed. Reference use only.",
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled"
    ]
    
    for idx, row in enumerate(recal_df.itertuples(), start=1):
        lines.append(f"ADJUSTED_SCHEDULE_1,{row.suggested_schedule},{row.suggested_schedule},{row.stop_id},{idx},{row.shape_dist_traveled:.4f}")
        
    return "\n".join(lines)

def render_recalibration_section(tab_id):
    if st.session_state.analysis_results is None:
        return
    if st.session_state.analysis_results.get('is_multi', False):
        return

    st.markdown("---")
    st.markdown("#### 📅 Schedule Recalibration")
    st.caption(f"Analysis: {st.session_state.raw_pipeline_data['title_info']}")

    st.markdown("""
Suggests adjusted stop arrival times based on observed historical performance.
The **target percentile** controls how conservative the new schedule is:
- **50th (median):** Minimises added journey time. Half of trips will still be over 0 seconds late.
- **75th:** ~75% of trips appear on-time or early. Moderate buffer.
- **85th:** ~85% of trips appear on-time or early.
- **95th:** Near-universal on-time appearance at the cost of longer scheduled journey times.
""")

    target_pct = st.slider(
        "Target Percentile",
        min_value=50, max_value=95, value=50, step=5,
        key=f"recal_percentile_slider_{tab_id}",
        help="Higher = more trips appear on-time, but scheduled journey times increase."
    )

    recal_df = compute_recalibration(
        st.session_state.raw_pipeline_data['st_filtered'],
        st.session_state.raw_pipeline_data['actual_relative_times'],
        target_pct
    )

    if recal_df is None:
        st.warning("Insufficient data to compute recalibration (need at least 3 observations per stop).")
        return

    st.markdown(f"**{len(recal_df)} stops with sufficient data — {target_pct}th percentile target**")

    st.dataframe(
        recal_df[['stop_name', 'current_schedule', 'suggested_schedule', 'adjustment_min', 'sample_size']],
        column_config={
            'stop_name': st.column_config.TextColumn("Stop"),
            'current_schedule': st.column_config.TextColumn("Current GTFS Time"),
            'suggested_schedule': st.column_config.TextColumn("Suggested Time"),
            'adjustment_min': st.column_config.NumberColumn("Adjustment (min)", format="%.1f"),
            'sample_size': st.column_config.NumberColumn("Observations", format="%d"),
        },
        use_container_width=True,
        hide_index=True
    )

    gtfs_content = generate_gtfs_stop_times_content(recal_df, st.session_state.raw_pipeline_data)
    
    st.download_button(
        label="⬇️ Download as GTFS stop_times.txt",
        data=gtfs_content,
        file_name="suggested_stop_times.txt",
        mime="text/plain",
        key=f"recal_dl_btn_{tab_id}",
        help="GTFS-format stop_times.txt with suggested adjusted schedule. Not a complete GTFS feed — reference only."
    )

    st.caption("When 'Align to First Observed Stop' mode was active, adjustments incorporate actual departure timing rather than pure GTFS-scheduled departure.")

# ==============================================================================
# 7. MAIN UI & TAB LAYOUT
# ==============================================================================
st.title("TTC ScheduleWatch")
st.caption("Open-data analysis tool of TTC streetcar performance versus published schedules (GTFS). Visualizes streetcar locations between March 15 - May 2 2026. This website is not affiliated with the TTC. ")

parquet_path = get_parquet_path()
available_routes = get_available_routes(parquet_path)
stops, trips, stop_times, shapes = load_gtfs()

if not st.session_state.show_settings:
    if st.button("⚙️ Run Custom Analysis", type="primary"):
        st.session_state.show_settings = True
        # -------------------------------------------------------------------
        # Shadow State Restoration Logic (Pushes saved values back to widgets)
        # -------------------------------------------------------------------
        if 'saved_ui_state' in st.session_state:
            for k, v in st.session_state.saved_ui_state.items():
                st.session_state[k] = v
        st.rerun()  # <--- PUT THIS BACK
        
if st.session_state.show_settings:
    with st.container():
        st.markdown("### ⚙️ Run Custom Analysis")
        render_filter_panel(available_routes, parquet_path, trips, stop_times, stops, shapes)
        st.markdown("---")

if st.session_state.analysis_results is not None:
    render_insights_panel(st.session_state.raw_pipeline_data, st.session_state.analysis_results)

# Consolidate top-level to 3 isolated zones
tab_map, tab_charts, tab_recal = st.tabs([
    "🗺️ Route Reliability Map",
    "📈 Charts & Analytics",
    "📅 Schedule Recalibration",
])

with tab_map:
    @st.fragment
    def render_map_tab():
        # Additive Map-Specific Accessibility Toggle
        st.selectbox(
            "👁️ Map Color Theme",
            options=["Default (Classic Red-Green)", "Accessible"],
            key="color_theme",
        )
        
        if not st.session_state.analysis_results:
            precomputed = load_precomputed_network()
            if precomputed:
                st.info("🗺️ **Showing Default Network Reliability View.** All-Day Weekdays, All Routes. Click the **⚙️ Run Custom Analysis** button above to run a custom analysis.")
                stops_df = pd.DataFrame(precomputed['stops'])
                segments_df = gpd.GeoDataFrame.from_features(precomputed['segments']['features'])
                
                # Inject anchors so the landing page colors lock to 0% - 100%
                stops_df, segments_df = inject_legend_anchors(stops_df, segments_df)
                
                # Calls generate_kepler_config dynamically to apply theme changes instantly
                map_instance = KeplerGl(height=600, data={"stops": stops_df, "segments": segments_df}, config=generate_kepler_config())
                keplergl_static(map_instance, center_map=True)
            else:
                st.info("🗺️ **Map View is Empty.** Please click the **⚙️ Run Custom Analysis** button above to run an analysis.")
        else:
            st.markdown(f"**Configuration:** {st.session_state.raw_pipeline_data['title_info']}")
            results = st.session_state.analysis_results
            if 'segments_df' in results and not results['segments_df'].empty:
                # Calls generate_kepler_config dynamically instead of using the static saved version
                map_instance = KeplerGl(height=600, data={"stops": results['stops_df'], "segments": results['segments_df']}, config=generate_kepler_config())
                keplergl_static(map_instance, center_map=True)
            else:
                st.warning("Spatial geometry could not be built for this route.")

        # -------------------------------------------------------------------------
        # ADDITIVE ACCESSIBLE FALLBACK VIEW FOR PRECOMPUTED AND CUSTOM MAPS
        # -------------------------------------------------------------------------
        active_stops_df = None
        active_segments_df = None
        is_custom = False

        if st.session_state.analysis_results:
            active_stops_df = st.session_state.analysis_results.get('stops_df')
            active_segments_df = st.session_state.analysis_results.get('segments_df')
            is_custom = True
        else:
            pre_network = load_precomputed_network()
            if pre_network:
                active_stops_df = pd.DataFrame(pre_network['stops'])
                active_segments_df = gpd.GeoDataFrame.from_features(pre_network['segments']['features'])

        if active_stops_df is not None and not active_stops_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 View Map Stops and Segments as an Accessible Table", expanded=False):
                st.caption("This collapsible data table acts as an accessible, high-contrast text alternative to the visual map representation above, supporting both screen readers and keyboard navigation.")
                if is_custom:
                    st.caption("Showing performance statistics for the currently calculated route analysis.")
                else:
                    st.caption("Showing default network overview statistics.")
                
                # Filter out non-geographic anchor legends
                clean_display_stops = active_stops_df[active_stops_df['stop_lat'].notna()].copy()
                if 'stop_name' in clean_display_stops.columns:
                    # Ensure the full, untruncated stop name is presented to prevent informational loss
                    clean_display_stops['stop_name'] = clean_display_stops['stop_name'].astype(str)

                st.markdown("##### Route Stops")
                
                # Safe column check: only show 'sample_size' if it exists in the active dataset
                stop_cols_to_show = [col for col in ['stop_name', 'reliability', 'sample_size'] if col in clean_display_stops.columns]
                
                st.dataframe(
                    clean_display_stops[stop_cols_to_show],
                    column_config={
                        'stop_name': st.column_config.TextColumn("Stop Station Name"),
                        'reliability': st.column_config.NumberColumn("On-Time Reliability Rate", format="%.1f%%"),
                        'sample_size': st.column_config.NumberColumn("Measured Runs (Sample Size)", format="%d")
                    },
                    use_container_width=True,
                    hide_index=True
                )

                if active_segments_df is not None and not active_segments_df.empty:
                    st.markdown("##### Corridor Segments")
                    display_segs = pd.DataFrame(active_segments_df).copy()
                    if 'geometry' in display_segs.columns:
                        display_segs = display_segs.drop(columns=['geometry'])
                    
                    # Safe column check: prevents KeyError when 'sample_size' is missing in precomputed JSON
                    seg_cols_to_show = [col for col in ['segment', 'avg_reliability', 'sample_size'] if col in display_segs.columns]
                    
                    st.dataframe(
                        display_segs[seg_cols_to_show],
                        column_config={
                            'segment': st.column_config.TextColumn("Inter-Stop Route Segment"),
                            'avg_reliability': st.column_config.NumberColumn("Average Corridor Segment Reliability", format="%.1f%%"),
                            'sample_size': st.column_config.NumberColumn("Aggregate Runs Measured", format="%d")
                        },
                        use_container_width=True,
                        hide_index=True
                    )

        st.markdown("---")
        st.markdown("### 🏘️ Equity Context Map")
        st.markdown("Neighbourhood-level equity indicators from Statistics Canada 2021 Census and City of Toronto Open Data, overlaid with transit reliability. **Use the layer panel (top-left of the map) to toggle between equity indicators.** Transit segments and stops reflect the current analysis if one has been run, otherwise show the all-routes precomputed network.")
        
        with st.expander("📖 Layer Guide", expanded=False):
            st.markdown("""
| Layer Name | What It Shows |
|---|---|
| Median Household Income ($) | Neighbourhood median household income |
| Low-Income Households (%) | Share of residents below low-income measure (after tax) |
| Transit Commuters (%) — Transit Dependence | Share of employed residents commuting by transit |
| Visible Minority Population (%) | Share identifying as visible minority |
| Recent Immigrants — Last 5 Years (%) | Share who immigrated within 5 years |
| Seniors 65+ (%) | Share of population aged 65 or older |
""")
        
        equity_gdf = load_equity_data()
        
        t_stops = None
        t_segs = None
        
        # Changed: Remove the 'is_multi' restriction so any active custom analysis (single or multi) is utilized
        if st.session_state.analysis_results is not None:
            t_stops = st.session_state.analysis_results.get('stops_df')
            t_segs = st.session_state.analysis_results.get('segments_df')
        else:
            precomputed = load_precomputed_network()
            if precomputed:
                t_stops = pd.DataFrame(precomputed['stops'])
                t_segs = gpd.GeoDataFrame.from_features(precomputed['segments']['features'])
                t_stops, t_segs = inject_legend_anchors(t_stops, t_segs)
                
        equity_config = generate_equity_kepler_config()
        data_dict = {"equity": equity_gdf}
        
        if t_stops is not None and t_segs is not None and not t_stops.empty and not t_segs.empty:
            data_dict["stops"] = t_stops
            data_dict["segments"] = t_segs
        else:
            layers = equity_config["config"]["visState"]["layers"]
            equity_config["config"]["visState"]["layers"] = [l for l in layers if l["id"] not in ("segments", "stops")]
            
            layer_order = equity_config["config"]["visState"]["layerOrder"]
            equity_config["config"]["visState"]["layerOrder"] = [lo for lo in layer_order if lo not in ("segments", "stops")]
            
            tooltip_fields = equity_config["config"]["visState"]["interactionConfig"]["tooltip"]["fieldsToShow"]
            if "segments" in tooltip_fields:
                del tooltip_fields["segments"]
            if "stops" in tooltip_fields:
                del tooltip_fields["stops"]
                
        equity_map = KeplerGl(height=650, data=data_dict, config=equity_config)
        keplergl_static(equity_map, center_map=True)

        # -------------------------------------------------------------------------
        # ADDITIVE ACCESSIBLE FALLBACK VIEW FOR CENSUS NEIGHBOURHOOD EQUITY DATA
        # -------------------------------------------------------------------------
        if equity_gdf is not None and not equity_gdf.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🏘️ View Neighbourhood Census Equity Metrics as an Accessible Table", expanded=False):
                st.caption("This collapsible database lists Statistics Canada 2021 Census equity indicators across municipal neighbourhoods, serving as an accessible high-contrast text alternative to the multi-layered visual map above.")
                
                display_equity = pd.DataFrame(equity_gdf).copy()
                if 'geometry' in display_equity.columns:
                    display_equity = display_equity.drop(columns=['geometry'])
                
                display_equity = display_equity.sort_values('area_name')
                st.dataframe(
                    display_equity[[
                        'area_name', 'median_income', 'low_income_pct', 
                        'transit_commute_pct', 'visible_minority_pct', 
                        'recent_immigrant_pct', 'senior_pct'
                    ]],
                    column_config={
                        'area_name': st.column_config.TextColumn("Neighbourhood Name"),
                        'median_income': st.column_config.NumberColumn("Median Household Income", format="$%d"),
                        'low_income_pct': st.column_config.NumberColumn("Low-Income Households", format="%.1f%%"),
                        'transit_commute_pct': st.column_config.NumberColumn("Transit Commute share", format="%.1f%%"),
                        'visible_minority_pct': st.column_config.NumberColumn("Visible Minority Share", format="%.1f%%"),
                        'recent_immigrant_pct': st.column_config.NumberColumn("Recent Immigrants (Last 5 Yrs)", format="%.1f%%"),
                        'senior_pct': st.column_config.NumberColumn("Seniors 65+", format="%.1f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )
        
        st.caption("Data: Statistics Canada 2021 Census via City of Toronto Neighbourhood Profiles (open.toronto.ca). All data used under open government licence.")
    
    render_map_tab()

with tab_charts:
    @st.fragment
    def render_charts_tab():
        # Split the consolidated tab back up into the normal 3 separate views
        tab_spaghetti, tab_stats, tab_analytics = st.tabs([
            "🍝 Time-Distance Chart",
            "📊 Density Chart",
            "📈 Analytics",
        ])

        with tab_spaghetti:
            if not st.session_state.analysis_results:
                st.info("🍝 **Time-Distance Chart is Empty.** Please click the **⚙️ Run Custom Analysis** button above to run an analysis.")
            elif st.session_state.analysis_results.get('is_multi', False):
                st.warning("⚠️ **Charts Disabled.** Detailed trip visualizations are only available when analyzing a single route.")
            else:
                st.markdown(f"**Configuration:** {st.session_state.raw_pipeline_data['title_info']}")
                
                # Plotly renders completely natively. Zooming/panning runs in-browser only (No Python execution).
                st.plotly_chart(
                    st.session_state.analysis_results['fig_B'],
                    use_container_width=True,
                    height=900,
                    config=PLOTLY_CONFIG,
                    key="fig_b_chart"
                )

                st.info(f"Trips with missing lines between points reflect periods where GPS ping frequency was under {MAX_ALLOWED_PING_GAP_SEC} seconds. These periods are left out in calculations.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # ISOLATED FRAGMENT: Changing dropdowns here triggers instantly, without reloading the chart
                @st.fragment
                def render_point_lookup():
                    with st.expander("📍 Look up coordinates & generate Google Maps Links", expanded=False):
                        st.markdown('<div class="sr-only">Accessible text alternative to the Time-Distance chart above.</div>', unsafe_allow_html=True)
                        st.caption("Select a specific trip and time below to generate a Google Maps link for that exact location. Fully navigable via keyboard.")
                        raw_lines = st.session_state.raw_pipeline_data.get('mode_b_lines', [])
                        if raw_lines:
                            line_options = {idx: f"{item['op_date']} | Trip ID: {item['t_id']} (Departed: {item['start_time']})" for idx, item in enumerate(raw_lines)}
                            selected_line_idx = st.selectbox(
                                "Select Target Active Run", 
                                options=list(line_options.keys()), 
                                format_func=lambda idx: line_options[idx],
                                key="point_run_selector"
                            )
                            
                            chosen_line_data = raw_lines[selected_line_idx]
                            valid_point_coords = [(idx, abs_time, latitude, longitude) for idx, (abs_time, latitude, longitude) in enumerate(zip(chosen_line_data['abs_time'], chosen_line_data['lat'], chosen_line_data['lon'])) if latitude is not None and longitude is not None]
                            
                            if valid_point_coords:
                                point_options = {idx: f"Time: {abs_time} (Lat: {latitude:.5f}, Lon: {longitude:.5f})" for idx, abs_time, latitude, longitude in valid_point_coords}
                                selected_point_idx = st.selectbox(
                                    "Select Logged Telemetry Point", 
                                    options=list(point_options.keys()), 
                                    format_func=lambda idx: point_options[idx],
                                    key="point_coord_selector"
                                )
                                
                                final_lat = chosen_line_data['lat'][selected_point_idx]
                                final_lon = chosen_line_data['lon'][selected_point_idx]
                                
                                st.success(
                                    f"📍 **Coordinates Resolved:** "
                                    f"[**Click here to open this location in Google Maps**]"
                                    f"(https://www.google.com/maps/search/?api=1&query={final_lat},{final_lon})"
                                )
                            else:
                                st.info("No geospatial records are available for this specific run.")
                
                render_point_lookup()

        with tab_stats:
            if not st.session_state.analysis_results:
                st.info("📊 **Density Chart is Empty.** Please click the **⚙️ Run Custom Analysis** button above to run an analysis.")
            elif st.session_state.analysis_results.get('is_multi', False):
                st.warning("⚠️ **Charts Disabled.** Detailed density plots are only available when analyzing a single route.")
            else:
                st.markdown(f"**Configuration:** {st.session_state.raw_pipeline_data['title_info']}")
                st.plotly_chart(st.session_state.analysis_results['fig_A'], use_container_width=True, height=900, config=PLOTLY_CONFIG, key="fig_a_chart")

        with tab_analytics:
            has_analysis = st.session_state.analysis_results is not None
            is_multi     = st.session_state.analysis_results.get('is_multi', False) if has_analysis else False
            trip_stats   = (st.session_state.raw_pipeline_data.get('trip_stats')
                            if has_analysis and st.session_state.raw_pipeline_data else None)

            # Load data for the Equity chart (Custom Analysis OR Precomputed Network)
            active_equity_stops = None
            if has_analysis:
                active_equity_stops = st.session_state.analysis_results['stops_df']
            else:
                pre_network = load_precomputed_network()
                if pre_network:
                    active_equity_stops = pd.DataFrame(pre_network['stops'])
                    # Treat all stops uniformly under a single network-wide category
                    active_equity_stops['route_id'] = "Streetcar Network"

            # ── SECTION 3: EQUITY ─────────────────────────────────────────────────
            st.markdown("### 🏘️ Equity Analysis")

            try:
                equity_gdf     = load_equity_data()
                equity_available = (equity_gdf is not None and not equity_gdf.empty)
            except Exception:
                equity_available = False

            if not equity_available:
                st.info(
                    "🏘️ **Equity data is not available.** Upload "
                    "`equity_neighbourhoods.geojson` to the HuggingFace repository "
                    "to enable this section. See the Route Reliability Map tab for "
                    "upload instructions."
                )
            elif active_equity_stops is None:
                st.info(
                    "🏘️ **No reliability data available.** Run an analysis or allow the default network to load to enable the equity scatter chart."
                )
            else:
                if not has_analysis:
                    st.info("📈 **Showing Network-Wide Equity Analysis.** This view uses precomputed reliability data for the entire streetcar network. For detailed Temporal and Space-Time charts, please run a custom analysis for a specific route.")
                
                EQUITY_METRIC_OPTIONS = {
                    "Median Household Income ($)":          "median_income",
                    "Low-Income Households (%)":            "low_income_pct",
                    "Transit Commuters (%)":                "transit_commute_pct",
                    "Visible Minority Population (%)":      "visible_minority_pct",
                    "Recent Immigrants — Last 5 Yrs (%)":   "recent_immigrant_pct",
                    "Seniors 65+ (%)":                      "senior_pct",
                }

                st.caption(
                    "Each dot represents one stop, plotted against the equity indicator "
                    "for the neighbourhood it falls within. Dots are colour-coded and "
                    "shape-coded by route — both visual channels are used so the chart "
                    "remains readable for users with colour vision differences."
                )

                selected_label = st.selectbox(
                    "Equity metric to compare against stop reliability:",
                    options = list(EQUITY_METRIC_OPTIONS.keys()),
                    key     = "equity_metric_select"
                )
                selected_field = EQUITY_METRIC_OPTIONS[selected_label]

                stops_clean = active_equity_stops[active_equity_stops['stop_lat'].notna()].copy()

                fig_eq = build_equity_scatter(
                    stops_clean, equity_gdf, selected_field, selected_label
                )
                st.plotly_chart(fig_eq, use_container_width=True, config=PLOTLY_CONFIG, key="chart_equity")
                announce_sr(
                    f"Equity scatter chart rendered: stop reliability versus "
                    f"{selected_label}."
                )
                
                with st.expander("📋 View data as accessible table", expanded=False):
                    st.caption(f"Accessible data table for Equity Scatter ({selected_label})")
                    stops_gdf = gpd.GeoDataFrame(
                        stops_clean.copy(),
                        geometry=gpd.points_from_xy(stops_clean['true_lon'], stops_clean['true_lat']),
                        crs=LATLON_PROJ
                    )
                    joined = gpd.sjoin(
                        stops_gdf[['stop_name','route_id','reliability','geometry']],
                        equity_gdf[['area_name', selected_field, 'geometry']],
                        how='left', predicate='within'
                    )
                    joined = joined.dropna(subset=[selected_field, 'reliability']).sort_values('reliability')
                    joined = joined.rename(columns={
                        'stop_name': 'Stop Name', 'route_id': 'Route',
                        'area_name': 'Neighbourhood', selected_field: selected_label,
                        'reliability': 'Reliability (%)'
                    })
                    st.write(f"Showing {len(joined)} joined stops.")
                    st.dataframe(joined[['Stop Name', 'Route', 'Neighbourhood', selected_label, 'Reliability (%)']], hide_index=True)
                    
            # ── SECTION 1: TEMPORAL PATTERNS ──────────────────────────────────────
            st.markdown("### ⏱️ Temporal Patterns")
            
            if not has_analysis or is_multi or trip_stats is None:
                st.info(
                    "⏱️ **Temporal pattern charts are available for single-signature analyses "
                    "only.** Switch to single-signature mode in the settings panel above and "
                    "re-run the analysis to access day-of-week, date trend, and departure-"
                    "time breakdowns."
                )
            else:
                show_dow      = trip_stats['n_unique_dow'] >= 3
                show_date     = trip_stats['n_unique_dates'] >= 7
                show_dep_time = trip_stats['hour_range'] >= 0.5   # 30 minutes minimum
                
                any_temporal  = show_dow or show_date or show_dep_time
                
                if not any_temporal:
                    st.info(
                        "⏱️ **Temporal charts require more data variation than is present in "
                        "this analysis.** Specifically: day-of-week distribution needs data "
                        "from at least 3 distinct days of the week; date trend needs at least "
                        "7 different operating dates; departure-time scatter needs at least "
                        "30 minutes of variation in trip departure times. Broaden your day "
                        "or time-window filters and re-run to enable these charts."
                    )
                else:
                    if 'show_temporal' not in st.session_state:
                        st.session_state.show_temporal = False
                    if st.button("Generate Temporal Charts", key="btn_temporal",
                                 help="Computes day-of-week, date trend, and departure-time charts."):
                        st.session_state.show_temporal = True

                    if st.session_state.get('show_temporal'):
                        if show_dow:
                            st.markdown("#### Delay by Day of Week")
                            st.caption(
                                "Each box shows the distribution of per-trip mean delays for that "
                                "day. The horizontal dashed line marks the scheduled baseline (zero "
                                "delay). Outlier points are shown individually."
                            )
                            fig = build_dow_chart(trip_stats)
                            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="chart_dow")
                            announce_sr("Day-of-week delay distribution chart rendered.")
                            
                            with st.expander("📋 View data as accessible table", expanded=False):
                                st.caption("Accessible data table for Per-Trip Mean Delay by Day of Week")
                                data = []
                                for d in range(7):
                                    delays = [trip_stats['per_trip_mean_delay'][i]/60 for i, dow in enumerate(trip_stats['per_trip_dow']) if dow == d and not np.isnan(trip_stats['per_trip_mean_delay'][i])]
                                    if len(delays) >= 3:
                                        data.append({
                                            "Day": DOW_LABELS[d], "N Trips": len(delays),
                                            "Median Delay (min)": np.median(delays),
                                            "Min (min)": np.min(delays), "Max (min)": np.max(delays)
                                        })
                                st.dataframe(pd.DataFrame(data), hide_index=True)
                        else:
                            st.info(
                                "**Day-of-week chart unavailable.** This chart requires data from "
                                "at least 3 distinct days of the week. Your current analysis "
                                "covers fewer — expand the day filters in the settings panel."
                            )

                        st.markdown("---")

                        if show_date:
                            st.markdown("#### Daily Median Deviation Over Time")
                            st.caption(
                                "Each point represents the median per-trip deviation across all trips "
                                "on that operating date. The dotted blue line shows a 7-day rolling "
                                "average when sufficient dates are available."
                            )
                            fig = build_date_trend_chart(trip_stats)
                            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="chart_date")
                            announce_sr("Date trend chart rendered.")
                            
                            with st.expander("📋 View data as accessible table", expanded=False):
                                st.caption("Accessible data table for Daily Median Deviation Over Time")
                                date_dict = {}
                                for d, delay in zip(trip_stats['per_trip_date'], trip_stats['per_trip_mean_delay']):
                                    if d and not np.isnan(delay):
                                        date_dict.setdefault(d, []).append(delay/60)
                                data = [{"Date": d, "N Trips": len(date_dict[d]), "Median Deviation (min)": np.median(date_dict[d])} for d in sorted(date_dict.keys())]
                                st.dataframe(pd.DataFrame(data), hide_index=True)
                        else:
                            st.info(
                                "**Date trend chart unavailable.** This chart requires at least "
                                "7 different operating dates. Your current analysis spans fewer. "
                                "Broaden the date range of your underlying dataset to enable this."
                            )

                        st.markdown("---")

                        if show_dep_time:
                            st.markdown("#### Delay vs Trip Departure Hour")
                            st.caption(
                                "Each point represents one trip. The x-axis shows the approximate "
                                "hour at which the trip departed its origin stop. The dotted trend "
                                "line indicates the overall directional relationship."
                            )
                            fig = build_departure_scatter(trip_stats)
                            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="chart_dep")
                            announce_sr("Departure-time scatter chart rendered.")
                            
                            with st.expander("📋 View data as accessible table", expanded=False):
                                st.caption("Accessible data table for Delay vs Trip Departure Hour")
                                data = []
                                for h, d, dt in zip(trip_stats['per_trip_hour'], trip_stats['per_trip_mean_delay'], trip_stats['per_trip_date']):
                                    if h is not None and not np.isnan(d):
                                        data.append({"Departure Hour": round(h, 1), "Date": dt, "Mean Delay (min)": d/60})
                                if data:
                                    df_dep = pd.DataFrame(data).sort_values("Departure Hour")
                                    st.dataframe(df_dep, hide_index=True)
                        else:
                            st.info(
                                "**Departure-time scatter unavailable.** This chart requires at "
                                "least 30 minutes of variation in trip departure times across the "
                                "analyzed sample. All trips in this signature depart within a "
                                "narrower window — consider analyzing a broader time range."
                            )

            st.markdown("---")

            # ── SECTION 2: SPACE-TIME STRUCTURE ───────────────────────────────────
            st.markdown("### 🗂️ Space-Time Structure")

            if not has_analysis or is_multi or trip_stats is None:
                st.info(
                    "🗂️ **Space-time structure charts are available for single-signature "
                    "analyses only.** Re-run in single-signature mode to access delay "
                    "variability and heatmap breakdowns."
                )
            else:
                show_time_hm = trip_stats['hour_range'] >= 0.5 and not is_multi
                show_dow_hm  = trip_stats['n_unique_dow'] >= 3 and not is_multi

                if 'show_spacetime' not in st.session_state:
                    st.session_state.show_spacetime = False
                if st.button("Generate Space-Time Charts", key="btn_spacetime",
                             help="Computes delay variability and heatmap breakdowns."):
                    st.session_state.show_spacetime = True

                if st.session_state.get('show_spacetime'):

                    st.markdown("#### Delay Distribution by Stop (Box-and-Whisker)")
                    st.caption(
                        "Shows the full distribution of arrival delay (relative to schedule) "
                        "at each stop in route order. Boxes represent the interquartile range; "
                        "whiskers extend to 1.5× IQR; outlier points are shown individually. "
                        "The dashed line at zero marks the scheduled arrival time."
                    )
                    fig = build_delay_variance_chart(trip_stats)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="chart_variance")
                    announce_sr("Stop delay distribution box plot rendered.")
                    
                    with st.expander("📋 View data as accessible table", expanded=False):
                        st.caption("Accessible data table for Delay Distribution by Stop")
                        data = []
                        for sid in trip_stats['stop_order']:
                            if sid in trip_stats['per_stop_delays'] and len(trip_stats['per_stop_delays'][sid]) >= 2:
                                delays = [d/60 for d in trip_stats['per_stop_delays'][sid]]
                                data.append({
                                    "Stop": trip_stats['stop_names'][sid],
                                    "N Observations": len(delays),
                                    "Median Delay (min)": np.median(delays),
                                    "Std Dev (min)": np.std(delays),
                                    "Min (min)": np.min(delays),
                                    "Max (min)": np.max(delays)
                                })
                        st.dataframe(pd.DataFrame(data), hide_index=True)

                    st.markdown("---")

                    if show_time_hm:
                        st.markdown("#### Median Deviation — Stop × Time of Day")
                        st.caption(
                            "Each cell shows the median arrival deviation (minutes) at that stop "
                            "during that time period. Red indicates late arrivals; blue indicates "
                            "early arrivals; white indicates on-time performance. Grey cells have "
                            "fewer than 2 observations and are shown as not-a-number."
                        )
                        fig = build_stop_time_heatmap(trip_stats)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="chart_time_hm")
                        announce_sr("Stop by time-of-day deviation heatmap rendered.")
                        
                        with st.expander("📋 View data as accessible table", expanded=False):
                            st.caption("Accessible data table for Median Deviation by Stop and Time of Day")
                            buckets = {b[0]: [] for b in TIME_BUCKETS}
                            for line in trip_stats['mode_b_lines']:
                                anc = line.get('anchor_sec')
                                if anc is None: continue
                                hour = (float(anc) % 86400.0) / 3600.0
                                for name, s, e in TIME_BUCKETS:
                                    if s <= hour < e:
                                        buckets[name].append(line)
                                        break
                            used_buckets = [name for name, _, _ in TIME_BUCKETS if len(buckets[name]) >= 3]
                            df_data = {}
                            for b_name in used_buckets:
                                row_data = {}
                                for sid in trip_stats['stop_order']:
                                    stop_name = trip_stats['stop_names'][sid]
                                    delays = []
                                    for line in buckets[b_name]:
                                        sd = line.get('stop_delays', {})
                                        if sid in sd and sid in trip_stats['rel_sec_map']:
                                            delays.append((sd[sid] - trip_stats['rel_sec_map'][sid])/60)
                                    row_data[clean_stop_name(stop_name)] = round(np.median(delays), 1) if len(delays) >= 2 else np.nan
                                df_data[b_name] = row_data
                            df_table = pd.DataFrame.from_dict(df_data, orient='index')
                            st.dataframe(df_table)
                    else:
                        st.info(
                            "**Stop × time-of-day heatmap unavailable.** This chart requires "
                            "at least 30 minutes of variation in trip departure times across the "
                            "analyzed sample. The current analysis window is too narrow to reveal "
                            "meaningful time-of-day patterns. Broaden the time filter to enable."
                        )

                    st.markdown("---")

                    if show_dow_hm:
                        st.markdown("#### Median Deviation — Stop × Day of Week")
                        st.caption(
                            "Each cell shows the median arrival deviation at that stop on that day "
                            "of the week. Same colour encoding as the time-of-day heatmap above."
                        )
                        fig = build_stop_dow_heatmap(trip_stats)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="chart_dow_hm")
                        announce_sr("Stop by day-of-week deviation heatmap rendered.")
                        
                        with st.expander("📋 View data as accessible table", expanded=False):
                            st.caption("Accessible data table for Median Deviation by Stop and Day of Week")
                            dow_dict = {d: [] for d in range(7)}
                            for i, d in enumerate(trip_stats['per_trip_dow']):
                                if d is not None:
                                    dow_dict[d].append(trip_stats['mode_b_lines'][i])
                            used_dows = [d for d in range(7) if len(dow_dict[d]) >= 3]
                            df_data = {}
                            for d in used_dows:
                                row_data = {}
                                for sid in trip_stats['stop_order']:
                                    stop_name = trip_stats['stop_names'][sid]
                                    delays = []
                                    for line in dow_dict[d]:
                                        sd = line.get('stop_delays', {})
                                        if sid in sd and sid in trip_stats['rel_sec_map']:
                                            delays.append((sd[sid] - trip_stats['rel_sec_map'][sid])/60)
                                    row_data[clean_stop_name(stop_name)] = round(np.median(delays), 1) if len(delays) >= 2 else np.nan
                                df_data[DOW_LABELS[d]] = row_data
                            df_table = pd.DataFrame.from_dict(df_data, orient='index')
                            st.dataframe(df_table)
                    else:
                        st.info(
                            "**Stop × day-of-week heatmap unavailable.** This chart requires "
                            "data from at least 3 distinct days of the week. Expand the day "
                            "filters in the settings panel and re-run to enable."
                        )

            st.markdown("---")
                    
    render_charts_tab()

with tab_recal:
    @st.fragment
    def render_recal_tab():
        has_analysis = st.session_state.analysis_results is not None
        is_multi     = st.session_state.analysis_results.get('is_multi', False) if has_analysis else False

        if not has_analysis:
            st.info(
                "📅 **No analysis loaded.** Use the **⚙️ Run Custom Analysis"
                "** button above to run an analysis. Schedule recalibration "
                "will become available once a single-signature analysis is complete."
            )
        elif is_multi:
            st.warning(
                "⚠️ **Schedule recalibration is only available for single-signature "
                "analyses.** The current result is a multi-route network analysis. "
                "Switch to single-signature mode and re-run to access recalibration."
            )
        else:
            st.markdown("#### 📅 Schedule Recalibration")
            st.caption(f"Analysis: {st.session_state.raw_pipeline_data['title_info']}")

            st.warning(
                "⚠️ **Schedule recalibration produces the most meaningful results "
                "when applied to a homogeneous group of trips**. This is ideally a single "
                "headsign operating within a consistent, narrow time window (e.g., "
                "AM peak only). Applying recalibration to a broad multi-hour dataset "
                "will produce adjustments that average across very different operating "
                "conditions and may be suboptimal for any specific time period. Use "
                "the time and day filters to narrow your analysis before downloading."
            )

            st.markdown("""
        The **target percentile** controls how conservative the adjusted schedule is:
        - **50th (median):** Minimises added journey time. Half of trips will still appear late relative to the new schedule.
        - **75th:** Approximately 75% of trips appear on-time or early. Moderate buffer.
        - **85th:** Industry-standard target. Approximately 85% of trips appear on-time or early.
        - **95th:** Highly conservative. Near-universal on-time performance at the cost of significantly longer scheduled journey times.
        """)

            target_pct = st.slider(
                "Target Percentile",
                min_value=50, max_value=95, value=85, step=5,
                key="recal_percentile_slider_tab",
                help="Higher = more trips appear on-time, but official journey times increase."
            )

            recal_df = compute_recalibration(
                st.session_state.raw_pipeline_data['st_filtered'],
                st.session_state.raw_pipeline_data['actual_relative_times'],
                target_pct
            )

            if recal_df is None:
                st.warning(
                    "Insufficient data to compute recalibration. Each stop requires at "
                    "least 3 observed arrivals. Try broadening your date range or day "
                    "filters to capture more historical runs."
                )
            else:
                st.markdown(
                    f"**{len(recal_df)} stops with sufficient data** — "
                    f"{target_pct}th percentile target applied."
                )

                st.dataframe(
                    recal_df[['stop_name','current_schedule','suggested_schedule',
                              'adjustment_min','sample_size']],
                    column_config={
                        'stop_name':          st.column_config.TextColumn("Stop"),
                        'current_schedule':   st.column_config.TextColumn("Current GTFS Time"),
                        'suggested_schedule': st.column_config.TextColumn("Suggested Time"),
                        'adjustment_min':     st.column_config.NumberColumn(
                                                  "Adjustment (min)", format="%.1f"),
                        'sample_size':        st.column_config.NumberColumn(
                                                  "Observations", format="%d"),
                    },
                    use_container_width=True,
                    hide_index=True
                )

                gtfs_content = generate_gtfs_stop_times_content(
                    recal_df, st.session_state.raw_pipeline_data
                )
                st.download_button(
                    label="⬇️ Download Adjusted Schedule as GTFS stop_times.txt",
                    data=gtfs_content,
                    file_name="suggested_stop_times.txt",
                    mime="text/plain",
                    key="recal_dl_btn_tab",
                    help=(
                        "Downloads a GTFS-format stop_times.txt file with the suggested "
                        "adjusted schedule. This covers one schedule signature only and "
                        "is not a complete GTFS feed. For reference use only."
                    )
                )

                st.caption(
                    "When 'Align to First Observed Stop' mode was "
                    "active, adjustments incorporate actual GPS-observed departure "
                    "timing rather than the GTFS-scheduled departure time."
                )
    render_recal_tab()

st.markdown("---")
st.caption("**Data Privacy Statement:** All data is open public data sourced from the City of Toronto Open Data Portal. © 2026 Neil Simmons. All rights reserved.")
