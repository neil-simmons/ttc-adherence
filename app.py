import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import plotly.graph_objects as go
import gc
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
    initial_sidebar_state="expanded"
)

# All data lives in the Hugging Face dataset repo.
# The Spaces repo (neil-simmons/TrackTO) contains only app code.
HF_REPO      = "neil-simmons/ttc-avl-data"
HF_REPO_TYPE = "dataset"

PARQUET_HISTORY = "ttc_all_streetcars_history.parquet"
GTFS_STOPS      = "stops.txt"
GTFS_TRIPS      = "trips.txt"
GTFS_STOP_TIMES = "stop_times.txt"
GTFS_SHAPES     = "shapes.txt"

START_DATE    = '2026-03-15'
END_DATE      = '2026-05-02 23:59:59'
STAT_HOLIDAYS = ['2026-04-03']

MAX_TRACK_DEVIATION_M   = 150
MAX_ALLOWED_PING_GAP_SEC = 120
UTM_PROJ   = "EPSG:32617"
LATLON_PROJ = "EPSG:4326"

# ==============================================================================
# 1. SESSION STATE INITIALIZATION
# ==============================================================================
defaults = {
    'signatures_loaded':  False,
    'signature_list':     [],
    'selected_signature': None,
    'raw_pipeline_data':  None,
    'analysis_results':   None,
    'stop_filter_ids':    None,
    'force_t0_disabled':  False,
    'reliability_window': 'Standard (-15s to +2min)',
    'route_selection':    None,
    'direction_selection': None,
    'stage2_vars':        None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==============================================================================
# 2. DATA LOADERS
# ==============================================================================

def _hf(filename):
    """Download a file from the HF dataset repo and return its local cache path."""
    return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type=HF_REPO_TYPE)

# Each file gets its own cached download so nothing re-fetches within a session.
@st.cache_resource(show_spinner="Connecting to AVL data source...")
def get_parquet_path():
    return _hf(PARQUET_HISTORY)

@st.cache_resource(show_spinner="Downloading GTFS stops...")
def get_stops_path():
    return _hf(GTFS_STOPS)

@st.cache_resource(show_spinner="Downloading GTFS trips...")
def get_trips_path():
    return _hf(GTFS_TRIPS)

@st.cache_resource(show_spinner="Downloading GTFS stop times...")
def get_stop_times_path():
    return _hf(GTFS_STOP_TIMES)

@st.cache_resource(show_spinner="Downloading GTFS shapes...")
def get_shapes_path():
    return _hf(GTFS_SHAPES)

@st.cache_data(show_spinner="Indexing available routes...")
def get_available_routes(path):
    table = pq.read_table(path, columns=['route_id'])
    unique_arr = pc.unique(table.column('route_id')).to_pylist()
    cleaned = set()
    for r in unique_arr:
        if pd.isna(r):
            continue
        r_str = str(r).replace('.0', '').strip()
        if r_str and r_str != 'nan':
            cleaned.add(r_str)
    return sorted(list(cleaned))

@st.cache_data(max_entries=1, show_spinner="Extracting route data (streaming pushdown)...")
def load_route_data(path, selected_route):
    """
    Streams and filters the parquet at disk level before allocating RAM.
    Aggressively downcasts types to stay within the 1 GB Spaces limit.
    """
    schema = pq.read_schema(path)
    route_id_type = schema.field('route_id').type

    if pa.types.is_integer(route_id_type):
        filter_val = [int(selected_route)]
    elif pa.types.is_floating(route_id_type):
        filter_val = [float(selected_route)]
    else:
        filter_val = [str(selected_route), f"{selected_route}.0"]

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

    hour             = local_time.dt.hour.astype(np.int8)
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
    """Reads all four GTFS files from their cached HF download paths."""
    str_dtype = 'string[pyarrow]'

    stops = pd.read_csv(
        get_stops_path(),
        usecols=['stop_id', 'stop_name', 'stop_lat', 'stop_lon'],
        dtype=str_dtype
    )
    stops['stop_id'] = stops['stop_id'].astype('category')

    trips = pd.read_csv(
        get_trips_path(),
        usecols=['route_id', 'trip_id', 'shape_id', 'trip_headsign'],
        dtype={'route_id': str_dtype, 'trip_id': str_dtype, 'shape_id': str_dtype, 'trip_headsign': 'category'}
    )
    trips['trip_id'] = trips['trip_id'].str.replace(r'\.0$', '', regex=True).str.strip().astype('category')

    stop_times = pd.read_csv(
        get_stop_times_path(),
        usecols=['trip_id', 'stop_id', 'arrival_time', 'stop_sequence', 'shape_dist_traveled'],
        dtype={'trip_id': str_dtype, 'stop_id': str_dtype, 'arrival_time': str_dtype}
    )
    stop_times['trip_id']            = stop_times['trip_id'].str.replace(r'\.0$', '', regex=True).str.strip().astype('category')
    stop_times['stop_id']            = stop_times['stop_id'].astype('category')
    stop_times['shape_dist_traveled'] = pd.to_numeric(stop_times['shape_dist_traveled'], downcast='float')
    stop_times['stop_sequence']       = pd.to_numeric(stop_times['stop_sequence'], downcast='integer')

    shapes = pd.read_csv(
        get_shapes_path(),
        usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'],
        dtype={'shape_id': str_dtype}
    )
    shapes['shape_pt_lat']      = pd.to_numeric(shapes['shape_pt_lat'], downcast='float')
    shapes['shape_pt_lon']      = pd.to_numeric(shapes['shape_pt_lon'], downcast='float')
    shapes['shape_pt_sequence'] = pd.to_numeric(shapes['shape_pt_sequence'], downcast='integer')

    gc.collect()
    return stops, trips, stop_times, shapes

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def parse_gtfs_time(time_str):
    if pd.isna(time_str):
        return np.nan
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def parse_user_time(time_str, default_sec):
    try:
        h, m = map(int, time_str.split(':'))
        return h * 3600 + m * 60
    except Exception:
        return default_sec

def format_relative_time(seconds):
    if pd.isna(seconds):
        return "N/A"
    sign = "+" if seconds >= 0 else "-"
    secs = abs(int(seconds))
    return f"{sign}{secs // 60:02d}m {secs % 60:02d}s"

def format_seconds_to_time(seconds):
    if pd.isna(seconds) or seconds < 0:
        return "N/A"
    h          = int(seconds // 3600)
    m          = int((seconds % 3600) // 60)
    display_h  = h % 24
    ampm       = "AM" if display_h < 12 else "PM"
    display_h  = 12 if display_h in (0, 12) else display_h % 12
    return f"{display_h:02d}:{m:02d} {ampm}"

# ==============================================================================
# 4. VISUALIZATION GENERATOR
# ==============================================================================
def generate_visuals_and_map():
    if not st.session_state.raw_pipeline_data:
        return

    data                 = st.session_state.raw_pipeline_data
    st_filtered          = data['st_filtered']
    actual_relative_times = data['actual_relative_times']
    mode_b_lines         = data['mode_b_lines']

    is_standard    = st.session_state.reliability_window.startswith('Standard')
    window_early   = -15  if is_standard else -300
    window_late    = 120  if is_standard else  300

    reliability_dict, reliability_vals, sample_sizes = {}, {}, {}
    for stop in st_filtered.itertuples():
        arr = actual_relative_times[stop.stop_id]
        sample_sizes[stop.stop_id] = len(arr)
        if not arr:
            reliability_dict[stop.stop_id] = "N/A"
            reliability_vals[stop.stop_id] = 0.0
            continue
        sched_sec = stop.relative_sec
        hits = sum(1 for t in arr if window_early <= (t - sched_sec) <= window_late)
        pct  = (hits / len(arr)) * 100
        reliability_dict[stop.stop_id] = f"{pct:.1f}%"
        reliability_vals[stop.stop_id] = pct

    # --- Figure A: Density ---
    fig_A       = go.Figure()
    y_tick_texts = [
        f"{row['stop_name']} ({row['shape_dist_traveled']:.1f} km) [{reliability_dict[row['stop_id']]}]"
        for _, row in st_filtered.iterrows()
    ]

    diffs             = np.diff(st_filtered['shape_dist_traveled'])
    valid_diffs       = diffs[diffs > 0]
    min_gap           = max(0.1, np.min(valid_diffs) if len(valid_diffs) > 0 else 0.4)
    global_cloud_width = 1.0 * min_gap
    global_box_width   = 0.2 * min_gap
    global_box_shift   = 0.15 * min_gap

    for stop in st_filtered.itertuples():
        offsets_arr = actual_relative_times[stop.stop_id]
        N = len(offsets_arr)
        if N == 0:
            continue

        times_min  = [round(t / 60.0, 1) for t in offsets_arr]
        y_center   = stop.shape_dist_traveled
        box_center = y_center - global_box_shift

        if N < 10:
            c_base, c_fill, c_box, c_outlier = 'red', 'rgba(255,0,0,0.4)', 'rgba(255,0,0,0.1)', 'red'
        elif N < 25:
            c_base, c_fill, c_box, c_outlier = 'goldenrod', 'rgba(218,165,32,0.4)', 'rgba(218,165,32,0.1)', 'goldenrod'
        else:
            c_base, c_fill, c_box, c_outlier = 'blue', 'rgba(0,100,255,0.4)', 'rgba(0,100,255,0.1)', 'orange'

        q1, med, q3 = np.percentile(times_min, [25, 50, 75])
        hover = (
            f"<span style='font-size:12px'><b>{stop.stop_name}</b></span><br><br>"
            f"Sample Size: <b>N = {N}</b><br>"
            f"Median Arrival: <b>{med:+.1f} min</b><br>"
            f"Interquartile Range: {q1:+.1f} to {q3:+.1f} min<extra></extra>"
        )

        fig_A.add_trace(go.Violin(
            x=times_min, y=np.repeat(y_center, N),
            orientation='h', side='positive', width=global_cloud_width,
            scalemode='count', line_color=c_base, fillcolor=c_fill,
            showlegend=False, points=False, box_visible=False,
            hovertemplate=hover
        ))
        fig_A.add_trace(go.Box(
            x=times_min, y=np.repeat(box_center, N),
            orientation='h', width=global_box_width,
            line_color=c_base, fillcolor=c_box,
            boxpoints='outliers', marker=dict(color=c_outlier, size=4, opacity=0.8),
            showlegend=False, hoveron='points',
            hovertemplate="<b>Outlier</b><br>Actual Arrival: %{x:+.1f} min<extra></extra>"
        ))

    # --- Figure B: Spaghetti ---
    fig_B = go.Figure()
    for line_data in mode_b_lines:
        N  = len(line_data['x'])
        cd = np.empty((N, 6), dtype=object)
        cd[:, 0] = line_data['op_date']
        cd[:, 1] = line_data['start_time']
        cd[:, 2] = line_data['t_id']
        cd[:, 3] = line_data['abs_time']
        cd[:, 4] = line_data['lat']
        cd[:, 5] = line_data['lon']

        fig_B.add_trace(go.Scattergl(
            x=line_data['x'], y=line_data['y'],
            mode='lines+markers', line=dict(width=0.3), marker=dict(size=1.5),
            opacity=1.0, connectgaps=False,
            name=line_data['name'], text=line_data['speed'], customdata=cd,
            hovertemplate=(
                "<b>Absolute Time:</b> %{customdata[3]}<br>"
                "<b>Relative Arrival:</b> %{x:+.1f} min<br>"
                "<b>Track Distance:</b> %{y:.2f} km<br>"
                "<b>Speed:</b> %{text:.1f} km/h<br><br>"
                "<b>Date:</b> %{customdata[0]}<br>"
                "<b>Trip ID:</b> %{customdata[2]}<br>"
                "<b>GPS:</b> %{customdata[4]:.5f}, %{customdata[5]:.5f}<br>"
                "<extra></extra>"
            )
        ))

    # Shared scheduled baseline trace
    sched_trace = go.Scattergl(
        x=st_filtered['relative_sec'] / 60.0,
        y=st_filtered['shape_dist_traveled'],
        mode='lines+markers',
        line=dict(color='#000000', width=1.4),
        marker=dict(symbol='circle', size=4.5, color='#000000'),
        name="Scheduled Baseline",
        hovertemplate="<b>%{customdata}</b><br>Scheduled Profile: %{text}<extra></extra>",
        text=[format_relative_time(s) for s in st_filtered['relative_sec']],
        customdata=st_filtered['stop_name']
    )
    fig_A.add_trace(sched_trace)
    fig_B.add_trace(sched_trace)

    # Common layout
    common_layout = dict(
        yaxis_title="Official Track Distance (km) & Stops",
        template="plotly_white",
        autosize=True,
        yaxis=dict(
            range=[
                st_filtered['shape_dist_traveled'].min() - 0.02,
                st_filtered['shape_dist_traveled'].max() + 0.05
            ],
            tickmode='array',
            tickvals=st_filtered['shape_dist_traveled'],
            ticktext=y_tick_texts,
            gridcolor='lightgray', automargin=True, zeroline=False,
            tickfont=dict(size=10)
        ),
        xaxis=dict(
            zeroline=False, showline=True, showgrid=True,
            gridcolor='lightgray', dtick=5, ticksuffix=" min",
            automargin=True, tickfont=dict(size=11)
        ),
        margin=dict(l=10, r=20, t=60, b=10),
        legend=dict(font=dict(size=10), itemsizing='constant')
    )

    base_title = f"{data['title_info']} | {st.session_state.reliability_window}"
    fig_A.update_layout(**common_layout, title=dict(text=f"{base_title} — Density", font=dict(size=16)), violinmode='overlay', boxmode='overlay', hovermode="closest", showlegend=False)
    fig_B.update_layout(**common_layout, title=dict(text=f"{base_title} — Spaghetti", font=dict(size=16)), hovermode="closest")

    # --- Kepler map data ---
    stops_df = st_filtered[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].copy()
    stops_df['reliability']  = stops_df['stop_id'].map(reliability_vals)
    stops_df['sample_size']  = stops_df['stop_id'].map(sample_sizes)

    segments = []
    for i in range(len(st_filtered) - 1):
        s1, s2 = st_filtered.iloc[i], st_filtered.iloc[i + 1]
        if s1.stop_lon == s2.stop_lon and s1.stop_lat == s2.stop_lat:
            continue
        segments.append({
            'segment': f"{s1.stop_name} to {s2.stop_name}",
            'avg_reliability': (reliability_vals[s1.stop_id] + reliability_vals[s2.stop_id]) / 2.0,
            'geometry': LineString([
                (float(s1.stop_lon), float(s1.stop_lat)),
                (float(s2.stop_lon), float(s2.stop_lat))
            ])
        })
    segments_df = gpd.GeoDataFrame(segments, geometry='geometry', crs=LATLON_PROJ) if segments else gpd.GeoDataFrame()

    kepler_config = {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                        "type": "geojson",
                        "config": {
                            "dataId": "segments", "label": "Route Segments",
                            "colorField": {"name": "avg_reliability", "type": "real"},
                            "colorScale": "quantize",
                            "visConfig": {
                                "thickness": 5,
                                "colorRange": {"colors": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]}
                            }
                        }
                    },
                    {
                        "type": "point",
                        "config": {
                            "dataId": "stops", "label": "Stops",
                            "colorField": {"name": "reliability", "type": "real"},
                            "colorScale": "quantize",
                            "sizeField": {"name": "sample_size", "type": "integer"},
                            "visConfig": {
                                "radiusRange": [5, 20],
                                "colorRange": {"colors": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]}
                            }
                        }
                    }
                ]
            }
        }
    }

    st.session_state.analysis_results = {
        'fig_A': fig_A, 'fig_B': fig_B,
        'stops_df': stops_df, 'segments_df': segments_df,
        'kepler_config': kepler_config
    }

# ==============================================================================
# 5. UI — BOOT & SIDEBAR
# ==============================================================================
st.title("TTC Streetcar Schedule Adherence")
st.caption("Open-data analysis of TTC streetcar performance versus published GTFS schedules. Developed for the Transit Data Challenge 2026.")

parquet_path     = get_parquet_path()
available_routes = get_available_routes(parquet_path)
stops, trips, stop_times, shapes = load_gtfs()

with st.sidebar:
    st.header("1. Route Configuration")

    selected_route = st.selectbox("Route", available_routes, index=0)

    gtfs_route_trips = trips[trips['route_id'] == selected_route].copy()
    headsigns        = gtfs_route_trips['trip_headsign'].dropna().unique()

    if len(headsigns) == 0:
        st.error(f"No GTFS data for Route {selected_route}.")
        st.stop()

    selected_dir = st.selectbox("Direction (Headsign)", headsigns)

    # Reset downstream state when route or direction changes
    if (selected_route != st.session_state.route_selection or
            selected_dir != st.session_state.direction_selection):
        st.session_state.route_selection    = selected_route
        st.session_state.direction_selection = selected_dir
        st.session_state.signatures_loaded  = False
        st.session_state.analysis_results   = None

    gtfs_route_trips = gtfs_route_trips[gtfs_route_trips['trip_headsign'] == selected_dir]

    # Stop filter — uses a representative trip from GTFS to enumerate stops
    valid_st_sidebar = stop_times[stop_times['trip_id'].isin(gtfs_route_trips['trip_id'])].copy()
    if not valid_st_sidebar.empty:
        valid_st_sidebar = valid_st_sidebar.merge(stops, on='stop_id', how='left')
        sample_t         = valid_st_sidebar['trip_id'].iloc[0]
        sample_stops     = valid_st_sidebar[valid_st_sidebar['trip_id'] == sample_t].sort_values('stop_sequence')
        if sample_stops['shape_dist_traveled'].max() > 500:
            sample_stops = sample_stops.copy()
            sample_stops['shape_dist_traveled'] /= 1000.0

        stop_options      = {
            row.stop_id: f"{row.stop_name} ({row.shape_dist_traveled:.1f} km)"
            for _, row in sample_stops.iterrows()
        }
        selected_stop_ids = st.multiselect(
            "Stop Filter",
            options=list(stop_options.keys()),
            default=list(stop_options.keys()),
            format_func=lambda x: stop_options[x],
            help="Filter both tabs to show only selected stops."
        )
        st.session_state.stop_filter_ids = selected_stop_ids

        if not selected_stop_ids:
            st.session_state.force_t0_disabled = True
        else:
            first_stop_id = sample_stops.iloc[0]['stop_id']
            st.session_state.force_t0_disabled = first_stop_id not in selected_stop_ids

        if st.button("Reset Stop Filter", use_container_width=True):
            st.session_state.stop_filter_ids   = None
            st.session_state.force_t0_disabled = False
            st.rerun()

    st.divider()

    st.header("Quick Adjustments")
    window_choice = st.radio(
        "On-Time Reliability Window",
        ["Standard (-15s to +2min)", "Symmetric (-5min to +5min)"],
        help="Defines what counts as on time. Affects reliability % labels on both tabs."
    )
    if window_choice != st.session_state.reliability_window:
        st.session_state.reliability_window = window_choice
        generate_visuals_and_map()

# ==============================================================================
# 6. UI — TABS
# ==============================================================================
tab_analysis, tab_map = st.tabs(["📊 Schedule Adherence Analysis", "🗺️ Route Reliability Map"])

with tab_analysis:
    st.subheader("2. Analysis Configuration")

    with st.form("filter_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            day_type  = st.radio("Day Type", ["Weekdays", "Saturdays", "Sundays & Holidays"])
            time_mode = st.radio("Time Application Mode", ["Trip Start Mode", "Overlap Mode"])
        with col2:
            start_time_input = st.text_input("Start Time (HH:MM)", value="07:00")
            end_time_input   = st.text_input("End Time (HH:MM)", value="09:00")
            force_t0         = st.checkbox(
                "Force t=0 Start Alignment",
                value=False,
                disabled=st.session_state.force_t0_disabled,
                help="Disabled when the first stop of the route is excluded from the Stop Filter."
            )
        load_sig_btn = st.form_submit_button("Load Signatures")

    if load_sig_btn:
        with st.spinner("Extracting and indexing historical data for route..."):
            df_hist = load_route_data(parquet_path, selected_route)

            if df_hist.empty:
                st.error(f"No historical GPS data found for Route {selected_route}.")
                st.stop()

            if day_type == "Saturdays":
                day_mask = (df_hist['day_of_week'] == 5) & (~df_hist['is_holiday'])
            elif day_type == "Sundays & Holidays":
                day_mask = (df_hist['day_of_week'] == 6) | (df_hist['is_holiday'])
            else:
                day_mask = (df_hist['day_of_week'] <= 4) & (~df_hist['is_holiday'])
            df_hist = df_hist[day_mask]

            filter_start_sec = parse_user_time(start_time_input, 0)
            filter_end_sec   = parse_user_time(end_time_input, 86399)

            historical_trip_ids = df_hist['trip_id'].unique()
            valid_trips         = gtfs_route_trips[gtfs_route_trips['trip_id'].isin(historical_trip_ids)]

            if valid_trips.empty:
                st.error("No historical data matches GTFS schedule for this day type / direction.")
            else:
                valid_st = stop_times[stop_times['trip_id'].isin(valid_trips['trip_id'])].copy()
                valid_st = valid_st.merge(stops, on='stop_id', how='left')
                valid_st['arrival_sec']  = valid_st['arrival_time'].apply(parse_gtfs_time)
                start_times_series       = valid_st.groupby('trip_id')['arrival_sec'].transform('min')
                valid_st['relative_sec'] = valid_st['arrival_sec'] - start_times_series
                valid_st                 = valid_st.sort_values(['trip_id', 'stop_sequence'])

                signatures_dict = {}
                for t_id, df_group in valid_st.groupby('trip_id', observed=True):
                    sig = tuple(zip(df_group['stop_id'], df_group['relative_sec']))
                    if not sig:
                        continue
                    if sig not in signatures_dict:
                        signatures_dict[sig] = []
                    signatures_dict[sig].append(t_id)

                first_stops     = valid_st.groupby('trip_id', observed=True).first().reset_index()
                last_stops      = valid_st.groupby('trip_id', observed=True).last().reset_index()
                trip_start_dict = dict(zip(first_stops['trip_id'], first_stops['arrival_sec']))
                trip_orig_dict  = dict(zip(first_stops['trip_id'], first_stops['stop_name']))
                trip_dest_dict  = dict(zip(last_stops['trip_id'],  last_stops['stop_name']))

                trip_hist_counts = df_hist.groupby('trip_id', observed=True)['op_date'].nunique().to_dict()

                sig_ui_list = []
                for sig, t_ids in signatures_dict.items():
                    hist_run_count = sum(trip_hist_counts.get(tid, 0) for tid in t_ids)
                    if hist_run_count == 0:
                        continue
                    start_secs = [trip_start_dict[tid] for tid in t_ids]
                    min_s, max_s = min(start_secs), max(start_secs)
                    if max_s < filter_start_sec or min_s > filter_end_sec:
                        continue
                    sig_ui_list.append({
                        'signature': sig, 't_ids': t_ids,
                        'orig': trip_orig_dict[t_ids[0]], 'dest': trip_dest_dict[t_ids[0]],
                        'stops': len(sig), 'min_sec': min_s, 'max_sec': max_s, 'runs': hist_run_count
                    })

                sig_ui_list = sorted(sig_ui_list, key=lambda x: x['min_sec'])

                del valid_st
                gc.collect()

                if not sig_ui_list:
                    st.warning("No GTFS signatures scheduled to run within your time range.")
                    st.session_state.signatures_loaded = False
                else:
                    st.session_state.signature_list   = sig_ui_list
                    st.session_state.signatures_loaded = True
                    st.session_state.stage2_vars = {
                        'trip_start_dict':  trip_start_dict,
                        'filter_start_sec': filter_start_sec,
                        'filter_end_sec':   filter_end_sec,
                        'time_mode':        time_mode,
                        'force_t0':         force_t0,
                        'day_type':         day_type,
                        'time_range_str':   f"{start_time_input}-{end_time_input}"
                    }

    if st.session_state.signatures_loaded:
        with st.form("run_analysis_form"):
            st.subheader("3. Select Signature & Run")
            sig_options = {
                i: f"({s['runs']} runs) | {format_seconds_to_time(s['min_sec'])} – {format_seconds_to_time(s['max_sec'])} | {s['orig']} → {s['dest']}"
                for i, s in enumerate(st.session_state.signature_list)
            }
            selected_sig_idx = st.selectbox(
                "Select Signature Window",
                options=list(sig_options.keys()),
                format_func=lambda x: sig_options[x]
            )
            run_btn = st.form_submit_button("Run Mathematical Pipeline", type="primary")

        if run_btn:
            with st.spinner("Running monotonic sequential tracker and spatial interpolations..."):
                try:
                    s2_vars      = st.session_state.stage2_vars
                    selected_sig = st.session_state.signature_list[selected_sig_idx]
                    matching_trip_ids = selected_sig['t_ids']

                    df_hist_raw = load_route_data(parquet_path, selected_route)

                    if s2_vars['day_type'] == "Saturdays":
                        day_mask = (df_hist_raw['day_of_week'] == 5) & (~df_hist_raw['is_holiday'])
                    elif s2_vars['day_type'] == "Sundays & Holidays":
                        day_mask = (df_hist_raw['day_of_week'] == 6) | (df_hist_raw['is_holiday'])
                    else:
                        day_mask = (df_hist_raw['day_of_week'] <= 4) & (~df_hist_raw['is_holiday'])

                    df_hist_filtered = df_hist_raw[day_mask]
                    trip_hist        = df_hist_filtered[df_hist_filtered['trip_id'].isin(matching_trip_ids)].copy()

                    valid_st_stage2 = stop_times[stop_times['trip_id'].isin(matching_trip_ids)].copy()
                    valid_st_stage2 = valid_st_stage2.merge(stops, on='stop_id', how='left')
                    valid_st_stage2['arrival_sec']  = valid_st_stage2['arrival_time'].apply(parse_gtfs_time)
                    start_times_s2                  = valid_st_stage2.groupby('trip_id', observed=True)['arrival_sec'].transform('min')
                    valid_st_stage2['relative_sec'] = valid_st_stage2['arrival_sec'] - start_times_s2

                    sample_trip = matching_trip_ids[0]
                    st_filtered = valid_st_stage2[valid_st_stage2['trip_id'] == sample_trip].copy().sort_values('stop_sequence')

                    if st_filtered['shape_dist_traveled'].max() > 500:
                        st_filtered['shape_dist_traveled'] /= 1000.0

                    if st.session_state.stop_filter_ids:
                        st_filtered = st_filtered[st_filtered['stop_id'].isin(st.session_state.stop_filter_ids)]

                    if len(st_filtered) < 2:
                        st.error("Not enough stops selected in filter to track.")
                        st.stop()

                    sample_shape_id = gtfs_route_trips[gtfs_route_trips['trip_id'] == sample_trip]['shape_id'].iloc[0]
                    shp_pts         = shapes[shapes['shape_id'] == sample_shape_id].copy().sort_values('shape_pt_sequence')
                    line_coords     = list(zip(shp_pts['shape_pt_lon'].astype(float), shp_pts['shape_pt_lat'].astype(float)))

                    if len(line_coords) < 2:
                        st.error("GTFS geometry error: route track has fewer than 2 coordinate points.")
                        st.stop()

                    target_line_utm = gpd.GeoDataFrame(
                        index=[0], crs=LATLON_PROJ,
                        geometry=[LineString(line_coords)]
                    ).to_crs(UTM_PROJ).geometry.iloc[0]

                    trip_hist_gdf = gpd.GeoDataFrame(
                        trip_hist, crs=LATLON_PROJ,
                        geometry=gpd.points_from_xy(trip_hist.longitude, trip_hist.latitude)
                    ).to_crs(UTM_PROJ)

                    trip_hist['dist_to_track_m'] = trip_hist_gdf.distance(target_line_utm)
                    valid_mask = trip_hist['dist_to_track_m'] <= MAX_TRACK_DEVIATION_M
                    trip_hist  = trip_hist[valid_mask].copy()

                    if trip_hist.empty:
                        st.warning("No GPS pings matched the route track. Vehicles may have been detouring off-route.")
                        st.stop()

                    trip_hist_gdf          = trip_hist_gdf[valid_mask].copy()
                    trip_hist['official_dist_km'] = trip_hist_gdf.geometry.apply(
                        lambda pt: target_line_utm.project(pt)
                    ) / 1000.0

                    actual_relative_times = {stop_id: [] for stop_id in st_filtered['stop_id']}
                    mode_b_lines          = []

                    for (op_date, t_id), group in trip_hist.groupby(['op_date', 'trip_id'], observed=True):
                        group = group.sort_values('system_time').reset_index(drop=True)
                        if len(group) < 3:
                            continue

                        gtfs_start_sec = s2_vars['trip_start_dict'].get(t_id)
                        if gtfs_start_sec is None or group['official_dist_km'].isna().all():
                            continue

                        max_dist_idx          = group['official_dist_km'].idxmax()
                        group                 = group.loc[:max_dist_idx].copy()
                        group['official_dist_km'] = group['official_dist_km'].cummax()
                        group                 = group.drop_duplicates(subset=['official_dist_km'], keep='first')
                        if len(group) < 2:
                            continue

                        interpolated_times = np.interp(
                            st_filtered['shape_dist_traveled'].values,
                            group['official_dist_km'].values,
                            group['op_seconds'].values,
                            left=np.nan, right=np.nan
                        )
                        run_interpolations = {
                            sid: t for sid, t in zip(st_filtered['stop_id'], interpolated_times)
                            if not np.isnan(t)
                        }
                        if not run_interpolations:
                            continue

                        anchor_stop      = st_filtered.iloc[1] if len(st_filtered) > 1 else st_filtered.iloc[0]
                        anchor_stop_dist = anchor_stop['shape_dist_traveled']

                        if group['official_dist_km'].iloc[0] > anchor_stop_dist:
                            continue

                        if s2_vars['force_t0']:
                            anchor_stop_id = anchor_stop['stop_id']
                            if anchor_stop_id not in run_interpolations:
                                continue
                            idx_after = np.searchsorted(group['official_dist_km'].values, anchor_stop_dist)
                            if idx_after == 0 or idx_after >= len(group):
                                continue
                            time_gap = group['op_seconds'].iloc[idx_after] - group['op_seconds'].iloc[idx_after - 1]
                            if time_gap > MAX_ALLOWED_PING_GAP_SEC:
                                continue
                            anchor_sec = run_interpolations[anchor_stop_id] - anchor_stop['relative_sec']
                        else:
                            anchor_sec = gtfs_start_sec

                        f_start, f_end = s2_vars['filter_start_sec'], s2_vars['filter_end_sec']
                        if "Trip Start Mode" in s2_vars['time_mode']:
                            is_valid = f_start <= anchor_sec <= f_end
                        else:
                            is_valid = any(f_start <= t <= f_end for t in run_interpolations.values())

                        if is_valid:
                            for s_id, t in run_interpolations.items():
                                actual_relative_times[s_id].append(t - anchor_sec)

                            dist_diff = group['official_dist_km'].diff()
                            time_diff = group['system_time'].diff()
                            group['prev_speed_kmh'] = np.where(
                                time_diff > 0, (dist_diff / time_diff) * 3600, 0
                            ).clip(lower=0)
                            group['relative_min'] = (group['op_seconds'] - anchor_sec) / 60.0

                            abs_time_series = (
                                pd.to_datetime(group['system_time'], unit='s', utc=True)
                                .dt.tz_convert('America/Toronto')
                                .dt.strftime('%I:%M:%S %p')
                                .tolist()
                            )
                            mode_b_lines.append({
                                'name':       f"{op_date} | {t_id}",
                                'op_date':    str(op_date),
                                'start_time': format_seconds_to_time(list(run_interpolations.values())[0]),
                                't_id':       str(t_id),
                                'x':          group['relative_min'].tolist(),
                                'y':          group['official_dist_km'].tolist(),
                                'abs_time':   abs_time_series,
                                'lat':        group['latitude'].tolist(),
                                'lon':        group['longitude'].tolist(),
                                'speed':      group['prev_speed_kmh'].tolist()
                            })

                    if not mode_b_lines:
                        st.warning("No tracked trips matched the criteria.")
                        st.stop()

                    title_info = (
                        f"Route {selected_route} | {s2_vars['day_type']} "
                        f"{s2_vars['time_range_str']} | "
                        f"{'Force t=0' if s2_vars['force_t0'] else 'GTFS Aligned'}"
                    )
                    st.session_state.raw_pipeline_data = {
                        'st_filtered':          st_filtered,
                        'actual_relative_times': actual_relative_times,
                        'mode_b_lines':         mode_b_lines,
                        'shape_id':             sample_shape_id,
                        'title_info':           title_info
                    }
                    generate_visuals_and_map()
                    st.success("Analysis complete.")

                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    if st.session_state.analysis_results:
        st.plotly_chart(st.session_state.analysis_results['fig_A'], use_container_width=True)
        st.plotly_chart(st.session_state.analysis_results['fig_B'], use_container_width=True)

with tab_map:
    st.subheader("Route Spatial Reliability")
    if not st.session_state.analysis_results:
        st.info("👈 Configure and run the analysis in the Schedule Adherence tab first.")
    else:
        st.markdown(f"**Configuration:** {st.session_state.raw_pipeline_data['title_info']} | {st.session_state.reliability_window}")
        results = st.session_state.analysis_results
        if 'segments_df' in results and not results['segments_df'].empty:
            map_instance = KeplerGl(
                height=600,
                data={"stops": results['stops_df'], "segments": results['segments_df']},
                config=results['kepler_config']
            )
            keplergl_static(map_instance, center_map=True)
        else:
            st.warning("Spatial geometry could not be built for this route.")

st.sidebar.divider()
st.sidebar.caption(
    "**Data Privacy Statement:** All data is open public data sourced from the "
    "City of Toronto Open Data Portal. AVL data reflects vehicle GPS locations "
    "only — zero passenger or Personally Identifiable Information (PII)."
)
