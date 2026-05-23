import time
import requests
import math
import csv
import os
from datetime import datetime
from google.transit import gtfs_realtime_pb2
from google.protobuf import text_format

# --- CONFIGURATION ---
FEED_URL = "https://gtfsrt.ttc.ca/vehicles/position"
POLL_INTERVAL = 15
CSV_FILENAME = "ttc_all_streetcars_history.csv"

STREETCAR_ROUTES = {"501", "503", "504", "505", "506", "508", "509", "510", "511", "512"}

vehicle_states = {}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_vehicle_positions():
    feed = gtfs_realtime_pb2.FeedMessage()
    
    # We ask the server nicely for the raw binary data
    headers = {
        'User-Agent': 'TTC-Streetcar-Project by Neil Simmons neil.simmons.mail@gmail.com',
        'Accept': 'application/x-protobuf, application/octet-stream'
    }
    
    try:
        response = requests.get(FEED_URL, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Safety Net: If the TTC still returns text, we parse it as text. Otherwise, parse as binary.
        content = response.content
        if content.startswith(b'header {') or content.startswith(b'{\n'):
            text_format.Parse(content.decode('utf-8'), feed)
        else:
            feed.ParseFromString(content)
            
        return feed.entity
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection error: {e}")
        return[]

def main():
    print("=====================================================")
    print(" TTC Live Streetcar Harvester Started")
    print(f" Polling every {POLL_INTERVAL} seconds.")
    print(f" Saving data to: {CSV_FILENAME}")
    print("=====================================================")

    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'system_time', 'ttc_timestamp', 'readable_time', 
                'route_id', 'trip_id', 'vehicle_id', 
                'latitude', 'longitude', 
                'dist_since_last_ping_km', 'cumulative_trip_dist_km'
            ])
        
        while True:
            loop_start_time = time.time()
            entities = get_vehicle_positions()
            updates_this_cycle = 0
            
            for entity in entities:
                if not entity.HasField('vehicle'):
                    continue
                
                route_id = entity.vehicle.trip.route_id
                
                if route_id not in STREETCAR_ROUTES:
                    continue
                
                vehicle_id = entity.vehicle.vehicle.id
                trip_id = entity.vehicle.trip.trip_id
                current_lat = entity.vehicle.position.latitude
                current_lon = entity.vehicle.position.longitude
                ttc_timestamp = entity.vehicle.timestamp
                
                if vehicle_id in vehicle_states:
                    state = vehicle_states[vehicle_id]
                    
                    if ttc_timestamp <= state['last_timestamp']:
                        continue
                        
                    if trip_id != state['trip_id']:
                        state['cumulative_dist'] = 0.0
                        state['trip_id'] = trip_id
                        
                    dist_moved = haversine(state['last_lat'], state['last_lon'], current_lat, current_lon)
                    state['cumulative_dist'] += dist_moved
                    
                    state['last_lat'] = current_lat
                    state['last_lon'] = current_lon
                    state['last_timestamp'] = ttc_timestamp
                    
                else:
                    dist_moved = 0.0
                    vehicle_states[vehicle_id] = {
                        'last_lat': current_lat,
                        'last_lon': current_lon,
                        'last_timestamp': ttc_timestamp,
                        'cumulative_dist': 0.0,
                        'trip_id': trip_id
                    }
                    state = vehicle_states[vehicle_id]

                readable_time = datetime.fromtimestamp(ttc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                writer.writerow([
                    int(time.time()), ttc_timestamp, readable_time,
                    route_id, trip_id, vehicle_id,
                    current_lat, current_lon,
                    round(dist_moved, 5), round(state['cumulative_dist'], 5)
                ])
                updates_this_cycle += 1

            f.flush()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {updates_this_cycle} new streetcar updates. Sleeping...")
            
            time_to_sleep = POLL_INTERVAL - (time.time() - loop_start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

if __name__ == "__main__":
    main()
