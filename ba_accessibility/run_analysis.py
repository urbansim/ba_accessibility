import os
import gc
import time
import glob
import math
import shutil
import argparse
import numpy as np
import pandas as pd
import pandana as pdna
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import sys
import subprocess
import psutil
import h3pandas


def process_update_demographics():
    s_time = time.time()
    jobs = gpd.read_file('data/original/jobs/Empleo.shp')
    job_cols = ['jobs', 'job_a', 'job_b', 'job_c', 'job_d', 'job_h']
    population = gpd.read_file('data/original/population/base_AGBA_200913.shp')
    population['ID'] = population['PROV'] + population['DEPTO'] + population['FRAC'] + population['RADIO']
    population_cols = ['POB10', 'HOG10', 'NBI_H10']
    if not os.path.exists('data/processed/zones'):
        os.makedirs('./data/processed/zones')
    zones = population.reset_index().to_crs(jobs.crs).append(jobs).dissolve()
    resolution = 8
    hexagons = zones.h3.polyfill_resample(resolution).reset_index()
    hexagons_with_jobs = gpd.sjoin(hexagons.drop(columns=['jobs']), jobs[['geometry', 'jobs']], how='left', predicate='intersects').drop(columns=['index_right'])
    hexagons_with_jobs = hexagons_with_jobs[~hexagons_with_jobs['jobs'].isnull()].copy()
    hexagons = hexagons[hexagons['h3_polyfill'].isin(hexagons_with_jobs['h3_polyfill'])][['h3_polyfill', 'geometry']].to_crs(22192)
    cols = {'jobs': job_cols, 'population': population_cols}
    agents_per_hexagon = {}
    for agent in ['jobs', 'population']:
        gdf = eval(agent)
        gdf = gdf.to_crs(22192)
        split_gdf = gpd.overlay(gdf, hexagons, how='intersection')
        split_gdf = split_gdf[['ID', 'h3_polyfill', 'geometry'] + cols[agent]]
        split_gdf['area'] = split_gdf.geometry.area
        zone_area = split_gdf.groupby('ID')['area'].sum().reset_index()
        zone_area = zone_area.rename(columns={'area': 'zone_area'})
        split_gdf = split_gdf.reset_index().merge(zone_area, on='ID', how='left')
        for col in cols[agent]:
            split_gdf[col] = split_gdf[col].fillna(0).astype('int') * split_gdf['area'] / split_gdf['zone_area']
        hexagon_agents = split_gdf.groupby('h3_polyfill').sum()[cols[agent]]
        for col in cols[agent]:
            hexagon_agents[col] = round(hexagon_agents[col])
        agents_per_hexagon[agent] = hexagon_agents
    hexagons = hexagons.set_index('h3_polyfill').join(agents_per_hexagon['jobs'])
    hexagons = hexagons.join(agents_per_hexagon['population']).reset_index()
    modified_routes = pd.read_csv('data/original/project_updates/modified_routes.csv').fillna(0)
    stops_locations = modified_routes.groupby('stop_id')['location'].min()
    buffer_cols = []
    for project_id in modified_routes.project_id.unique():
        project_stations = gpd.read_file('results/project_%s_trajectory_nodes.shp' % project_id).to_crs(22192)
        project_stations = project_stations.set_index('id').join(stops_locations)
        project_stations.loc[project_stations['location'] == 'CABA', 'buffer'] = project_stations['geometry'].buffer(1500)
        project_stations.loc[project_stations['location'] == 'CORDON_1', 'buffer'] = project_stations['geometry'].buffer(1500)
        project_stations.loc[project_stations['location'] == 'CORDON_2', 'buffer'] = project_stations['geometry'].buffer(1500)
        project_stations = project_stations.drop(columns=['geometry']).rename(columns={'buffer':'geometry'})
        buffer_col = 'buff' + str(project_id)
        project_stations = project_stations.set_geometry('geometry').rename(columns={'location': buffer_col})
        project_stations = project_stations.dissolve()
        hexagons = gpd.sjoin(hexagons, project_stations[[buffer_col, 'geometry']], how='left', predicate='intersects').drop(columns=['index_right'])
        buffer_cols += [buffer_col]
    hexagons['lijobs'] = hexagons['job_a'] + hexagons['job_b'] + hexagons['job_c'] + hexagons['job_d'] + hexagons['job_h']
    cols = ['h3_polyfill', 'geometry', 'jobs', 'lijobs'] + population_cols + buffer_cols
    hexagons[cols].to_file('data/processed/zones/zones.shp')
    print('Took {:,.2f} seconds to process jobs shapefile'.format(time.time() - s_time))


def process_update_gtfs():
    print('Processing all GTFS feeds to required input format')
    copy_baseline_files()
    for dir in sorted(os.listdir('data/processed/gtfs_baseline')):
        print('---------------------------')
        print('Formatting', dir)
        s_time = time.time()
        frequencies = glob.glob('./data/processed/gtfs_baseline/%s/frequencies.txt' % (dir))
        if len(frequencies) > 0:
            frequencies, stop_times, trips, routes, agencies = read_files(dir)
            frequencies, stop_times = format_inputs(frequencies, stop_times)
            expanded_stop_times, unique_trips = expand_stop_times(frequencies, stop_times)
            expanded_trips = expand_trips(trips, unique_trips)
            export_outputs(dir, stop_times, trips, expanded_stop_times, expanded_trips)
        print('Took %s seconds to process %s' %(time.time() - s_time, dir))
    create_gtfs_with_project()


def copy_baseline_files():
    for mode in ['subte', 'trenes', 'colectivos']:
        original_path = ('data/original/gtfs_baseline/%s' % (mode))
        processed_path = ('data/processed/gtfs_baseline/%s' % (mode))
        agencies = pd.read_csv('%s/agency.txt' % original_path)
        if len(agencies.index) == 1:
            if not os.path.exists('data/processed/gtfs_baseline/%s' % (mode)):
                os.makedirs('./data/processed/gtfs_baseline/%s' % (mode))
            gtfs_files = ['agency', 'calendar', 'feed_info', 'routes', 'stop_times', 'stops', 'trips', 'frequencies']
            for file in gtfs_files:
                if os.path.isfile('data/original/gtfs_baseline/%s/%s.txt' % (mode, file)):
                    shutil.copy('./data/original/gtfs_baseline/%s/%s.txt' % (mode, file),
                                './data/processed/gtfs_baseline/%s/%s.txt' % (mode, file))
        else:
            feed_info = pd.read_csv('%s/feed_info.txt' % original_path)
            calendar = pd.read_csv('%s/calendar.txt' % original_path)
            agencies = pd.read_csv('%s/agency.txt' % original_path)
            routes = pd.read_csv('%s/routes.txt' % original_path)
            trips = pd.read_csv('%s/trips.txt' % original_path)
            stops = pd.read_csv('%s/stops.txt' % original_path)
            stop_times = pd.read_csv('%s/stop_times.txt' % original_path)
            frequencies = glob.glob('./%s/frequencies.txt' % original_path)
            frequencies = pd.read_csv('%s/frequencies.txt' % original_path) if len(frequencies) > 0 else None
            for agency in agencies.agency_id.unique():
                subset = {}
                subset['feed_info'] = feed_info.copy()
                subset['agency'] = agencies[agencies['agency_id'] == agency].copy()
                express_routes = [route for route in routes['route_desc'].unique() if'Expreso' in route]
                subset['routes'] = routes[(routes['agency_id'] == agency) & (~routes['route_desc'].isin(express_routes))].copy()
                subset['trips'] = trips[trips['route_id'].isin(subset['routes'].route_id)].copy()
                subset['calendar'] = calendar[calendar['service_id'].isin(subset['trips'].service_id)].copy()
                subset['stop_times'] = stop_times[stop_times['trip_id'].isin(subset['trips'].trip_id)].copy()
                subset['stops'] = stops[stops['stop_id'].isin(subset['stop_times'].stop_id)].copy()
                if len(subset['routes'].index) > 0:
                    if not os.path.exists('%s_%s' % (processed_path, agency)):
                        os.makedirs('./%s_%s' % (processed_path, agency))
                    if frequencies is not None:
                        subset['frequencies'] = frequencies[frequencies['trip_id'].isin(subset['trips'].trip_id)].copy()
                    for key in subset.keys():
                        subset[key].to_csv('./%s_%s/%s.txt' % (processed_path, agency, key), index=False)


def read_files(mode):
    print('Reading files')
    frequencies = pd.read_csv('data/processed/gtfs_baseline/%s/frequencies.txt' % mode)
    stop_times = pd.read_csv('data/processed/gtfs_baseline/%s/stop_times.txt' % mode)
    trips = pd.read_csv('data/processed/gtfs_baseline/%s/trips.txt' % mode)
    routes = pd.read_csv('data/processed/gtfs_baseline/%s/routes.txt' % mode)
    agencies = pd.read_csv('data/processed/gtfs_baseline/%s/agency.txt' % mode)
    return frequencies, stop_times, trips, routes, agencies


def format_inputs(frequencies, stop_times):
    print('Formatting inputs')
    frequencies = time_to_seconds(frequencies, ['start_time', 'end_time'])
    stop_times = time_to_seconds(stop_times, ['arrival_time', 'departure_time'])
    stop_times = stop_times.sort_values(by=['trip_id', 'stop_sequence'])
    if len(stop_times[stop_times['arrival_time'].isnull()].index) > 0:
        min_stop_sequence = stop_times.groupby('trip_id')['stop_sequence'].min().reset_index()
        min_stop_sequence = min_stop_sequence.rename(columns={'stop_sequence': 'min_stop_sequence'})
        stop_times = stop_times.set_index('trip_id').join(min_stop_sequence.set_index('trip_id')).reset_index()
        stop_times['min_arrival'] = stop_times['arrival_time'].ffill()
        stop_times.loc[~stop_times['arrival_time'].isnull(), 'min_arrival'] = stop_times['min_arrival'].shift()
        stop_times.loc[stop_times['stop_sequence'] == stop_times['min_stop_sequence'], 'min_arrival'] = stop_times['arrival_time']
        stop_times.loc[~stop_times['arrival_time'].isnull(), 'min_dist'] = stop_times['shape_dist_traveled']
        stop_times['min_dist'] = stop_times['min_dist'].ffill()
        stop_times.loc[~stop_times['arrival_time'].isnull(), 'min_dist'] = stop_times['min_dist'].shift()
        stop_times.loc[stop_times['stop_sequence'] == stop_times['min_stop_sequence'], 'min_dist'] = stop_times['shape_dist_traveled']
        stop_times['trip_arrival'] = stop_times['trip_id'] + '_' + stop_times['min_arrival'].astype('str')
        max_arrivals = stop_times.groupby('trip_arrival')['arrival_time', 'shape_dist_traveled'].max().reset_index()
        max_arrivals = max_arrivals.rename(columns={'arrival_time': 'max_arrival', 'shape_dist_traveled': 'max_dist'})
        stop_times = stop_times.set_index('trip_arrival').join(max_arrivals.set_index('trip_arrival'))
        stop_times['total_dist'] = stop_times['max_dist'] - stop_times['min_dist']
        stop_times['pct'] = (stop_times['shape_dist_traveled'] - stop_times['min_dist'])/stop_times['total_dist']
        stop_times['arrival_time'] = stop_times['min_arrival'] + stop_times['pct'] * (stop_times['max_arrival'] - stop_times['min_arrival'])
        stop_times['departure_time'] = stop_times['arrival_time']
        stop_times = stop_times.reset_index()
    stop_times['stop_duration'] = stop_times['departure_time'] - stop_times['arrival_time']
    return frequencies, stop_times


def time_to_seconds(df, cols):
    print('Converting times to seconds')
    for col in cols:
        times = df.loc[~df[col].isnull()][col].str.split(':')
        times = pd.DataFrame(times.tolist(), columns=['h', 'm', 's'], index=df.loc[~df[col].isnull()].index)
        times[col] = 3600 * times['h'].astype('int') + 60 * times['m'].astype('int') + times['s'].astype('int')
        df = df.drop(columns=[col]).join(times[[col]])
    return df


def expand_stop_times(frequencies, stop_times):
    print('Expanding stop times')
    start_times = pd.DataFrame(frequencies.groupby(['trip_id']).size(), columns=['number_start_times']).reset_index()
    frequencies = frequencies.merge(start_times, on='trip_id', how='left').sort_values(by='trip_id')
    frequencies.loc[~frequencies['trip_id'].eq(frequencies['trip_id'].shift()), 'start_time_idx'] = 1
    frequencies.loc[~frequencies['trip_id'].eq(frequencies['trip_id'].shift(-1)), 'start_time_idx'] = frequencies['number_start_times']
    frequencies['start_time_idx'] = frequencies['start_time_idx'].interpolate().astype('int')
    trip_ids = frequencies.trip_id.unique()
    len_batch = 25
    num_batches = max(round(len(trip_ids)/len_batch), 1)
    results = {}
    unique_trips = []
    lower_bound = 0
    prev_progress = 0
    for i in range(0, num_batches + 1):
        percentage_progress = 100 * i/num_batches
        if (percentage_progress - prev_progress) > 10:
            prev_progress = percentage_progress
            print('    ', round(percentage_progress), '% progress')
        upper_bound = min((i+1)*len_batch, len(trip_ids))
        if upper_bound > lower_bound:
            selected_trip_ids = trip_ids[lower_bound:upper_bound]
            selected_frequencies = frequencies[frequencies['trip_id'].isin(selected_trip_ids)]
            selected_stop_times = stop_times[stop_times['trip_id'].isin(selected_trip_ids)]

            min_arrival_time = selected_stop_times.groupby('trip_id').arrival_time.min().reset_index()
            min_arrival_time = min_arrival_time.rename(columns={'arrival_time': 'min_arrival_time'})
            selected_stop_times = selected_stop_times.merge(min_arrival_time, on='trip_id', how='left')

            df = selected_frequencies.merge(selected_stop_times, on='trip_id', how='left')
            df['trip_range'] = df['end_time'] - df['start_time']
            df['number_of_trips'] = (df['trip_range'] / df['headway_secs']).astype('int')
            df = df.reindex(np.repeat(df.index, df.number_of_trips)).reset_index(drop=True)
            df = df.sort_values(by=['trip_id', 'start_time', 'stop_sequence'])
            df['trip_start_sequence'] = df['trip_id'] + df['start_time'].astype('str') + df['stop_sequence'].astype('str')
            df.loc[~df['trip_start_sequence'].eq(df['trip_start_sequence'].shift()), 'trip_repetition'] = 0
            df.loc[~df['trip_start_sequence'].eq(df['trip_start_sequence'].shift(-1)), 'trip_repetition'] = df['number_of_trips'] - 1
            df['trip_repetition'] = df['trip_repetition'].interpolate().astype('int')
            df['trip_suffix'] = df['start_time_idx']*1000 + df['trip_repetition'] + 1
            df['trip_suffix'] = df['trip_suffix'].astype('int')
            df['arrival_time'] = df['start_time'] + df['headway_secs'] * df['trip_repetition'] + df['arrival_time'] - df['min_arrival_time'] #min_arrival_time
            df['departure_time'] = df['arrival_time'] + df['stop_duration']
            df['arrival_time'] = pd.to_datetime(df['arrival_time'].round(), unit='s').dt.time
            df['departure_time'] = pd.to_datetime(df['departure_time'].round(), unit='s').dt.time
            df['complete_trip_id'] = df['trip_id'] + '_' + df['trip_suffix'].astype('str')
            df = df.sort_values(by=['trip_id', 'trip_suffix'])
            df['trip_id'] = df['complete_trip_id']
            unique_trips += df['trip_id'].unique().tolist()
            results[i] = df
            lower_bound = upper_bound
    print('Concatenating Batches')
    stop_times = pd.concat(list(results.values()))
    print('Stop times expansion done')
    return stop_times, unique_trips


def expand_trips(trips, unique_trips):
    print('Expanding_trips')
    unique_trips = pd.DataFrame(unique_trips, columns=['complete_trip_id'])
    unique_trips[['trip_id', 'trip_suffix']] = unique_trips['complete_trip_id'].str.split('_', expand=True)
    expanded_trips = trips.set_index('trip_id').join(unique_trips.set_index('trip_id')[['trip_suffix']])
    expanded_trips = expanded_trips.reset_index()
    expanded_trips['trip_id'] = expanded_trips['trip_id'] + '_' + expanded_trips['trip_suffix'].astype('str')
    return expanded_trips


def export_outputs( mode, stop_times, trips, expanded_stop_times, expanded_trips):
    print('Exporting stop times')
    expanded_stop_times[stop_times.columns].to_csv('./data/processed/gtfs_baseline/%s/stop_times.txt' % (mode), index=False)
    print('Exporting trips')
    expanded_trips[trips.columns].to_csv('./data/processed/gtfs_baseline/%s/trips.txt' % (mode), index=False)
    print('Output export done')


def create_gtfs_with_project():
    new_stops = pd.read_csv('data/original/project_updates/new_stops.csv').fillna(0)
    stop_updates = pd.read_csv('data/original/project_updates/modified_routes.csv').fillna(0)
    travel_time_updates = pd.read_csv('data/original/project_updates/modified_times_between_stops.csv').fillna(0)
    for mode in travel_time_updates['mode'].unique():
        original_path = ('data/processed/gtfs_baseline/%s' % mode)
        trips = pd.read_csv('%s/trips.txt' % original_path, dtype={'trip_id': object})
        stops = pd.read_csv('%s/stops.txt' % original_path, dtype={'trip_id': object})
        stop_times = pd.read_csv('%s/stop_times.txt' % original_path, dtype={'trip_id': object})
        projects = travel_time_updates[travel_time_updates['mode'] == mode]['project_id'].unique()
        for project_id in projects:
            copy_with_project_files(project_id)
            updated_stops, updated_stop_times, updated_trips = update_frequencies(stops, stop_times, trips, new_stops, stop_updates, travel_time_updates, project_id)
            updated_stops.to_csv('data/processed/gtfs_project_%s/%s/stops.txt' % (project_id, mode),index=False)
            updated_stop_times.to_csv('data/processed/gtfs_project_%s/%s/stop_times.txt' % (project_id, mode), index=False)
            updated_trips.to_csv('data/processed/gtfs_project_%s/%s/trips.txt' % (project_id, mode), index=False)
            export_project_shape(stop_times, trips, project_id, updated_stops, travel_time_updates)

def export_project_shape(stop_times, trips, project_id, stops, travel_time_updates):
    edges = travel_time_updates[travel_time_updates['project_id']==project_id]
    edges = edges.rename(columns={'stop_id_from': 'from', 'stop_id_to': 'to'})
    nodes = stops.drop(columns=['stop_name']).rename(columns={'stop_id':'id', 'stop_lat': 'y', 'stop_lon': 'x'})
    nodes = nodes[(nodes['id'].isin(edges['from']))|(nodes['id'].isin(edges['to']))]
    export_shp(nodes, edges, name_shp='project_%s_trajectory' % project_id)
    affected_routes = [route for route in edges['route_id'].unique() if route in trips.route_id.unique()]
    if len(affected_routes)>0:
        stop_times = time_to_seconds(stop_times, ['arrival_time', 'departure_time'])
        stop_times_routes = stop_times.merge(trips[['trip_id', 'route_id', 'service_id', 'direction_id']], on='trip_id', how ='left')
        stop_times_routes['route_service_direction'] = stop_times_routes['route_id'].astype('str') + '_' + stop_times_routes['service_id'].astype('str') + '_' + stop_times_routes['direction_id'].astype('str')
        stop_times_routes['route_service_direction_trip'] = stop_times_routes['route_service_direction'] + '_' + stop_times_routes['trip_id'].astype('str')
        num_stops = stop_times_routes.groupby(['route_service_direction_trip'])['stop_id'].count()
        stop_times_routes = stop_times_routes.set_index('route_service_direction_trip')
        stop_times_routes['num_stops_trip'] = num_stops
        max_num_stops = stop_times_routes.reset_index().groupby(['route_service_direction'])['num_stops_trip'].max()
        stop_times_routes = stop_times_routes.reset_index().set_index('route_service_direction')
        stop_times_routes['max_num_stops'] = max_num_stops
        stop_times_routes = stop_times_routes[stop_times_routes['num_stops_trip'] == stop_times_routes['max_num_stops']]
        min_trips = stop_times_routes.groupby(['route_id', 'service_id', 'direction_id', 'stop_id'])['trip_id'].min()
        baseline_tt = stop_times_routes[stop_times_routes['trip_id'].isin(min_trips)]
        baseline_tt = baseline_tt[baseline_tt['service_id'] == baseline_tt['service_id'].min()]
        baseline_tt['to'] = baseline_tt['stop_id'].astype('int')
        baseline_tt['from'] = baseline_tt['stop_id'].shift().fillna(0).astype('int')
        baseline_tt['prev_arrival_time'] = baseline_tt['arrival_time'].shift().fillna(0)
        baseline_tt['min_since_prev'] = (baseline_tt['arrival_time'] - baseline_tt['prev_arrival_time'])/60
        edges = baseline_tt[baseline_tt['stop_sequence']>1]
        edges = edges[edges['route_id'].isin(travel_time_updates[travel_time_updates['project_id']==project_id]['route_id'].unique())]
        nodes = stops.drop(columns=['stop_name']).rename(columns={'stop_id': 'id', 'stop_lat': 'y', 'stop_lon': 'x'})
        nodes = nodes[(nodes['id'].isin(edges['from'])) | (nodes['id'].isin(edges['to']))]
        export_shp(nodes, edges, name_shp='project_%s_baseline_trajectory' % project_id)


def copy_with_project_files(project_id):
    for dir in sorted(os.listdir('data/processed/gtfs_baseline')):
        if not os.path.exists('data/processed/gtfs_project_%s/%s' % (project_id, dir)):
            os.makedirs('./data/processed/gtfs_project_%s/%s' % (project_id, dir))
        gtfs_files = ['agency', 'calendar', 'feed_info', 'routes', 'stop_times', 'stops', 'trips', 'frequencies']
        for file in gtfs_files:
            if os.path.isfile('data/processed/gtfs_baseline/%s/%s.txt' % (dir, file)):
                shutil.copy('./data/processed/gtfs_baseline/%s/%s.txt' % (dir, file),
                            './data/processed/gtfs_project_%s/%s/%s.txt' % (project_id, dir, file))


def update_frequencies(stops, stop_times, trips, new_stops, stop_updates, travel_time_updates, project_id):
    travel_time_updates = travel_time_updates[travel_time_updates['project_id']==project_id]
    stop_updates = stop_updates[stop_updates['project_id']==project_id]
    stops = stops.append(new_stops)
    stop_times = time_to_seconds(stop_times, ['arrival_time', 'departure_time'])
    stop_times_routes = stop_times.merge(trips[['trip_id', 'route_id', 'service_id', 'direction_id']], on='trip_id', how ='left')
    routes_to_update = travel_time_updates[travel_time_updates['project_id'] == project_id]['route_id'].unique()
    unchanged_trips = stop_times_routes[~stop_times_routes['route_id'].isin(routes_to_update)]['trip_id'].unique()
    if len(stop_times_routes[stop_times_routes['route_id'].isin(routes_to_update)].index) == 0:
        sample_route = stop_times_routes['route_id'].unique()[0]
        stop_times_routes_project = stop_times_routes[stop_times_routes['route_id'] == sample_route]
        start_times = pd.DataFrame(stop_times_routes_project.groupby(['route_id', 'service_id', 'direction_id'])['arrival_time'].min()).rename(columns={'arrival_time': 'start_time'})
        end_times = pd.DataFrame(stop_times_routes_project.groupby(['route_id', 'service_id', 'direction_id'])['arrival_time'].max()).rename(columns={'arrival_time': 'end_time'})
        frequencies = pd.DataFrame()
        for route in routes_to_update:
            start_times_route = start_times.reset_index().copy()
            end_times_route = end_times.reset_index().copy()
            start_times_route['route_id'] = route
            end_times_route['route_id'] = route
            start_times_route = start_times_route.set_index(['route_id', 'service_id', 'direction_id'])
            end_times_route = end_times_route.set_index(['route_id', 'service_id', 'direction_id'])
            frequencies_route = start_times_route.join(end_times_route)
            frequencies = frequencies.append(frequencies_route)
    else:
          stop_times_routes_project = stop_times_routes[stop_times_routes['route_id'].isin(routes_to_update)]
          start_times = pd.DataFrame(stop_times_routes_project.groupby(['route_id', 'service_id', 'direction_id'])['arrival_time'].min()).rename(columns={'arrival_time': 'start_time'})
          end_times = pd.DataFrame(stop_times_routes_project.groupby(['route_id', 'service_id', 'direction_id'])['arrival_time'].max()).rename(columns={'arrival_time': 'end_time'})
          frequencies = start_times.join(end_times)
    headways = travel_time_updates[['route_id', 'headway_min']].groupby('route_id').min().reset_index()
    headways['headway_secs'] = headways['headway_min'] * 60
    frequencies = frequencies.reset_index().merge(headways[['route_id', 'headway_secs']], on='route_id', how='left')
    frequencies['exact_times'] = 1
    updated_stop_times, updated_trips = update_travel_times(routes_to_update, frequencies, stop_times_routes, stop_updates, travel_time_updates)
    unchanged_stop_times = stop_times[stop_times['trip_id'].isin(unchanged_trips)]
    unchanged_stop_times['arrival_time'] = pd.to_datetime(unchanged_stop_times['arrival_time'].round(), unit='s').dt.time
    unchanged_stop_times['departure_time'] = pd.to_datetime(unchanged_stop_times['departure_time'].round(), unit='s').dt.time
    stop_times = unchanged_stop_times.append(updated_stop_times)[list(stop_times.columns)]
    trips = trips[trips['trip_id'].isin(unchanged_trips)].append(updated_trips)
    return stops, stop_times, trips


def update_travel_times(routes_to_update, frequencies, stop_times_routes, stop_updates, travel_time_updates):
    modified_stop_times = pd.DataFrame()
    start_trip_id = stop_times_routes['trip_id'].max() + str(1)
    for route in routes_to_update:
        i = 1
        for service_id in frequencies.service_id.unique():
            for direction in frequencies.direction_id.unique():
                selection = (stop_times_routes['route_id'] == route) & (stop_times_routes['service_id'] == service_id) & (stop_times_routes['direction_id'] == direction)
                if len(stop_times_routes[selection]) != 0:
                    start_trip_id = stop_times_routes[selection]['trip_id'].min()
                start_arrival_time = frequencies[frequencies['route_id']==route].start_time.min()
                stop_times_route = stop_updates[(stop_updates['route_id'] == route) & (stop_updates['direction_id']==direction)]
                stop_times_route['arrival_min'] = start_arrival_time
                stop_times_route['departure_min'] = start_arrival_time
                stop_times_route['prev_stop_id'] = stop_times_route['stop_id'].shift().fillna(0).astype('int')
                stop_times_route['from_to'] = stop_times_route['prev_stop_id'].astype('str') + '_'+ stop_times_route['stop_id'].astype('str')
                tt_updates_route = travel_time_updates[travel_time_updates['route_id']==route]
                stop_times_route = stop_times_route.merge(tt_updates_route[['from_to', 'delta']], on=['from_to'], how='left')
                stop_times_route['arrival_time'] = stop_times_route['arrival_min'] + stop_times_route['delta'].fillna(0) * 60
                stop_times_route['departure_time'] = stop_times_route['departure_min'] + stop_times_route['delta'].fillna(0) * 60
                stop_times_route['stop_duration'] = 0
                stop_times_route['service_id'] = service_id
                stop_times_route['trip_id'] = str(start_trip_id) + '_' + str(i)
                selection = (frequencies['route_id']==route) & (frequencies['service_id']==service_id) & (frequencies['direction_id']==direction)
                frequencies.loc[selection, 'trip_id'] = str(start_trip_id) + '_' + str(i)
                stop_times_route['shape_dist_traveled'] = 0
                modified_stop_times = modified_stop_times.append(stop_times_route)
                i += 1
    expanded_stop_times, unique_trips = expand_stop_times(frequencies, modified_stop_times.drop(columns=['route_id', 'service_id', 'direction_id']))
    expanded_trips = expanded_stop_times.groupby('trip_id').min()[['route_id', 'service_id', 'direction_id']].reset_index()
    return expanded_stop_times, expanded_trips


def run(project_ids, start_time, end_time, weekday):
    bbox = (-59.3177426256, -35.3267410094, -57.6799695705, -34.1435770646)
    project_scenarios = ['project_' + project_id for project_id in project_ids]
    for scenario in ['baseline'] + project_scenarios:
        print('------------------------------------')
        print('TOTAL MEMORY:', psutil.Process().memory_info().rss / (1024 * 1024))
        print('------------------------------------')
        ua_args = [sys.executable, 'utils.py', '-ua', '-bb', '(%s, %s, %s, %s)' % bbox, '-sc', scenario, '-st', start_time, '-et', end_time, '-d', weekday]
        subprocess.check_call(ua_args)
        net, zones_net, travel_data = create_pandana_network()
        calculate_indicators(scenario, zones_net, travel_data)
        del net, zones_net, travel_data
        gc.collect()
    results = pd.DataFrame()
    for scenario in project_scenarios:
        results = compare_indicators(scenario, results)
    summarize_results(results)


def create_pandana_network():
    print('Creating Pandana Network')
    s_time = time.time()
    ua_nodes = pd.read_csv('results/ua_nodes.csv', dtype={'x': float, 'y': float}).set_index('id_int')
    ua_edges = pd.read_csv('results/ua_edges.csv', dtype={'from_int':int, 'to_int': int, 'weight': float})
    net = pdna.Network(ua_nodes["x"], ua_nodes["y"], ua_edges["from_int"], ua_edges["to_int"],ua_edges[["weight"]], twoway=False)
    zones = gpd.read_file('results/zones.shp').set_index('node_id')
    net.set(zones['id_int'], variable=zones.jobs, name='jobs')
    net.set(zones['id_int'], variable=zones.lijobs, name='lijobs')
    zones = zones.set_index('id_int')
    zones.to_file('results/zones.shp')
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    ed_args = [sys.executable, 'utils.py', '-ed']
    subprocess.check_call(ed_args)
    travel_data = pd.read_csv('results/distances.csv')
    travel_data = calculate_pandana_distances(travel_data, net)
    return net, zones, travel_data


def calculate_pandana_distances(travel_data, net):
    print('Starting shortest path calculation')
    batch_length = 4000000
    n = math.ceil(len(travel_data.index)/batch_length)
    updated_travel_data = pd.DataFrame()
    for i in range(0, n + 1):
        print(i)
        if i*batch_length < len(travel_data.index):
            subset = travel_data.iloc[(i-1)*batch_length: i*batch_length].copy()
        else:
            subset = travel_data.iloc[(i-1)*batch_length:].copy()
        subset['pandana_distance'] = net.shortest_path_lengths(list(subset['node_from']), list(subset['node_to']))
        updated_travel_data = pd.concat([updated_travel_data, subset], axis=0)
    travel_data = updated_travel_data[updated_travel_data['pandana_distance'] <= 90]
    print('Pandana shortest paths done')
    return travel_data


def export_shp(nodes, edges, name_shp='test', df=None):
    nodes_from = nodes.rename(columns={'id': 'from', 'x': 'x_from', 'y': 'y_from'})
    nodes_to = nodes.rename(columns={'id': 'to', 'x': 'x_to', 'y': 'y_to'})
    edges = edges.merge(nodes_from, on='from', how='left')
    edges = edges.merge(nodes_to, on='to', how='left')
    edges['geometry'] = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in zip(edges['x_from'], edges['y_from'], edges['x_to'], edges['y_to'])]
    edges_gdf = gpd.GeoDataFrame(edges, crs={'init': 'epsg:4326'}, geometry='geometry')
    nodes['geometry'] = [Point(xy) for xy in zip(nodes['x'], nodes['y'])]
    nodes_gdf = gpd.GeoDataFrame(nodes, crs={'init': 'epsg:4326'}, geometry='geometry')
    nodes_gdf.to_file('results/%s_nodes.shp' % name_shp)
    edges_gdf.to_file('results/%s_edges.shp' % name_shp)
    if df is not None:
        df['geometry'] = [Point(xy) for xy in zip(df['x'], df['y'])]
        zones_gdf = GeoDataFrame(df, geometry='geometry', crs={'init': 'epsg:4326'})
        zones_gdf.to_file(name_shp + '_zones.shp')


def calculate_indicators(scenario, zones, travel_data):
    s_time = time.time()
    print('Calculating indicators from skims')
    within_60_min = travel_data[travel_data['pandana_distance']<=60]
    jobs_60 = within_60_min.groupby('from_id')['jobs', 'lijobs'].sum().rename(columns={'jobs':'jobs_60', 'lijobs':'lijobs_60'})
    within_90_min = travel_data[travel_data['pandana_distance']<=90]
    jobs_90 = within_90_min.groupby('from_id')['jobs', 'lijobs'].sum().rename(columns={'jobs':'jobs_90', 'lijobs':'lijobs_90'})
    zones = zones.set_index('h3_polyfil').join(jobs_60)
    zones = zones.join(jobs_90)
    times_to_cbd = travel_data[travel_data['to_id'] == '88c2e31ad1fffff'].rename(columns={'pandana_distance': 'time_cbd'})
    zones = zones.join(times_to_cbd.set_index('from_id')[['time_cbd']])
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    if not os.path.exists('results'):
        os.makedirs('./results')
    zones[['jobs', 'lijobs', 'jobs_60', 'lijobs_60', 'jobs_90', 'lijobs_90', 'time_cbd']].to_csv('results/%s.csv' % scenario)


def compare_indicators(scenario, results):
    print('Comparing scenario %s with Baseline' % scenario)
    zones = gpd.read_file('results/zones.shp')
    buffer_col = 'buff' + scenario.replace('project_', '')
    baseline = pd.read_csv('results/baseline.csv').set_index('h3_polyfil')
    project = pd.read_csv('results/%s.csv' % scenario).set_index('h3_polyfil')
    job_cols = [col for col in baseline.columns if 'jobs' in col] + ['time_cbd']
    project = project[job_cols]
    for col in job_cols:
        project = project.rename(columns={col: col+'p'})
    comparison = baseline[job_cols].join(project)
    comparison = zones.set_index('h3_polyfil')[['geometry', 'POB10', 'HOG10', 'NBI_H10', buffer_col]].join(comparison)
    project_id = scenario.replace('project_', '')
    nodes = gpd.read_file('results/%s_trajectory_nodes.shp' % scenario).to_crs(comparison.crs)
    edges = gpd.read_file('results/%s_trajectory_edges.shp' % scenario).to_crs(comparison.crs)
    comparison['project_id'] = project_id
    for jt in ['', 'li']:
        for tr in ['60', '90']:
            comparison['%sjobs_%sd' % (jt, tr)] = comparison['%sjobs_%sp' % (jt, tr)] - comparison['%sjobs_%s' % (jt, tr)]
            comparison['%sacc_%s' % (jt, tr)] = 100 * (comparison['%sjobs_%s' % (jt, tr)] / comparison['%sjobs' % jt].sum())
            comparison['%sacc_%sp' % (jt, tr)] = 100 * (comparison['%sjobs_%sp' % (jt, tr)] / comparison['%sjobsp' % jt].sum())
            comparison['%sacc_%sd' % (jt, tr)] = comparison['%sacc_%sp' % (jt, tr)] - comparison['%sacc_%s' % (jt, tr)]
            comparison['%spct_ch_acc%s' % (jt, tr)] = 100 * (comparison['%sacc_%sd' % (jt, tr)]) / comparison['%sacc_%s' % (jt, tr)]
            comparison['%spop_acc%s' % (jt, tr)] = comparison['%sacc_%s' % (jt, tr)] * comparison['POB10']
            comparison['%spov_acc%s' % (jt, tr)] = comparison['%sacc_%s' % (jt, tr)] * comparison['NBI_H10']
            comparison['%spop_acc%sp' % (jt, tr)] = comparison['%sacc_%sp' % (jt, tr)] * comparison['POB10']
            comparison['%spov_acc%sp' % (jt, tr)] = comparison['%sacc_%sp' % (jt, tr)] * comparison['NBI_H10']
            if jt == '':
                fig, ax = plt.subplots(1, figsize=(8, 8))
                comparison.plot(column='%sacc_%sd' % (jt, tr), ax=ax, cmap = 'Spectral', legend=True) #cmap= 'RdYlBu') #, linewidth=1, ax=ax, edgecolor='0.9', legend=True)
                edges.plot(ax=ax, color='black', linewidth=0.5)
                nodes.plot(ax=ax, color='black', markersize=0.5)
                ax.axis('off')
                plt.savefig("results/accessibility_change_%s_min.png" % tr)
    results = results.append(comparison)
    return results


def summarize_results(results):
    results_analysis = pd.DataFrame()
    analysis = 'all'
    for project_id in results.project_id.unique():
        for buff in [1000, 'gov']:
            for threshold in [60, 90]:
                if buff == 'gov':
                    buffer = gpd.read_file('data/original/gov_buffers/Area de Influencia Belgrano Sur.shp')
                    buffer = buffer.to_crs(results.crs).rename(columns={'descripcio': 'stop_name_from'})
                    remove_stations = [station for station in buffer.stop_name_from.str.lower().unique() if 'constitu' in station or 'aires' in station]
                    project_stations = buffer[~buffer['stop_name_from'].str.lower().isin(remove_stations)]
                else:
                    modified_routes = pd.read_csv('data/original/project_updates/modified_routes.csv').fillna(0)
                    modified_routes = modified_routes[modified_routes['project_id'].astype('str') == project_id]
                    stops_locations = modified_routes.groupby('stop_id')['stop_name_from'].min()
                    project_stations = gpd.read_file('results/project_%s_trajectory_nodes.shp' % project_id)
                    project_stations = project_stations.set_index('id').join(stops_locations)
                    project_stations = project_stations.to_crs(results.crs)
                    project_stations['buffer'] = project_stations['geometry'].buffer(buff)
                    project_stations = project_stations.drop(columns=['geometry']).rename(columns={'buffer': 'geometry'})
                    project_stations = project_stations.set_geometry('geometry')
                    remove_stations = [station for station in project_stations.stop_name_from.str.lower().unique() if 'constitu' in station or 'aires' in station]
                    project_stations = project_stations[~project_stations['stop_name_from'].str.lower().isin(remove_stations)]
                if analysis == 'all':
                    project_stations = project_stations.dissolve()
                    project_stations['stop_name_from'] = 'all'
                results_project = results[results['project_id'] == project_id].copy()
                results_stations = gpd.sjoin(results_project, project_stations[['stop_name_from', 'geometry']], how='left', predicate='intersects').drop(columns=['index_right'])
                results_stations['pop_acc'] = results_stations['acc_%s' % str(threshold)] * results_stations['POB10']
                results_stations['pov_acc'] = results_stations['acc_%s' % str(threshold)] * results_stations['NBI_H10']
                results_stations['pop_accp'] = results_stations['acc_%sp' % str(threshold)] * results_stations['POB10']
                results_stations['pov_accp'] = results_stations['acc_%sp' % str(threshold)] * results_stations['NBI_H10']
                jobs_baseline = 'jobs_%s' % str(threshold)
                jobs_project = 'jobs_%sp' % str(threshold)
                results_stations['pop_jobs'] = results_stations[jobs_baseline] * results_stations['POB10']
                results_stations['pov_jobs'] = results_stations[jobs_baseline] * results_stations['NBI_H10']
                results_stations['pop_jobsp'] = results_stations[jobs_project] * results_stations['POB10']
                results_stations['pov_jobsp'] = results_stations[jobs_project] * results_stations['NBI_H10']
                results_stations = results_stations.groupby('stop_name_from')['POB10', 'NBI_H10', 'pop_acc', 'pop_accp', 'pov_acc', 'pov_accp', 'pop_jobs', 'pop_jobsp', 'pov_jobs', 'pov_jobsp'].sum()
                results_stations['pop_acc'] = results_stations['pop_acc'] / results_stations['POB10']
                results_stations['pop_accp'] = results_stations['pop_accp'] / results_stations['POB10']
                results_stations['pop_accd'] = results_stations['pop_accp'] - results_stations['pop_acc']
                results_stations['pov_acc'] = results_stations['pov_acc'] / results_stations['NBI_H10']
                results_stations['pov_accp'] = results_stations['pov_accp'] / results_stations['NBI_H10']
                results_stations['pov_accd'] = results_stations['pov_accp'] - results_stations['pov_acc']
                results_stations['pop_jobs'] = results_stations['pop_jobs'] / results_stations['POB10']
                results_stations['pov_jobs'] = results_stations['pov_jobs'] / results_stations['NBI_H10']
                results_stations['pop_jobsp'] = results_stations['pop_jobsp'] / results_stations['POB10']
                results_stations['pov_jobsp'] = results_stations['pov_jobsp'] / results_stations['NBI_H10']
                results_stations['pop_pct_ch'] = 100 * (results_stations['pop_jobsp'] - results_stations['pop_jobs']) / results_stations['pop_jobs']
                results_stations['pov_pct_ch'] = 100 * (results_stations['pov_jobsp'] - results_stations['pov_jobs']) / results_stations['pov_jobs']
                results_stations = results_stations[~results_stations['pop_accd'].isnull()].drop(columns=['POB10', 'NBI_H10']).sort_values(by='pop_accd', ascending=False)
                results_stations['project'] = project_id
                results_stations['threshold'] = str(threshold)
                results_stations['buffer'] = str(buff)
                if len(results_analysis.index) == 0:
                    results_analysis = results_stations
                else:
                    results_analysis = results_analysis.append(results_stations)

    results_analysis = results_analysis.sort_values(by=['buffer', 'threshold', 'project'])
    names = {'pop_acc': 'Baseline Population Weighted Accessibility',
             'pop_accp': 'With Project Population Weighted Accessibility',
             'pop_accd': 'Change in Population Weighted Accessibility',
             'pov_acc': 'Baseline Poverty Weighted Accessibility',
             'pov_accp': 'With Project Poverty Weighted Accessibility',
             'pov_accd': 'Change in Poverty Weighted Accessibility',
             'pop_jobs': 'Baseline Population Weighted Number of Jobs Accessible',
             'pop_jobsp': 'With Project Population Weighted Number of Jobs Accessible',
             'pop_pct_ch': 'Percentage Change in Population Weighted Number of Jobs Accessible',
             'pov_jobs': 'Baseline Poverty Weighted Number of Jobs Accessible',
             'pov_jobsp': 'With Project Poverty Weighted Number of Jobs Accessible',
             'pov_pct_ch': 'Percentage Change in Poverty Weighted Number of Jobs Accessible',}
    results_analysis = results_analysis.reset_index().set_index(['stop_name_from', 'buffer', 'threshold', 'project'])
    results_analysis = results_analysis[list(names.keys())]
    results_analysis = results_analysis.rename(columns=names)
    results_analysis.to_csv('results/results_all_projects.csv')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ud", "--update_demographics", action="store_true", default=False, help="update demographics")
    parser.add_argument("-ug", "--update_gtfs", action="store_true", default=False, help="update_gtfs")
    parser.add_argument("-p", "--project_ids", type=str, nargs='+', default=[])
    parser.add_argument("-st", "--start_time", type=str, default=False, help="start time for_analysis in 24 hr format (ej 07:00)")
    parser.add_argument("-et", "--end_time", type=str, default=False, help="end time for analysis in 24 hr format (ej 08:00)")
    parser.add_argument("-d", "--weekday", type=str, default=False, help="week day for analysis in 24 hr format (ej monday)")
    args = parser.parse_args()

    update_demographics = args.update_demographics if args.update_demographics else False
    update_gtfs = args.update_gtfs if args.update_gtfs else False
    project_ids = args.project_ids if args.project_ids else ['1']
    start_time = args.start_time if args.start_time else '07:00:00'
    end_time = args.end_time if args.end_time else '08:00:00'
    weekday = args.weekday if args.weekday else 'monday'

    if update_gtfs:
        process_update_gtfs()
    if update_demographics:
        process_update_demographics()
    run(project_ids, start_time, end_time, weekday)