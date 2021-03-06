import os
import time
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import pandana as pdna
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.geometry import LineString
from scipy.spatial import distance
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs
import urbanaccess as ua
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
    zones = pd.concat([population.reset_index().to_crs(jobs.crs), jobs]).dissolve()
    resolution = 8
    hexagons = zones.h3.polyfill_resample(resolution).reset_index()
    hexagons = hexagons.rename(columns={'h3_polyfill': 'zone_id'})
    hexagons_with_jobs = gpd.sjoin(hexagons.drop(columns=['jobs']), jobs[['geometry', 'jobs']], how='left', predicate='intersects').drop(columns=['index_right'])
    hexagons_with_jobs = hexagons_with_jobs[~hexagons_with_jobs['jobs'].isnull()].copy()
    hexagons = hexagons[hexagons['zone_id'].isin(hexagons_with_jobs['zone_id'])][['zone_id', 'geometry']].to_crs(22192)
    cols = {'jobs': job_cols, 'population': population_cols}
    agents_per_hexagon = {}
    for agent in ['jobs', 'population']:
        gdf = eval(agent)
        gdf = gdf.to_crs(22192)
        split_gdf = gpd.overlay(gdf, hexagons, how='intersection')
        split_gdf = split_gdf[['ID', 'zone_id', 'geometry'] + cols[agent]]
        split_gdf['area'] = split_gdf.geometry.area
        zone_area = split_gdf.groupby('ID')['area'].sum().reset_index()
        zone_area = zone_area.rename(columns={'area': 'zone_area'})
        split_gdf = split_gdf.reset_index().merge(zone_area, on='ID', how='left')
        for col in cols[agent]:
            split_gdf[col] = split_gdf[col].fillna(0).astype('int') * split_gdf['area'] / split_gdf['zone_area']
        hexagon_agents = split_gdf.groupby('zone_id').sum()[cols[agent]]
        for col in cols[agent]:
            hexagon_agents[col] = round(hexagon_agents[col])
        agents_per_hexagon[agent] = hexagon_agents
    hexagons = hexagons.set_index('zone_id').join(agents_per_hexagon['jobs'])
    hexagons = hexagons.join(agents_per_hexagon['population']).reset_index()
    modified_routes = pd.read_csv('data/original/project_updates/modified_routes.csv').fillna(0)
    stops_locations = modified_routes.groupby('stop_id')['location'].min()
    buffer_cols = []
    for project_id in modified_routes.project_id.unique():
        project_stations = gpd.read_file('results/project_trajectories/project_%s_trajectory_nodes.shp' % project_id).to_crs(22192)
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
    cols = ['zone_id', 'geometry', 'jobs', 'lijobs'] + population_cols + buffer_cols
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
    tt_with_delta = pd.DataFrame()
    for project in travel_time_updates.project_id.unique():
        tt_proj = travel_time_updates[travel_time_updates['project_id']==project]
        for route in tt_proj.route_id.unique():
            tt_proj_route = tt_proj[tt_proj['route_id'] == route]
            for dir in tt_proj_route['direction_id'].unique():
                tt = tt_proj_route[tt_proj_route['direction_id'] == dir]
                tt['delta'] = tt['time_minutes'].cumsum()
                tt_with_delta = pd.concat([tt_with_delta, tt])
    travel_time_updates = tt_with_delta.copy()
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


def export_project_shape(stop_times, trips, project_id, stops, travel_time_updates):
    if not os.path.exists('results/project_trajectories'):
        os.makedirs('results/project_trajectories')
    edges = travel_time_updates[travel_time_updates['project_id']==project_id]
    edges = edges.rename(columns={'stop_id_from': 'from', 'stop_id_to': 'to'})
    nodes = stops.drop(columns=['stop_name']).rename(columns={'stop_id':'id', 'stop_lat': 'y', 'stop_lon': 'x'})
    nodes = nodes[(nodes['id'].isin(edges['from']))|(nodes['id'].isin(edges['to']))]
    export_shp(nodes, edges, name_shp='project_trajectories/project_%s_trajectory' % project_id)
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
        baseline_tt['time_minutes'] = (baseline_tt['arrival_time'] - baseline_tt['prev_arrival_time'])/60
        edges = baseline_tt[baseline_tt['stop_sequence'] > 1]
        edges = edges[edges['route_id'].isin(travel_time_updates[travel_time_updates['project_id'] == project_id]['route_id'].unique())]
        nodes = stops.drop(columns=['stop_name']).rename(columns={'stop_id': 'id', 'stop_lat': 'y', 'stop_lon': 'x'})
        nodes = nodes[(nodes['id'].isin(edges['from'])) | (nodes['id'].isin(edges['to']))]
        export_shp(nodes, edges, name_shp='project_trajectories/project_%s_baseline_trajectory' % project_id)


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
    stops = pd.concat([stops, new_stops])
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
            frequencies = pd.concat([frequencies, frequencies_route])
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
    stop_times = pd.concat([unchanged_stop_times, updated_stop_times])[list(stop_times.columns)]
    trips = pd.concat([trips[trips['trip_id'].isin(unchanged_trips)], updated_trips])
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
                modified_stop_times = pd.concat([modified_stop_times, stop_times_route])
                i += 1
    expanded_stop_times, unique_trips = expand_stop_times(frequencies, modified_stop_times.drop(columns=['route_id', 'service_id', 'direction_id']))
    expanded_trips = expanded_stop_times.groupby('trip_id').min()[['route_id', 'service_id', 'direction_id']].reset_index()
    return expanded_stop_times, expanded_trips


def read_process_zones(bbox):
    zones = gpd.read_file('data/processed/zones/zones.shp')
    zones['centroid'] = zones['geometry'].centroid
    zones = zones.set_geometry('centroid')
    zones = zones.rename(columns={'geometry': 'polygon_geometry', 'centroid': 'geometry'})
    zones = zones.set_geometry('geometry')
    zones = zones.to_crs(4326)
    zones['x'] = zones.geometry.x
    zones['y'] = zones.geometry.y
    if not os.path.isfile('results/osm_nodes.csv'):
        nodes, edges = ua.osm.load.ua_network_from_bbox(bbox=bbox, remove_lcn=True)
        nodes.to_csv('results/osm_nodes.csv', index=False)
        edges.to_csv('results/osm_edges.csv', index=False)
    else:
        nodes = pd.read_csv('results/osm_nodes.csv')
        edges = pd.read_csv('results/osm_edges.csv')
        nodes.index = nodes['id']
        edges['from_'] = edges['from']
        edges['to_'] = edges['to']
        edges = edges.set_index(['from_', 'to_'])
    net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["distance"]], twoway=False)
    zones['node_id'] = net.get_node_ids(zones['x'], zones['y'])
    zones = zones.rename(columns={'geometry': 'centroid', 'polygon_geometry': 'geometry'})
    zones = zones.set_geometry('geometry').drop(columns='centroid')
    zones = zones.to_crs(22192)
    zones['x_proj'] = zones.geometry.centroid.x
    zones['y_proj'] = zones.geometry.centroid.y
    zones = zones.set_index('node_id')
    return nodes, edges, zones

def create_ua_network(bbox, scenario, start_time, end_time, weekday):
    bbox = eval(bbox)
    nodes, edges, zones = read_process_zones(bbox)
    print('Creating UrbanAccess Network')
    gtfs_path = './data/processed/gtfs_%s' % scenario
    ua.osm.network.create_osm_net(osm_edges=edges, osm_nodes=nodes, travel_speed_mph=3)
    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=gtfs_path, bbox=bbox, validation=True,
                                               verbose=True, remove_stops_outsidebbox=True,
                                               append_definitions=True)
    loaded_feeds = preprocess_loaded_feeds(loaded_feeds)
    ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds, day=weekday,
                                       timerange=[start_time, end_time], calendar_dates_lookup=None,
                                       time_aware=True, simplify=True,
                                       use_existing_stop_times_int=True,
                                       )  #timerange_pad="02:00:00")
    loaded_feeds = ua.gtfs.headways.headways(loaded_feeds, [start_time, end_time])
    loaded_feeds.headways = loaded_feeds.headways.groupby('node_id_route').min().reset_index()
    loaded_feeds.headways.loc[loaded_feeds.headways['mean'].isnull(), 'mean'] = 60
    ua.network.integrate_network(urbanaccess_network=ua.network.ua_network, urbanaccess_gtfsfeeds_df=loaded_feeds, headways=True)
    ua_nodes = ua.ua_network.net_nodes[["id", "x", "y"]]
    ua_edges = ua.ua_network.net_edges[["from_int", "to_int", "weight"]]
    ua_nodes.to_csv('results/ua_nodes.csv')
    ua_edges.to_csv('results/ua_edges.csv', index=False)
    id_df = ua.ua_network.net_nodes.reset_index()[['id_int', 'id']]
    id_df = id_df.set_index('id')
    zones.index = zones.index.astype('str')
    zones = zones.join(id_df)
    zones.index.name = 'node_id'
    zones.to_file('results/zones.shp')


def preprocess_loaded_feeds(loaded_feeds):
    stop_times_df = loaded_feeds.stop_times
    stop_times_df['unique_trip_id'] = stop_times_df['trip_id'].str.cat(stop_times_df['unique_agency_id'].astype('str'), sep='_')
    stop_times_df['unique_stop_id'] = stop_times_df['stop_id'].str.cat(stop_times_df['unique_agency_id'].astype('str'), sep='_')
    stop_times_df['departure_time_sec_interpolate'] = stop_times_df['departure_time_sec']
    active_service_ids = []
    agency_id_col = 'unique_agency_id'
    agency_ids = stop_times_df[agency_id_col].unique()
    for agency in agency_ids:
        agency_calendar = loaded_feeds.calendar.loc[loaded_feeds.calendar[agency_id_col] == agency]
        agency_calendar_dates = loaded_feeds.calendar_dates.loc[loaded_feeds.calendar_dates[agency_id_col] == agency]
        agency_active_service_ids = ua.gtfs.network._calendar_service_id_selector(calendar_df=agency_calendar, calendar_dates_df=agency_calendar_dates, day=weekday)
        active_service_ids.extend(agency_active_service_ids)
    columns = ['route_id', 'direction_id', 'trip_id', 'service_id', 'unique_agency_id','unique_feed_id']
    if 'direction_id' not in loaded_feeds.trips.columns: columns.remove('direction_id')
    calendar_selected_trips_df = ua.gtfs.network._trip_selector(trips_df=loaded_feeds.trips[columns], service_ids=active_service_ids)
    calendar_selected_trips_df['unique_trip_id'] = calendar_selected_trips_df['trip_id'].str.cat(calendar_selected_trips_df['unique_agency_id'].astype('str'), sep='_')
    uniquetriplist = calendar_selected_trips_df['unique_trip_id'].unique().tolist()
    stop_times_df.sort_values(by=['unique_trip_id', 'stop_sequence'], inplace=True)
    stop_times_df = stop_times_df[stop_times_df['unique_trip_id'].isin(uniquetriplist)]
    stop_times_df = ua.gtfs.network._time_difference(stop_times_df=stop_times_df)
    loaded_feeds.stop_times_int = stop_times_df
    return loaded_feeds



def calculate_distance_matrix():
    print('Calculating euclidian distance matrix')
    df = gpd.read_file('results/zones.shp').set_index('id_int')
    id_col = 'zone_id'
    coords = [coords for coords in zip(df['y_proj'], df['x_proj'])]
    distances = distance.cdist(coords, coords, 'euclidean')
    distances = pd.DataFrame(distances, columns=df[id_col].unique(), index=df[id_col].unique())
    distances = distances.stack().reset_index().rename(columns={'level_0': 'from_id', 'level_1': 'to_id', 0: 'euclidean_distance'})
    df_to = df.reset_index().rename(columns={'zone_id': 'to_id', 'id_int': 'node_to'}).set_index('to_id')[['jobs', 'lijobs', 'node_to']]
    df_from = df.reset_index().rename(columns={'zone_id': 'from_id', 'id_int': 'node_from'}).set_index('from_id')[['node_from']]
    distances = distances.set_index('to_id').join(df_to).reset_index()
    distances = distances.set_index('from_id').join(df_from).reset_index()
    print('Distance matrix calculation done')
    distances.to_csv('results/distances.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-ug", "--update_gtfs", action="store_true", default=False, help="update_gtfs")
    parser.add_argument("-ud", "--update_demographics", action="store_true", default=False, help="update demographics")
    parser.add_argument("-bb", "--bounding_box", type=str, default=False, help="")
    parser.add_argument("-sc", "--scenario", type=str, default=False, help="")
    parser.add_argument("-st", "--start_time", type=str, default=False, help="start time for_analysis in 24 hr format (ej 07:00)")
    parser.add_argument("-et", "--end_time", type=str, default=False, help="end time for analysis in 24 hr format (ej 08:00)")
    parser.add_argument("-d", "--weekday", type=str, default=False, help="week day for analysis in 24 hr format (ej monday)")
    parser.add_argument("-ua", "--urbanaccess_net", action="store_true", default=False, help="create urbanaccess net")
    parser.add_argument("-ed", "--euclidean_dist", action="store_true", default=False, help="create euclidean distance matrix")
    args = parser.parse_args()


    update_gtfs = args.update_gtfs if args.update_gtfs else False
    update_demographics = args.update_demographics if args.update_demographics else False
    bounding_box = args.bounding_box if args.bounding_box else '(-59.3177426256, -35.3267410094, -57.6799695705, -34.1435770646)'
    scenario = args.scenario if args.scenario else '07:00:00'
    start_time = args.start_time if args.start_time else '07:00:00'
    end_time = args.end_time if args.end_time else '08:00:00'
    weekday = args.weekday if args.weekday else 'monday'
    urbanaccess_net = args.urbanaccess_net if args.urbanaccess_net else False
    euclidean_dist = args.euclidean_dist if args.euclidean_dist else False

    if update_gtfs:
        process_update_gtfs()

    if update_demographics:
        process_update_demographics()

    if urbanaccess_net:
        create_ua_network(bounding_box, scenario, start_time, end_time, weekday)

    if euclidean_dist:
        calculate_distance_matrix()
