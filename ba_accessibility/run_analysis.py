import os
import time
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import pandana as pdna
import geopandas as gpd
import urbanaccess as ua
from geopandas import GeoDataFrame
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.geometry import LineString
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs
import h3pandas

def process_update_demographics(divide_zones = True):
    s_time = time.time()
    jobs = gpd.read_file('data/original/jobs/Empleo.shp')
    job_cols = ['jobs', 'job_a', 'job_b', 'job_c', 'job_d', 'job_h']
    population = gpd.read_file('data/original/population/base_AGBA_200913.shp')
    population['ID'] = population['PROV'] + population['DEPTO'] + population['FRAC'] + population['RADIO']
    population_cols = ['POB10', 'HOG10', 'NBI_H10']
    if not os.path.exists('data/processed/zones'):
        os.makedirs('./data/processed/zones')
    zones = population.reset_index().to_crs(jobs.crs).append(jobs)
    if divide_zones == True:
        resolution = 8  # 10
        hexagons = zones.h3.polyfill_resample(resolution).reset_index()
        hexagons = gpd.sjoin(hexagons.drop(columns=['jobs']), jobs[['geometry', 'jobs']], how='left', predicate='intersects').drop(columns=['index_right'])
        hexagons = hexagons[~hexagons['jobs'].isnull()].copy()
        hexagons = hexagons[['h3_polyfill', 'geometry']].to_crs(22192)
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
            project_stations.loc[project_stations['location'] == 'CORDON_1', 'buffer'] = project_stations['geometry'].buffer(2000)
            project_stations.loc[project_stations['location'] == 'CORDON_2', 'buffer'] = project_stations['geometry'].buffer(2500)
            project_stations = project_stations.drop(columns=['geometry']).rename(columns={'buffer':'geometry'})
            buffer_col = 'buff' + str(project_id)
            project_stations = project_stations.set_geometry('geometry').rename(columns={'location': buffer_col})
            project_stations = project_stations.dissolve()
            hexagons = gpd.sjoin(hexagons, project_stations[[buffer_col, 'geometry']], how='left', predicate='intersects').drop(columns=['index_right'])
            buffer_cols += [buffer_col]
        hexagons['lijobs'] = hexagons['job_a'] + hexagons['job_b'] + hexagons['job_c'] + hexagons['job_d'] + hexagons['job_h']
        cols = ['h3_polyfill', 'geometry', 'jobs', 'lijobs'] + population_cols + buffer_cols
        hexagons[cols].to_file('data/processed/zones/zones.shp')
    else:
        jobs[['ID', 'geometry'] + job_cols + population_cols].to_file('data/processed/zones/zones.shp')
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
                subset['routes'] = routes[routes['agency_id'] == agency].copy()
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
        stop_times['arrival_time'] = stop_times['arrival_time'].interpolate()
        stop_times['departure_time'] = stop_times['departure_time'].interpolate()
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

            #min_arrival_time = stop_times.arrival_time.min()
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
            export_project_shape(project_id, updated_stops, travel_time_updates)

def export_project_shape(project_id, stops, travel_time_updates):
    edges = travel_time_updates[travel_time_updates['project_id']==project_id]
    edges = edges.rename(columns={'stop_id_from': 'from', 'stop_id_to': 'to'})
    nodes = stops.drop(columns=['stop_name']).rename(columns={'stop_id':'id', 'stop_lat': 'y', 'stop_lon': 'x'})
    nodes = nodes[(nodes['id'].isin(edges['from']))|(nodes['id'].isin(edges['to']))]
    export_shp(nodes, edges, name_shp='project_%s_trajectory' % project_id)


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
    stops = stops.append(new_stops)
    stop_times = time_to_seconds(stop_times, ['arrival_time', 'departure_time'])
    stop_times_routes = stop_times.merge(trips[['trip_id', 'route_id', 'service_id', 'direction_id']], on='trip_id', how ='left')
    routes_to_update = travel_time_updates[travel_time_updates['project_id'] == project_id]['route_id'].unique()
    unchanged_trips = stop_times_routes[~stop_times_routes['route_id'].isin(routes_to_update)]['trip_id'].unique()
    stop_times_routes = stop_times_routes[stop_times_routes['route_id'].isin(routes_to_update)]

    headways = travel_time_updates[['route_id', 'headway_min']].groupby('route_id').min().reset_index()
    headways['headway_secs'] = headways['headway_min'] * 60
    start_times = pd.DataFrame(stop_times_routes.groupby(['route_id', 'service_id', 'direction_id'])['arrival_time'].min()).rename(columns={'arrival_time': 'start_time'})
    end_times = pd.DataFrame(stop_times_routes.groupby(['route_id', 'service_id', 'direction_id'])['arrival_time'].max()).rename(columns={'arrival_time': 'end_time'})
    frequencies = start_times.join(end_times)
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
    for route in routes_to_update:
        i = 1
        for service_id in frequencies.service_id.unique():
            for direction in frequencies.direction_id.unique():
                selection = (stop_times_routes['route_id'] == route) & (stop_times_routes['service_id'] == service_id) & (stop_times_routes['direction_id'] == direction)
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


def run(project_id, start_time, end_time, weekday):
    bbox = (-59.3177426256, -35.3267410094, -57.6799695705, -34.1435770646)
    nodes, edges, zones = read_process_zones(bbox)
    for scenario in ['baseline', 'project_' + project_id]:
        ua_net = create_ua_network(nodes, edges, bbox, scenario, start_time, end_time, weekday)
        net, zones_net = create_pandana_network(ua_net, scenario, zones)
        calculate_indicators(scenario, net, zones_net)
    compare_indicators(zones, 'project_' + project_id)


def create_ua_network(nodes, edges, bbox, scenario, start_time, end_time, weekday):
    print('Creating UrbanAccess Network')
    gtfs_path = './data/processed/gtfs_%s' % scenario
    ua.osm.network.create_osm_net(osm_edges=edges, osm_nodes=nodes, travel_speed_mph=3)
    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=gtfs_path, bbox=bbox, validation=True,
                                               verbose=True, remove_stops_outsidebbox=True,
                                               append_definitions=True)
    ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds, day=weekday,
                                       timerange=[start_time, end_time], calendar_dates_lookup=None,
                                       time_aware=True, simplify=True)
    loaded_feeds = ua.gtfs.headways.headways(loaded_feeds, [start_time, end_time])
    loaded_feeds.headways = loaded_feeds.headways.groupby('node_id_route').min().reset_index()
    ua.network.integrate_network(urbanaccess_network=ua.network.ua_network, urbanaccess_gtfsfeeds_df=loaded_feeds, headways=True)
    return ua.ua_network


def create_pandana_network(ua_net, scenario, zones):
    print('Creating Pandana Network')
    s_time = time.time()
    net = pdna.Network(ua_net.net_nodes["x"],
                       ua_net.net_nodes["y"],
                       ua_net.net_edges["from_int"],
                       ua_net.net_edges["to_int"],
                       ua_net.net_edges[["weight"]],
                       twoway=False)
    id_df = ua_net.net_nodes.reset_index()[['id_int', 'id']]
    id_df = id_df.set_index('id')
    zones.index = zones.index.astype('str')
    zones = zones.join(id_df)
    net.set(zones['id_int'], variable=zones.jobs, name='jobs')
    net.set(zones['id_int'], variable=zones.lijobs, name='lijobs')
    zones = zones.set_index('id_int')
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    # travel_data = calculate_distance_matrix(zones, 'h3_polyfill')
    # travel_data = calculate_pandana_distances(travel_data, net, zones, 'h3_polyfill')
    # travel_data.to_csv('results/travel_data_%s.csv' % scenario)
    return net, zones


def calculate_distance_matrix(df, id_col):
    coords = [coords for coords in zip(df['y_proj'], df['x_proj'])]
    distances = distance.cdist(coords, coords, 'euclidean')
    df = pd.DataFrame(distances, columns=df[id_col].unique(), index=df[id_col].unique())
    df = df.stack().reset_index().rename(columns={'level_0': 'from_id', 'level_1': 'to_id', 0: 'euclidean_distance'})
    return df


def calculate_pandana_distances(travel_data, net, df, df_id):
    df_from = df.reset_index().rename(columns={df_id: 'from_id', 'id_int': 'node_from'})[['from_id', 'node_from']]
    df_to = df.reset_index().rename(columns={df_id: 'to_id', 'id_int': 'node_to'})[['to_id', 'node_to']]
    travel_data = travel_data.merge(df_from, on='from_id', how='left')
    travel_data = travel_data.merge(df_to, on='to_id', how='left')
    travel_data['pandana_distance'] = net.shortest_path_lengths(list(travel_data['node_from']), list(travel_data['node_to']))
    if travel_data['pandana_distance'].max() > 4000000:
        print('WARNING: NO PATH BETWEEN SOME OD PAIRS')
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
        #net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["distance"]], twoway=False)
        #remove_nodes = set(net.low_connectivity_nodes(impedance=200, count=10, imp_name="distance"))
        #edges = edges[~(edges['from'].isin(remove_nodes) | edges['to'].isin(remove_nodes))]
        #nodes = nodes[~(nodes.index.isin(edges['from'])) | (nodes.index.isin(edges['to']))]
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


def calculate_indicators(scenario, net, zones):
    s_time = time.time()
    print('Aggregating variables')
    zones['jobs_60'] = net.aggregate(60, type='sum', decay='flat', name='jobs')
    zones['lijobs_60'] = net.aggregate(60, type='sum', decay='flat', name='lijobs')
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    if not os.path.exists('results'):
        os.makedirs('./results')
    zones[['h3_polyfil', 'jobs', 'lijobs', 'jobs_60', 'lijobs_60']].to_csv('results/%s.csv' % scenario)
    #zones[['ID', 'jobs', 'jobs_60']].to_csv('results/%s.csv' % scenario)


def compare_indicators(zones, scenario, divide_zones=True):
    buffer_col = 'buff' + scenario.replace('project_', '')
    if divide_zones==True:
        baseline = pd.read_csv('results/baseline.csv').set_index('h3_polyfil')
        project = pd.read_csv('results/%s.csv' % scenario).set_index('h3_polyfil')
    else:
        baseline = pd.read_csv('results/baseline.csv').set_index('ID')
        project = pd.read_csv('results/%s.csv' % scenario).set_index('ID')
    job_cols = [col for col in baseline.columns if 'jobs' in col]
    project = project[job_cols]
    for col in job_cols:
        project = project.rename(columns={col: col+'p'})
    comparison = baseline[job_cols].join(project)
    if divide_zones == True:
        comparison = zones.set_index('h3_polyfil')[['geometry', 'POB10', 'NBI_H10', buffer_col]].join(comparison)
    else:
        comparison = zones.set_index('ID')[['geometry', 'POB10', 'NBI_H10', buffer_col]].join(comparison)


    breakpoint()
    comparison['jobs_60d'] = comparison['jobs_60p'] - comparison['jobs_60']
    comparison['acc_60'] = 100 * (comparison['jobs_60'] / comparison['jobs'].sum())
    comparison['acc_60p'] = 100 * (comparison['jobs_60p'] / comparison['jobsp'].sum())
    comparison['acc_60d'] = comparison['acc_60p'] - comparison['acc_60']
    comparison['pct_ch_acc'] = 100 * (comparison['acc_60d']) / comparison['acc_60']
    comparison['pop_acc'] = comparison['acc_60'] * comparison['POB10']
    comparison['pov_acc'] = comparison['acc_60'] * comparison['NBI_H10']
    comparison['pop_accp'] = comparison['acc_60p'] * comparison['POB10']
    comparison['pov_accp'] = comparison['acc_60p'] * comparison['NBI_H10']


    comparison['lijobs_60d'] = comparison['lijobs_60p'] - comparison['lijobs_60']
    comparison['liacc_60'] = 100 * (comparison['lijobs_60'] / comparison['lijobs'].sum())
    comparison['liacc_60p'] = 100 * (comparison['lijobs_60p'] / comparison['lijobsp'].sum())
    comparison['liacc_60d'] = comparison['liacc_60p'] - comparison['liacc_60']
    comparison['lipct_ch_acc'] = 100 * (comparison['liacc_60d']) / comparison['liacc_60']
    comparison['lipop_acc'] = comparison['liacc_60'] * comparison['POB10']
    comparison['lipov_acc'] = comparison['liacc_60'] * comparison['NBI_H10']
    comparison['lipop_accp'] = comparison['liacc_60p'] * comparison['POB10']
    comparison['lipov_accp'] = comparison['liacc_60p'] * comparison['NBI_H10']

    print('---------------------------------------------')
    print('ENTIRE REGION')
    print('---------------------------------------------')
    orig_pop_acc = comparison['pop_acc'].sum()/comparison['POB10'].sum()
    orig_pov_acc = comparison['pov_acc'].sum()/comparison['NBI_H10'].sum()
    project_pop_acc = comparison['pop_accp'].sum()/comparison['POB10'].sum()
    project_pov_acc = comparison['pov_accp'].sum()/comparison['NBI_H10'].sum()
    pop_acc_change = project_pop_acc - orig_pop_acc
    pov_acc_change = project_pov_acc - orig_pov_acc
    pop_acc_pct_change = 100 * (pop_acc_change / orig_pop_acc)
    pov_acc_pct_change = 100 * (pov_acc_change/orig_pov_acc)
    print('Original population weighted job accessibility:', orig_pop_acc)
    print('Change in population weighted job accessibility:', pop_acc_change)
    print('Percentage change in population weighted job accessibility:', pop_acc_pct_change)
    print('Original poverty weighted job accessibility:', orig_pov_acc)
    print('Change in poverty weighted job accessibility:', pov_acc_change)
    print('Percentage change in poverty weighted job accessibility:', pov_acc_pct_change)
    print('---------------------------------------------')
    print('ENTIRE REGION - LOW INCOME JOBS')
    print('---------------------------------------------')
    orig_pop_acc = comparison['lipop_acc'].sum()/comparison['POB10'].sum()
    orig_pov_acc = comparison['lipov_acc'].sum()/comparison['NBI_H10'].sum()
    project_pop_acc = comparison['lipop_accp'].sum()/comparison['POB10'].sum()
    project_pov_acc = comparison['lipov_accp'].sum()/comparison['NBI_H10'].sum()
    pop_acc_change = project_pop_acc - orig_pop_acc
    pov_acc_change = project_pov_acc - orig_pov_acc
    pop_acc_pct_change = 100 * (pop_acc_change / orig_pop_acc)
    pov_acc_pct_change = 100 * (pov_acc_change/orig_pov_acc)
    print('Original population weighted low income job accessibility:', orig_pop_acc)
    print('Change in population weighted low income job accessibility:', pop_acc_change)
    print('Percentage change in population weighted low income job accessibility:', pop_acc_pct_change)
    print('Original poverty weighted low income job accessibility:', orig_pov_acc)
    print('Change in poverty weighted low income job accessibility:', pov_acc_change)
    print('Percentage change in poverty weighted low income job accessibility:', pov_acc_pct_change)
    print('---------------------------------------------')
    print('AREA OF INFLUENCE')
    print('---------------------------------------------')
    area_comparison = comparison[~comparison[buffer_col].isnull()]
    orig_pop_acc = area_comparison['pop_acc'].sum()/area_comparison['POB10'].sum()
    orig_pov_acc = area_comparison['pov_acc'].sum()/area_comparison['NBI_H10'].sum()
    project_pop_acc = area_comparison['pop_accp'].sum()/area_comparison['POB10'].sum()
    project_pov_acc = area_comparison['pov_accp'].sum()/area_comparison['NBI_H10'].sum()
    pop_acc_change = project_pop_acc - orig_pop_acc
    pov_acc_change = project_pov_acc - orig_pov_acc
    pop_acc_pct_change = 100 * (pop_acc_change / orig_pop_acc)
    pov_acc_pct_change =  100 * (pov_acc_change/orig_pov_acc)
    print('Original population weighted job accessibility in BUFFER:', orig_pop_acc)
    print('Change in population weighted job accessibility in BUFFER:', pop_acc_change)
    print('Percentage change in population weighted job accessibility in BUFFER:', pop_acc_pct_change)
    print('Original poverty weighted job accessibility in BUFFER:', orig_pov_acc)
    print('Change in poverty weighted job accessibility in BUFFER:', pov_acc_change)
    print('Percentage change in poverty weighted job accessibility in BUFFER:', pov_acc_pct_change)
    print('---------------------------------------------')
    print('AREA OF INFLUENCE - LOW INCOME JOBS')
    print('---------------------------------------------')
    orig_pop_acc = area_comparison['lipop_acc'].sum()/area_comparison['POB10'].sum()
    orig_pov_acc = area_comparison['lipov_acc'].sum()/area_comparison['NBI_H10'].sum()
    project_pop_acc = area_comparison['lipop_accp'].sum()/area_comparison['POB10'].sum()
    project_pov_acc = area_comparison['lipov_accp'].sum()/area_comparison['NBI_H10'].sum()
    pop_acc_change = project_pop_acc - orig_pop_acc
    pov_acc_change = project_pov_acc - orig_pov_acc
    pop_acc_pct_change = 100 * (pop_acc_change / orig_pop_acc)
    pov_acc_pct_change = 100 * (pov_acc_change/orig_pov_acc)
    print('Original population weighted job accessibility in BUFFER:', orig_pop_acc)
    print('Change in population weighted job accessibility in BUFFER:', pop_acc_change)
    print('Percentage change in population weighted job accessibility in BUFFER:', pop_acc_pct_change)
    print('Original poverty weighted job accessibility in BUFFER:', orig_pov_acc)
    print('Change in poverty weighted job accessibility in BUFFER:', pov_acc_change)
    print('Percentage change in poverty weighted job accessibility in BUFFER:', pov_acc_pct_change)

    breakpoint()
    comparison = comparison.reindex(sorted(comparison.columns), axis=1)
    comparison = comparison.reset_index().fillna(0)
    id_cols = ['h3_polyfil', 'NBI_H10', 'POB10', buffer_col]
    job_cols = ['jobs', 'jobs_60', 'jobs_60p', 'jobs_60d', 'acc_60', 'acc_60p', 'acc_60d', 'pct_ch_acc', 'pop_acc', 'pop_accp', 'pov_acc', 'pov_accp']
    low_income_job_cols = ['li' + col for col in job_cols]
    comparison = comparison[id_cols + job_cols + low_income_job_cols + 'geometry']
    comparison.to_file('results/final_results_%s.shp' % scenario)
    breakpoint()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ud", "--update_demographics", action="store_true", default=False, help="update demographics")
    parser.add_argument("-ug", "--update_gtfs", action="store_true", default=False, help="update_gtfs")
    parser.add_argument("-p", "--project_id", type=str, default=False, help="project_id to evaluate")
    parser.add_argument("-st", "--start_time", type=str, default=False, help="start time for_analysis in 24 hr format (ej 07:00)")
    parser.add_argument("-et", "--end_time", type=str, default=False, help="end time for analysis in 24 hr format (ej 08:00)")
    parser.add_argument("-d", "--weekday", type=str, default=False, help="week day for analysis in 24 hr format (ej monday)")
    args = parser.parse_args()

    update_demographics = args.update_demographics if args.update_demographics else False
    update_gtfs = args.update_gtfs if args.update_gtfs else False
    project_id = args.project_id if args.project_id else '1'
    start_time = args.start_time if args.start_time else '07:00:00'
    end_time = args.end_time if args.end_time else '08:00:00'
    weekday = args.weekday if args.weekday else 'monday'

    if update_gtfs:
        process_update_gtfs()
    if update_demographics:
        process_update_demographics()
    run(project_id, start_time, end_time, weekday)


