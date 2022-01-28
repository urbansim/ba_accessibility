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
from shapely.geometry import Point
from shapely.geometry import LineString
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs
import h3pandas

def process_update_jobs():
    s_time = time.time()
    zones = gpd.read_file('data/original/jobs/Empleo.shp')
    resolution=8 #10
    hexagons = zones.h3.polyfill_resample(resolution)
    hexagons = hexagons.to_crs(22192)
    hexagons['hex_area'] = hexagons.geometry.area
    zone_area = hexagons.groupby('ID')['hex_area'].sum().reset_index()
    zone_area = zone_area.rename(columns={'hex_area': 'zone_area'})
    hexagons = hexagons.reset_index().merge(zone_area, on='ID', how='left')
    job_cols = ['jobs', 'job_a', 'job_b', 'job_c', 'job_d', 'job_h']
    for col in job_cols:
        hexagons[col] = round(hexagons[col].fillna(0).astype('int') * hexagons['hex_area'] / hexagons['zone_area'])
    if not os.path.exists('data/processed/jobs'):
        os.makedirs('./data/processed/jobs')
    hexagons[['h3_polyfill', 'ID', 'geometry'] + job_cols].to_file('data/processed/jobs/jobs_hexagons.shp')
    print('Took {:,.2f} seconds to process jobs shapefile'.format(time.time() - s_time))


def process_update_gtfs():
    print('Processing all GTFS feeds to required input format')
    copy_files()
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


def copy_files():
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
    for i in range(0, num_batches):
        percentage_progress = 100 * i/num_batches
        if (percentage_progress - prev_progress) > 10:
            prev_progress = percentage_progress
            print('    ', round(percentage_progress), '% progress')
        upper_bound = min((i+1)*len_batch, len(trip_ids))
        selected_trip_ids = trip_ids[lower_bound:upper_bound]
        selected_frequencies = frequencies[frequencies['trip_id'].isin(selected_trip_ids)]
        selected_stop_times = stop_times[stop_times['trip_id'].isin(selected_trip_ids)]
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
        min_arrival_time = stop_times.arrival_time.min()
        df['arrival_time'] = df['start_time'] + df['headway_secs'] * df['trip_repetition'] + df['arrival_time'] - min_arrival_time
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
    for dir in sorted(os.listdir('data/processed/gtfs_baseline')):
        if not os.path.exists('data/processed/gtfs_project/%s' % (dir)):
            os.makedirs('./data/processed/gtfs_project/%s' % (dir))
        gtfs_files = ['agency', 'calendar', 'feed_info', 'routes', 'stop_times', 'stops', 'trips', 'frequencies']
        for file in gtfs_files:
            if os.path.isfile('data/processed/gtfs_baseline/%s/%s.txt' % (dir, file)):
                shutil.copy('./data/processed/gtfs_baseline/%s/%s.txt' % (dir, file),
                            './data/processed/gtfs_project/%s/%s.txt' % (dir, file))
    updates = pd.read_csv('data/original/project_modifications.csv')
    updates = updates.fillna(0)
    updates['stop_id'] = updates['stop_id'].astype('int')
    updates['cumsum'] = updates['min_since_prev'].cumsum()
    updates.loc[updates['min_since_prev'] == 0, 'restart'] = -1 * updates['cumsum']
    updates['restart'] = updates['restart'].fillna(0).cumsum()
    updates['cumsum'] = updates['cumsum'] + updates['restart']
    updated_stop_times = pd.DataFrame()
    for project in updates.project_id.unique():
        updates_project = updates[updates['project_id'] == project]
        mode = updates_project['mode'].unique()[0]
        original_path = ('data/processed/gtfs_baseline/%s' % mode)
        trips = pd.read_csv('%s/trips.txt' % original_path)
        stop_times = pd.read_csv('%s/stop_times.txt' % original_path)
        stop_times_routes = stop_times.merge(trips[['trip_id', 'route_id', 'service_id']], on='trip_id', how='left')
        for route in updates_project['route_id'].unique():
            updates_project_route = updates_project[updates_project['route_id'] == route]
            updates_project_route['prev_stop_id'] = updates_project_route['stop_id'].shift().fillna(0).astype('int')
            updates_project_route['to_from'] = updates_project_route['prev_stop_id'].astype('str') + '_' + updates_project_route['stop_id'].astype('str')
            stop_times_route = stop_times_routes[stop_times_routes['route_id'] == route]
            stop_times_route = stop_times_route.sort_values(by=['trip_id', 'service_id', 'stop_sequence'])
            stop_times_route = time_to_seconds(stop_times_route, ['arrival_time', 'departure_time'])
            arrival_min = stop_times_route.groupby('trip_id')['arrival_time'].min()
            departure_min = stop_times_route.groupby('trip_id')['departure_time'].min()
            stop_times_route = stop_times_route.set_index('trip_id')
            stop_times_route['arrival_min'] = arrival_min
            stop_times_route['departure_min'] = departure_min
            stop_times_route['prev_stop_id'] = stop_times_route['stop_id'].shift().fillna(0).astype('int')
            stop_times_route['to_from'] = stop_times_route['prev_stop_id'].astype('str') + '_' + stop_times_route['stop_id'].astype('str')
            stop_times_route = stop_times_route.reset_index()
            stop_times_route = stop_times_route.merge(updates_project_route[['to_from', 'cumsum']], on=['to_from'], how='left')
            stop_times_route['arrival_time'] = stop_times_route['arrival_min'] + stop_times_route['cumsum'].fillna(0) * 60
            stop_times_route['departure_time'] = stop_times_route['departure_min'] + stop_times_route['cumsum'].fillna(0) * 60
            stop_times_route['arrival_time'] = pd.to_datetime(stop_times_route['arrival_time'].round(), unit='s').dt.time
            stop_times_route['departure_time'] = pd.to_datetime(stop_times_route['departure_time'].round(), unit='s').dt.time
            stop_times_route = stop_times_route[list(stop_times.columns)]
            updated_stop_times = updated_stop_times.append(stop_times_route)
    equal_stop_times = stop_times_routes[~stop_times_routes['route_id'].isin(updates['route_id'])][list(stop_times.columns)]
    updated_stop_times.append(equal_stop_times).to_csv('data/processed/gtfs_project/%s/stop_times.txt' % mode)


def run(start_time, end_time, weekday):
    for scenario in ['baseline', 'project']:
        create_ua_network(scenario, start_time, end_time, weekday)
        net = create_pandana_network(scenario)
        net, zones = read_process_zones(net)
        calculate_indicators(scenario, net, zones)
    compare_indicators(zones)


def create_ua_network(scenario, start_time, end_time, weekday):
    print('Creating UrbanAccess Network')
    gtfs_path = './data/processed/gtfs_%s' % scenario
    bbox = (-59.3177426256,-35.3267410094,-57.6799695705,-34.1435770646)
    nodes, edges = ua.osm.load.ua_network_from_bbox(bbox=bbox, remove_lcn=True)
    ua.osm.network.create_osm_net(osm_edges=edges, osm_nodes=nodes, travel_speed_mph=3)
    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=gtfs_path, bbox=bbox, validation=True,
                                               verbose=True, remove_stops_outsidebbox=True,
                                               append_definitions=True)
    ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds,
                                       day=weekday,
                                       timerange=[start_time, end_time],
                                       calendar_dates_lookup=None,
                                       time_aware=True,
                                       simplify=True)
    loaded_feeds = ua.gtfs.headways.headways(loaded_feeds, [start_time, end_time])
    ua.network.integrate_network(urbanaccess_network=ua.network.ua_network, urbanaccess_gtfsfeeds_df=loaded_feeds, headways=True)
    ua.network.save_network(urbanaccess_network=ua.network.ua_network, filename='final_%s_net.h5' % scenario, overwrite_key=True)


def create_pandana_network(scenario):
    print('Loading Precomputed UrbanAccess Network')
    ua_net = ua.network.load_network(filename='final_%s_net.h5' % scenario)
    #export_shp(ua_net.net_nodes, ua_net.net_edges)

    print('Creating Pandana Network')
    s_time = time.time()
    net = pdna.Network(ua_net.net_nodes["x"],
                       ua_net.net_nodes["y"],
                       ua_net.net_edges["from_int"],
                       ua_net.net_edges["to_int"],
                       ua_net.net_edges[["weight"]],
                       twoway=False)
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    return net


def export_shp(nodes, edges, name_shp='test', df=None):
    nodes_from = nodes.rename(columns={'id': 'from', 'x': 'x_from', 'y': 'y_from'})
    nodes_to = nodes.rename(columns={'id': 'to', 'x': 'x_to', 'y': 'y_to'})
    edges = edges.merge(nodes_from, on='from', how='left')
    edges = edges.merge(nodes_to, on='to', how='left')
    edges['geometry'] = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in
                         zip(edges['x_from'], edges['y_from'], edges['x_to'], edges['y_to'])]
    nodes['geometry'] = [Point(xy) for xy in zip(nodes['x'], nodes['y'])]
    edges_gdf = gpd.GeoDataFrame(edges, crs={'init': 'epsg:4326'}, geometry='geometry')
    nodes_gdf = gpd.GeoDataFrame(nodes, crs={'init': 'epsg:4326'}, geometry='geometry')

    nodes_gdf.to_file(name_shp + '_nodes.shp')
    edges_gdf.to_file(name_shp + '_edges.shp')
    if df is not None:
        df['geometry'] = [Point(xy) for xy in zip(df['x'], df['y'])]
        zones_gdf = GeoDataFrame(df, geometry='geometry', crs={'init': 'epsg:4326'})
        zones_gdf.to_file(name_shp + '_zones.shp')


def read_process_zones(net):
    zones = gpd.read_file('data/processed/jobs/jobs_hexagons.shp')
    zones['centroid'] = zones['geometry'].centroid
    zones = zones.set_geometry('centroid')
    zones = zones.rename(columns={'geometry': 'polygon_geometry', 'centroid': 'geometry'})
    zones = zones.set_geometry('geometry')
    zones = zones.to_crs(4326)
    zones['x'] = zones.geometry.x
    zones['y'] = zones.geometry.y
    zones['node_id'] = net.get_node_ids(zones['x'], zones['y'])
    zones = zones.to_crs(22192)
    zones = zones.rename(columns={'geometry': 'centroid', 'polygon_geometry': 'geometry'})
    zones = zones.set_geometry('geometry').drop(columns='centroid')
    net.set(zones.node_id, variable=zones.jobs, name='jobs')
    zones = zones.set_index('node_id')
    return net, zones


def calculate_indicators(scenario, net, zones):
    s_time = time.time()
    print('Aggregating variables')
    for i in [15, 30, 45, 60]:
        zones['jobs_' + str(i)] = net.aggregate(i, type='sum', decay='flat', name='jobs')
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    if not os.path.exists('results'):
        os.makedirs('./results')
    zones[['h3_polyfil', 'ID', 'jobs', 'jobs_15', 'jobs_30', 'jobs_45', 'jobs_60']].to_csv('results/%s.csv' % scenario)


def compare_indicators(zones):
    baseline = pd.read_csv('results/baseline.csv').set_index('h3_polyfil')
    project = pd.read_csv('results/project.csv').set_index('h3_polyfil')
    job_cols = [col for col in baseline.columns if 'jobs' in col]
    project = project[job_cols]
    for col in job_cols:
        project = project.rename(columns={col: col+'_p'})
    comparison = baseline[job_cols].join(project)
    for col in job_cols:
        comparison['pct' + col.replace('jobs', '')] = comparison[col] / comparison['jobs'].sum()
        comparison['pct' + col.replace('jobs', '') + '_p'] = comparison[col + '_p'] / comparison['jobs_p'].sum()
        comparison[col + '_d'] = comparison[col + '_p'] - comparison[col]
        comparison[col.replace('jobs', 'pct_ch')] = (comparison[col + '_d']) / comparison[col]
    comparison = comparison.fillna(0)
    comparison = zones.set_index('h3_polyfil')[['geometry']].join(comparison)
    comparison.to_file('results.shp')
    breakpoint()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-uj", "--update_jobs", action="store_true", default=False, help="update_jobs")
    parser.add_argument("-ug", "--update_gtfs", action="store_true", default=False, help="update_gtfs")
    parser.add_argument("-st", "--start_time", type=str, default=False, help="start time for_analysis in 24 hr format (ej 07:00)")
    parser.add_argument("-et", "--end_time", type=str, default=False, help="end time for analysis in 24 hr format (ej 08:00)")
    parser.add_argument("-d", "--weekday", type=str, default=False, help="week day for analysis in 24 hr format (ej monday)")
    args = parser.parse_args()

    update_jobs = args.update_jobs if args.update_jobs else False
    update_gtfs = args.update_gtfs if args.update_gtfs else False
    start_time = args.start_time if args.start_time else '07:00:00'
    end_time = args.end_time if args.end_time else '08:00:00'
    weekday = args.weekday if args.weekday else 'monday'

    if update_jobs:
        process_update_jobs()
    if update_gtfs:
        process_update_gtfs()
    run(start_time, end_time, weekday)


