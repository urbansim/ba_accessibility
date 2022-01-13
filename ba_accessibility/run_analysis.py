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


def process_gtfs():
    print('Processing all GTFS feeds to required input format')
    for mode in ['subte', 'trenes', 'colectivos']:
        print('---------------------------')
        print('Formatting', mode)
        s_time = time.time()
        copy_files(mode)
        frequencies = glob.glob('./data/original_gtfs/%s/frequencies.txt' % mode)
        if len(frequencies) > 0:
            frequencies, stop_times, trips, routes, agencies = read_files(mode)
            frequencies, stop_times, routes, agencies = format_inputs(frequencies, stop_times, routes, agencies)
            expanded_stop_times, unique_trips = expand_stop_times(frequencies, stop_times)
            expanded_trips = expand_trips(trips, unique_trips)
            export_outputs(mode, stop_times, trips, expanded_stop_times, expanded_trips, routes, agencies)
        print('Took %s seconds to process %s' %(time.time() - s_time, mode))



def copy_files(mode):
    if not os.path.exists('data/processed_gtfs/%s' % mode):
        os.makedirs('./data/processed_gtfs/%s' % mode)
    gtfs_files = ['agency', 'calendar', 'feed_info', 'routes', 'shapes', 'stop_times', 'stops', 'trips']
    for file in gtfs_files:
        shutil.copy('./data/original_gtfs/%s/%s.txt' % (mode, file),
                    './data/processed_gtfs/%s/%s.txt' % (mode, file))


def read_files(mode):
    print('Reading files')
    frequencies = pd.read_csv('data/original_gtfs/%s/frequencies.txt' % mode)
    stop_times = pd.read_csv('data/original_gtfs/%s/stop_times.txt' % mode)
    trips = pd.read_csv('data/original_gtfs/%s/trips.txt' % mode)
    routes = pd.read_csv('data/original_gtfs/%s/routes.txt' % mode)
    agencies = pd.read_csv('data/original_gtfs/%s/agency.txt' % mode)
    return frequencies, stop_times, trips, routes, agencies


def format_inputs(frequencies, stop_times, routes, agencies):
    print('Formatting inputs')
    frequencies = time_to_seconds(frequencies, ['start_time', 'end_time'])
    stop_times = time_to_seconds(stop_times, ['arrival_time', 'departure_time'])
    stop_times = stop_times.sort_values(by=['trip_id', 'stop_sequence'])
    if len(stop_times[stop_times['arrival_time'].isnull()].index) > 0:
        stop_times['arrival_time'] = stop_times['arrival_time'].interpolate()
        stop_times['departure_time'] = stop_times['departure_time'].interpolate()
    stop_times['stop_duration'] = stop_times['departure_time'] - stop_times['arrival_time']
    routes['agency_id'] = routes['agency_id'].min()
    agencies = agencies.head(1).copy()
    return frequencies, stop_times, routes, agencies


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
    num_batches = round(len(trip_ids)/len_batch)
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


def export_outputs(mode, stop_times, trips, expanded_stop_times, expanded_trips, routes, agencies):
    print('Exporting stop times')
    expanded_stop_times[stop_times.columns].to_csv('./data/processed_gtfs/%s/stop_times.txt' % mode, index=False)
    print('Exporting trips')
    expanded_trips[trips.columns].to_csv('./data/processed_gtfs/%s/trips.txt' % mode, index=False)
    print('Exporting routes')
    routes.to_csv('./data/processed_gtfs/%s/routes.txt' % mode, index=False)
    print('Exporting agencies')
    agencies.to_csv('./data/processed_gtfs/%s/agency.txt' % mode, index=False)


def run(start_time, end_time, weekday):
    create_ua_network(start_time, end_time, weekday)
    create_pandana_network()
    net = create_pandana_network()
    net, zones = read_process_zones(net)
    calculate_indicators(net, zones)


def create_ua_network(start_time, end_time, weekday):
    print('Creating UrbanAccess Network')
    gtfs_path = './data/processed_gtfs'
    bbox = (-59.3201, -35.1845, -57.7988, -34.2464)
    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=gtfs_path, bbox=bbox, validation=True,
                                               verbose=True, remove_stops_outsidebbox=True,
                                               append_definitions=True)

    #loaded_feeds.stops.plot(kind='scatter', x='stop_lon', y='stop_lat', s=0.1)

    ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds,
                                       day=weekday,
                                       timerange=[start_time, end_time],
                                       calendar_dates_lookup=None)

    nodes, edges = ua.osm.load.ua_network_from_bbox(bbox=bbox,remove_lcn=True)
    ua.osm.network.create_osm_net(osm_edges=edges, osm_nodes=nodes, travel_speed_mph=3)
    ua_net = ua.network.ua_network
    ua.network.integrate_network(urbanaccess_network=ua_net, headways=False)
    ua.network.save_network(urbanaccess_network=ua_net, filename='final_net.h5', overwrite_key = True)


def create_pandana_network():
    print('Loading Precomputed UrbanAccess Network')
    ua_net = ua.network.load_network(filename='final_net.h5')
    export_shp(ua_net.net_nodes, ua_net.net_edges)

    print('Creating Pandana Network')
    s_time = time.time()
    net = pdna.Network(ua_net.net_nodes["x"],
                       ua_net.net_nodes["y"],
                       ua_net.net_edges["from_int"],
                       ua_net.net_edges["to_int"],
                       ua_net.net_edges[["weight"]],
                       twoway=False)
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    precompute_time = 45
    print('Precomputing network for distance %s.' % precompute_time)
    print('Network precompute starting.')
    net.precompute(precompute_time)
    print('Network precompute done.')
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
    zones = gpd.read_file('data/original_jobs/Empleo.shp')
    zones['x'] = zones['geometry'].centroid.x
    zones['y'] = zones['geometry'].centroid.y
    zones['node_id'] = net.get_node_ids(zones['x'], zones['y'])
    net.set(zones.node_id, variable=zones.jobtotal, name='jobs')
    zones = zones.set_index('node_id')
    return net, zones


def calculate_indicators(net, zones):
    s_time = time.time()
    print('Aggregating variables')
    for i in [15, 30, 45]:
        zones['jobs_' + str(i)] = net.aggregate(i, type='sum', decay='linear', name='jobs')
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    zones.plot('jobs_45', cmap='gist_heat_r', edgecolor='none', figsize=(20,20), legend=True)
    zones.plot('jobtotal', cmap='gist_heat_r', edgecolor='none', figsize=(20,20), legend=True)
    zones.to_file('results.shp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--update_gtfs", action="store_true", default=False, help="update_gtfs")
    parser.add_argument("-st", "--start_time", type=str, default=False, help="start time for_analysis in 24 hr format (ej 07:00)")
    parser.add_argument("-et", "--end_time", type=str, default=False, help="end time for analysis in 24 hr format (ej 08:00)")
    parser.add_argument("-d", "--weekday", type=str, default=False, help="week day for analysis in 24 hr format (ej monday)")
    args = parser.parse_args()
    update_gtfs = args.update_gtfs if args.update_gtfs else False
    start_time = args.start_time if args.start_time else '07:00:00'
    end_time = args.end_time if args.end_time else '08:00:00'
    weekday = args.weekday if args.weekday else 'monday'

    if update_gtfs:
        process_gtfs()
    run(start_time, end_time, weekday)


