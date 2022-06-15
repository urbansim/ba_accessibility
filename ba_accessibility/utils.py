import os
import argparse
import pandas as pd
import pandana as pdna
import geopandas as gpd
from scipy.spatial import distance
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs
import urbanaccess as ua

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
                                       use_existing_stop_times_int=True)
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
    stop_times_df = stop_times_df[stop_times_df['unique_trip_id'].isin(uniquetriplist)]
    stop_times_df = ua.gtfs.network._time_difference(stop_times_df=stop_times_df)
    loaded_feeds.stop_times_int = stop_times_df
    return loaded_feeds



def calculate_distance_matrix():
    print('Calculating euclidian distance matrix')
    df = gpd.read_file('results/zones.shp').set_index('id_int')
    id_col = 'h3_polyfil'
    coords = [coords for coords in zip(df['y_proj'], df['x_proj'])]
    distances = distance.cdist(coords, coords, 'euclidean')
    distances = pd.DataFrame(distances, columns=df[id_col].unique(), index=df[id_col].unique())
    distances = distances.stack().reset_index().rename(columns={'level_0': 'from_id', 'level_1': 'to_id', 0: 'euclidean_distance'})
    df_to = df.reset_index().rename(columns={'h3_polyfil': 'to_id', 'id_int': 'node_to'}).set_index('to_id')[['jobs', 'lijobs', 'node_to']]
    df_from = df.reset_index().rename(columns={'h3_polyfil': 'from_id', 'id_int': 'node_from'}).set_index('from_id')[['node_from']]
    distances = distances.set_index('to_id').join(df_to).reset_index()
    distances = distances.set_index('from_id').join(df_from).reset_index()
    print('Distance matrix calculation done')
    distances.to_csv('results/distances.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bb", "--bounding_box", type=str, default=False, help="")
    parser.add_argument("-sc", "--scenario", type=str, default=False, help="")
    parser.add_argument("-st", "--start_time", type=str, default=False, help="start time for_analysis in 24 hr format (ej 07:00)")
    parser.add_argument("-et", "--end_time", type=str, default=False, help="end time for analysis in 24 hr format (ej 08:00)")
    parser.add_argument("-d", "--weekday", type=str, default=False, help="week day for analysis in 24 hr format (ej monday)")
    parser.add_argument("-ua", "--urbanaccess_net", action="store_true", default=False, help="create urbanaccess net")
    parser.add_argument("-ed", "--euclidean_dist", action="store_true", default=False, help="create euclidean distance matrix")
    args = parser.parse_args()

    bounding_box = eval(args.bounding_box) if args.bounding_box else (-59.3177426256, -35.3267410094, -57.6799695705, -34.1435770646)
    scenario = args.scenario if args.scenario else '07:00:00'
    start_time = args.start_time if args.start_time else '07:00:00'
    end_time = args.end_time if args.end_time else '08:00:00'
    weekday = args.weekday if args.weekday else 'monday'
    urbanaccess_net = args.urbanaccess_net if args.urbanaccess_net else False
    euclidean_dist = args.euclidean_dist if args.euclidean_dist else False

    if urbanaccess_net:
        create_ua_network(bounding_box, scenario, start_time, end_time, weekday)

    if euclidean_dist:
        calculate_distance_matrix()
