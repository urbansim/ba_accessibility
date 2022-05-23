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
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.geometry import LineString
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs
import h3pandas
import matplotlib.pyplot as plt

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
    ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds, day=weekday,
                                       timerange=[start_time, end_time], calendar_dates_lookup=None,
                                       time_aware=True, simplify=True)
    loaded_feeds = ua.gtfs.headways.headways(loaded_feeds, [start_time, end_time])
    loaded_feeds.headways = loaded_feeds.headways.groupby('node_id_route').min().reset_index()
    loaded_feeds.headways.loc[loaded_feeds.headways['mean'].isnull(), 'mean'] = 60
    ua.network.integrate_network(urbanaccess_network=ua.network.ua_network, urbanaccess_gtfsfeeds_df=loaded_feeds, headways=True)
    ua_nodes = ua.ua_network.net_nodes[["id","x","y"]]
    ua_edges = ua.ua_network.net_edges[["from_int", "to_int", "weight"]]
    ua_nodes.to_csv('results/ua_nodes.csv')
    ua_edges.to_csv('results/ua_edges.csv', index=False)
    id_df = ua.ua_network.net_nodes.reset_index()[['id_int', 'id']]
    id_df = id_df.set_index('id')
    zones.index = zones.index.astype('str')
    zones = zones.join(id_df)
    zones.index.name = 'node_id'
    zones.to_file('results/zones.shp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bb", "--bounding_box", type=str, default=False, help="")
    parser.add_argument("-sc", "--scenario", type=str, default=False, help="")
    parser.add_argument("-st", "--start_time", type=str, default=False, help="start time for_analysis in 24 hr format (ej 07:00)")
    parser.add_argument("-et", "--end_time", type=str, default=False, help="end time for analysis in 24 hr format (ej 08:00)")
    parser.add_argument("-d", "--weekday", type=str, default=False, help="week day for analysis in 24 hr format (ej monday)")
    args = parser.parse_args()

    bounding_box = eval(args.bounding_box) if args.bounding_box else (-59.3177426256, -35.3267410094, -57.6799695705, -34.1435770646)
    scenario = args.scenario if args.scenario else '07:00:00'
    start_time = args.start_time if args.start_time else '07:00:00'
    end_time = args.end_time if args.end_time else '08:00:00'
    weekday = args.weekday if args.weekday else 'monday'

    create_ua_network(bounding_box, scenario, start_time, end_time, weekday)
