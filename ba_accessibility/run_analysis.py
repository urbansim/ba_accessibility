import os
import gc
import sys
import time
import math
import yaml
import psutil
import argparse
import subprocess
import pandas as pd
import pandana as pdna
import geopandas as gpd
import matplotlib.pyplot as plt


def run(project_ids, update_gtfs, update_demographics):
    with open("data/configs.yaml") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    bbox = eval(configs['bounding_box'])
    start_time = configs['start_time']
    end_time = configs['end_time']
    weekday = configs['weekday']
    process_updates(update_gtfs, update_demographics)
    project_scenarios = ['project_' + project_id for project_id in project_ids]
    for scenario in ['baseline'] + project_scenarios:
        create_ua_network(bbox, scenario, start_time, end_time, weekday)
        net, zones_net, travel_data = create_pandana_network()
        calculate_indicators(scenario, zones_net, travel_data)
        del net, zones_net, travel_data
        gc.collect()
    results = pd.DataFrame()
    for scenario in project_scenarios:
        results = compare_indicators(scenario, results)
    summarize_results(results)


def process_updates(update_gtfs, update_demographics):
    if update_gtfs:
        update_args = [sys.executable, 'utils.py', '-ug']
        subprocess.check_call(update_args)
    if update_demographics:
        update_args = [sys.executable, 'utils.py', '-ud']
        subprocess.check_call(update_args)


def create_ua_network(bbox, scenario, start_time, end_time, weekday):
    ua_args = [sys.executable, 'utils.py', '-ua', '-bb', bbox, '-sc',
               scenario, '-st', start_time, '-et', end_time, '-d', weekday]
    subprocess.check_call(ua_args)


def create_pandana_network():
    print('Creating Pandana Network')
    s_time = time.time()
    ua_nodes = pd.read_csv('results/ua_nodes.csv', dtype={'x': float, 'y': float}).set_index('id_int')
    ua_edges = pd.read_csv('results/ua_edges.csv', dtype={'from_int': int, 'to_int': int, 'weight': float})
    net = pdna.Network(ua_nodes["x"], ua_nodes["y"], ua_edges["from_int"], ua_edges["to_int"],
                       ua_edges[["weight"]], twoway=False)
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
    batch_length = 10000000
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


def calculate_indicators(scenario, zones, travel_data):
    s_time = time.time()
    print('Calculating indicators from skims')
    within_60_min = travel_data[travel_data['pandana_distance'] <= 60]
    rename_dict = {'jobs': 'jobs_60', 'lijobs': 'lijobs_60'}
    jobs_60 = within_60_min.groupby('from_id')['jobs', 'lijobs'].sum().rename(columns=rename_dict)
    within_90_min = travel_data[travel_data['pandana_distance'] <= 90]
    rename_dict = {'jobs': 'jobs_90', 'lijobs': 'lijobs_90'}
    jobs_90 = within_90_min.groupby('from_id')['jobs', 'lijobs'].sum().rename(columns=rename_dict)
    zones = zones.set_index('zone_id').join(jobs_60)
    zones = zones.join(jobs_90)
    rename_dict = {'pandana_distance': 'time_cbd'}
    times_to_cbd = travel_data[travel_data['to_id'] == '88c2e31ad1fffff'].rename(columns=rename_dict)
    zones = zones.join(times_to_cbd.set_index('from_id')[['time_cbd']])
    print('Took {:,.2f} seconds'.format(time.time() - s_time))
    if not os.path.exists('results'):
        os.makedirs('./results')
    zones_cols = ['jobs', 'lijobs', 'jobs_60', 'lijobs_60', 'jobs_90', 'lijobs_90', 'time_cbd']
    zones[zones_cols].to_csv('results/%s.csv' % scenario)


def compare_indicators(scenario, results):
    print('Comparing scenario %s with Baseline' % scenario)
    zones = gpd.read_file('results/zones.shp')
    buffer_col = 'buff' + scenario.replace('project_', '')
    baseline = pd.read_csv('results/baseline.csv').set_index('zone_id')
    project = pd.read_csv('results/%s.csv' % scenario).set_index('zone_id')
    job_cols = [col for col in baseline.columns if 'jobs' in col] + ['time_cbd']
    project = project[job_cols]
    for col in job_cols:
        project = project.rename(columns={col: col+'p'})
    comparison = baseline[job_cols].join(project)
    comparison = zones.set_index('zone_id')[['geometry', 'POB10', 'HOG10', 'NBI_H10', buffer_col]].join(comparison)
    project_id = scenario.replace('project_', '')
    nodes = gpd.read_file('results/project_trajectories/%s_trajectory_nodes.shp' % scenario).to_crs(comparison.crs)
    edges = gpd.read_file('results/project_trajectories/%s_trajectory_edges.shp' % scenario).to_crs(comparison.crs)
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
                comparison.plot(column='%sacc_%sd' % (jt, tr), ax=ax, cmap='Spectral', legend=True)
                edges.plot(ax=ax, color='black', linewidth=0.5)
                nodes.plot(ax=ax, color='black', markersize=0.5)
                ax.axis('off')
                plt.savefig("results/accessibility_change_%s_min.png" % tr)
    results = pd.concat([results, comparison])
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
                    all_stations = buffer.stop_name_from.str.lower().unique()
                    # remove Constitución and Buenos Aires from analysis
                    remove_stations = [s for s in all_stations if 'constitu' in s or 'aires' in s]
                    project_stations = buffer[~buffer['stop_name_from'].str.lower().isin(remove_stations)]
                else:
                    modified_routes = pd.read_csv('data/original/project_updates/modified_routes.csv').fillna(0)
                    modified_routes = modified_routes[modified_routes['project_id'].astype('str') == project_id]
                    stops_locations = modified_routes.groupby('stop_id')['stop_name_from'].min()
                    project_stations = gpd.read_file('results/project_trajectories/project_%s_trajectory_nodes.shp' % project_id)
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
                    results_analysis = pd.concat([results_analysis, results_stations])

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
             'pov_pct_ch': 'Percentage Change in Poverty Weighted Number of Jobs Accessible'}
    results_analysis = results_analysis.reset_index().set_index(['stop_name_from', 'buffer', 'threshold', 'project'])
    results_analysis = results_analysis[list(names.keys())]
    results_analysis = results_analysis.rename(columns=names)
    results_analysis.to_csv('results/results_all_projects.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ud", "--update_demographics", action="store_true", default=False, help="update demographics")
    parser.add_argument("-ug", "--update_gtfs", action="store_true", default=False, help="update_gtfs")
    parser.add_argument("-p", "--project_ids", type=str, nargs='+', default=[])
    args = parser.parse_args()

    update_gtfs = args.update_gtfs if args.update_gtfs else False
    update_demographics = args.update_demographics if args.update_demographics else False
    project_ids = args.project_ids if args.project_ids else ['1']

    run(project_ids, update_gtfs, update_demographics)
