import os
import glob
import random
import logging

import numpy as np
from tqdm import tqdm

from DriveSceneGen.utils.io import *
from DriveSceneGen.vectorization.evaluation import map_metrics

logger = get_logger('compute_map_metrics', logging.WARNING)


if __name__ == "__main__":
    n_proc = 1  # Numer of available processors
    
    map_range = 120
    map_res = 256
    
    ######################################### Load Data #######################################

    # Ground Truth Data
    # gt_data_dir = '/media/shuo/Cappuccino/DriveSceneGen/ground_truth'
    # gt_data_dir = '/media/shuo/Cappuccino/DriveSceneGen/waymo1.2_mini'
    gt_data_dir = '/media/shuo/Cappuccino/DriveSceneGen/waymo1.2'
    
    gt_metrics_dir = os.path.join(gt_data_dir, 'metrics')
    os.makedirs(gt_metrics_dir, exist_ok=True)
    
    if not os.path.exists(get_cache_name(gt_data_dir, 'graph')):
        print('Caching graph filenames...')
        cache_all_filenames(gt_data_dir, 'graph')
    if not os.path.exists(get_cache_name(gt_data_dir, 'track')):
        print('Caching track filenames...')
        cache_all_filenames(gt_data_dir, 'track')
    
    gt_graph_files = get_all_filenames(gt_data_dir, 'graph')
    gt_track_files = get_all_filenames(gt_data_dir, 'track')

    gt_graph_files = random.sample(gt_graph_files, 5000)
    # gt_track_files = random.sample(gt_track_files, 5000)
    
    # Compute Map statistics using HDMapGen metrics
    # map_metrics.compute_map_stats(gt_graph_files, gt_metrics_dir, map_range=map_range, map_res=map_res)
    # map_metrics.compute_map_stats(gt_graph_files, gt_metrics_dir) # 5000 samples about 6-12 hours
    
    # Generated Data
    gen_data_dir = f'/media/shuo/Cappuccino/DriveSceneGen/generated_{map_range}m_5k'
    gen_metrics_dir = os.path.join(gen_data_dir, 'metrics')
    os.makedirs(gen_metrics_dir, exist_ok=True)
    
    gen_graph_files = glob.glob(os.path.join(gen_data_dir, 'graph/*'))
    # gen_agent_files = glob.glob(os.path.join(gen_data_dir, 'agent/*'))
    # gen_track_files = glob.glob(os.path.join(gen_data_dir, 'track/*'))
    
    # Compute Map statistics using HDMapGen metrics
    # map_metrics.compute_map_stats(gen_graph_files, gen_metrics_dir, map_range=map_range, map_res=map_res) # 5000 samples about 45s to 1m30s
    
    ######################################### Compute Metrics #######################################
    
    # Compare & Compute Metrics
    gt_stats = np.load(os.path.join(gt_metrics_dir, 'stats.npy'))
    gt_degrees = np.load(os.path.join(gt_metrics_dir, 'degrees.npy'))
    gt_spectrum = np.load(os.path.join(gt_metrics_dir, 'spectrum.npy'))
    
    gen_stats = np.load(os.path.join(gen_metrics_dir, 'stats.npy'))
    gen_degrees = np.load(os.path.join(gen_metrics_dir, 'degrees.npy'))
    gen_spectrum = np.load(os.path.join(gen_metrics_dir, 'spectrum.npy'))
    
    map_metrics.compute_map_metrics(gt_stats, gt_degrees, gt_spectrum, gen_stats, gen_degrees, gen_spectrum)
    
    # map_metrics.compute_agent_metrics(gt_agent_files, gen_agent_files)