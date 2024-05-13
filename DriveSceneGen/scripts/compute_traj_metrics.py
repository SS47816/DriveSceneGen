import os
import glob
import random
import logging

import numpy as np

from DriveSceneGen.utils.io import *
from DriveSceneGen.vectorization.evaluation import map_metrics

logger = get_logger('compute_map_metrics', logging.WARNING)


if __name__ == "__main__":
    n_proc = 1  # Numer of available processors
    
    map_range = 80
    map_res = 256
    
    ######################################### Load Data #######################################

    # Ground Truth Data
    gt_data_dir = '/media/shuo/Cappuccino/DriveSceneGen/waymo1.2'
    # gt_agent_files = glob.glob(os.path.join(gt_data_dir, 'agent/*'))
    gt_metrics_dir = os.path.join(gt_data_dir, 'metrics')
    os.makedirs(gt_metrics_dir, exist_ok=True)
    
    if not os.path.exists(get_cache_name(gt_data_dir, 'track')):
        print('Caching track filenames...')
        cache_all_filenames(gt_data_dir, 'track')
    gt_track_files = get_all_filenames(gt_data_dir, 'track')
    gt_track_files = random.sample(gt_track_files, 4785)
    
    # Generated Data
    gen_data_dir = f'/media/shuo/Cappuccino/DriveSceneGen/generated_{map_range}m_5k'
    gen_metrics_dir = os.path.join(gen_data_dir, 'metrics')
    os.makedirs(gen_metrics_dir, exist_ok=True)
    gen_agent_files = glob.glob(os.path.join(gen_data_dir, 'agent/*'))
    # gen_track_files = glob.glob(os.path.join(gen_data_dir, 'track/*'))
    if not os.path.exists(get_cache_name(gen_data_dir, 'track')):
        print('Caching track filenames...')
        cache_all_filenames(gen_data_dir, 'track')
    gen_track_files = get_all_filenames(gen_data_dir, 'track')
        
    ######################################### Compute Metrics #######################################
    
    # gt_track_files = glob.glob(os.path.join(gt_data_dir, 'track/*'))
    # gt_track_files = random.sample(gt_track_files, 5000)
    # map_metrics.compute_track_metrics(gt_track_files, gt_track_files)
    
    map_metrics.compute_vehicle_placement_metrics(gt_track_files, gen_agent_files, gt_metrics_dir, gen_metrics_dir)