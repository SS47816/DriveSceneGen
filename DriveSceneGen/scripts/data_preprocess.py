import glob
import os
import argparse
import tensorflow as tf
import pickle
import numpy as np
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from multiprocessing import Pool
from DriveSceneGen.utils.datasets.waymo.data_utils import *
from DriveSceneGen.utils.datasets.waymo.waymo_types import *

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')

# Data process
class DataProcess(object):
    def __init__(self, files):

        self.data_files = files
    def build_map(self, map_features, dynamic_map_states):
        # parsed_data.map_features, parsed_data.dynamic_map_states
        # each scenario has a set of map_features
        self.traffic_signals = dynamic_map_states
        self.lane_polylines={}
        self.polylines=[]
        self.crosswalks={}
        self.lanes ={}
        self.crosswalks_p={}
        self.road_polylines={}
        self.speed_bump={}
        self.driveway={}
        self.stop_sign={}
        # static map features
        
        for cur_data in map_features:
            map_id = cur_data.id
            if cur_data.lane.ByteSize() > 0:
                self.lanes[map_id] = cur_data.lane
                data_type = lane_type[cur_data.lane.type]
                global_type = polyline_type[data_type]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0) 
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])    # direction vector
                if len(cur_polyline) > 1:
                    direction = wrap_to_pi(np.arctan2(cur_polyline[1:, 1]-cur_polyline[:-1, 1],cur_polyline[1:, 0]-cur_polyline[:-1, 0]))
                    direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
                else:
                    direction = np.array([0])[:, np.newaxis]
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:],direction), axis=-1)
                self.lane_polylines[map_id]=cur_polyline
                self.polylines.append(cur_polyline)
                
                
            elif cur_data.road_line.ByteSize() > 0 :
                data_type = road_line_type[cur_data.road_line.type]
                global_type = polyline_type[data_type]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0) 
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])  
                if len(cur_polyline) > 1:
                    direction = wrap_to_pi(np.arctan2(cur_polyline[1:, 1]-cur_polyline[:-1, 1],cur_polyline[1:, 0]-cur_polyline[:-1, 0]))
                    direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
                else:
                    direction = np.array([0])[:, np.newaxis]  
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:],direction), axis=-1)
                self.polylines.append(cur_polyline)
                self.road_polylines[map_id]=cur_polyline
                
            elif cur_data.road_edge.ByteSize() > 0 :
                data_type = road_edge_type[cur_data.road_edge.type]
                global_type = polyline_type[data_type]                
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0) 
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])   
                if len(cur_polyline) > 1:
                    direction = wrap_to_pi(np.arctan2(cur_polyline[1:, 1]-cur_polyline[:-1, 1],cur_polyline[1:, 0]-cur_polyline[:-1, 0]))
                    direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
                else:
                    direction = np.array([0])[:, np.newaxis] 
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:],direction), axis=-1)
                self.polylines.append(cur_polyline)
                self.road_polylines[map_id]=cur_polyline
                                 
            elif cur_data.stop_sign.ByteSize() > 0:
                point = cur_data.stop_sign.position
                global_type = polyline_type['TYPE_STOP_SIGN']
                cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type,0]).reshape(1, 8)
                self.polylines.append(cur_polyline)
                self.stop_sign[map_id] =cur_polyline
                
            elif cur_data.crosswalk.ByteSize() > 0:
                self.crosswalks[map_id] = cur_data.crosswalk
                global_type = polyline_type['TYPE_CROSSWALK']
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                if len(cur_polyline) > 1:
                    direction = wrap_to_pi(np.arctan2(cur_polyline[1:, 1]-cur_polyline[:-1, 1],cur_polyline[1:, 0]-cur_polyline[:-1, 0]))
                    direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
                else:
                    direction = np.array([0])[:, np.newaxis]
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:],direction), axis=-1)            
                self.polylines.append(cur_polyline)
                self.crosswalks_p[map_id] = cur_polyline
                
            elif cur_data.speed_bump.ByteSize() > 0:
                global_type = polyline_type['TYPE_SPEED_BUMP']
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                if len(cur_polyline) > 1:
                    direction = wrap_to_pi(np.arctan2(cur_polyline[1:, 1]-cur_polyline[:-1, 1],cur_polyline[1:, 0]-cur_polyline[:-1, 0]))
                    direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
                else:
                    direction = np.array([0])[:, np.newaxis]
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:],direction), axis=-1)            
                self.polylines.append(cur_polyline)
                self.speed_bump[map_id] =cur_polyline

                               
            elif cur_data.driveway.ByteSize() > 0:    
                global_type = polyline_type['TYPE_DRIVEWAY']
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.driveway.polygon], axis=0)
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                if len(cur_polyline) > 1:
                    direction = wrap_to_pi(np.arctan2(cur_polyline[1:, 1]-cur_polyline[:-1, 1],cur_polyline[1:, 0]-cur_polyline[:-1, 0]))
                    direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
                else:
                    direction = np.array([0])[:, np.newaxis]
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:],direction), axis=-1)      
                
                self.polylines.append(cur_polyline)
                self.driveway[map_id] = cur_polyline
               
                    
            else:
                raise TypeError      
        try:
            self.polylines = np.concatenate(self.polylines, axis=0).astype(np.float32)  # [n,8]  
        except:
            self.polylines = np.zeros((0, 8), dtype=np.float32)
            print('Empty polylines: ')
    def decode_tracks_from_proto(self,tracks):
        self.track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': [],
            'track_index':[]
        }
        for track_index,cur_data in enumerate(tracks):  # number of objects
            cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, wrap_to_pi(x.heading),
                                x.velocity_x, x.velocity_y, x.valid,cur_data.object_type], dtype=np.float32) for x in cur_data.states]
            cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp=91, 10)
            self.track_infos['object_id'].append(cur_data.id)
            self.track_infos['object_type'].append(object_type[cur_data.object_type])
            self.track_infos['trajs'].append(cur_traj)
            self.track_infos['track_index'].append(track_index)
        self.track_infos['trajs'] = np.stack(self.track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 10)
    
    def process_data(self, save_path, viz=False):  
        ret_info = []  
        for data_file in self.data_files:
            
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            for data in dataset:
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())   
                scenario_id = parsed_data.scenario_id   #  2fcc1e81956cdf14
                timestamps_seconds = list(parsed_data.timestamps_seconds)
                timestep = parsed_data.current_time_index
                sdc_id = parsed_data.sdc_track_index    # index `for ego vehicle
                track_to_predict = [cur_pred.track_index for cur_pred in parsed_data.tracks_to_predict]
                time_len = len(parsed_data.tracks[sdc_id].states) # time length
                self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)
                self.decode_tracks_from_proto(parsed_data.tracks)  
                info={
                }
                info['tracks_info'] =self.track_infos                      # (num_objects, num_timestamp, 10)
                info['scenario_id'] = scenario_id   
                ret_info.append(scenario_id)      
                info['lanes_info'] =self.lanes
                # info['dynamic_info'] = self.traffic_signals
                info['lane'] = self.lane_polylines
                info['crosswalk'] = self.crosswalks_p
                info['speed_bump'] = self.speed_bump
                info['drive_way'] = self.driveway
                info['stop_sign'] = self.stop_sign
                info['road_polylines'] =self.road_polylines
                info['sdc_track_index'] = sdc_id
                info['predict_list'] = track_to_predict      
                output_file = os.path.join(save_path, f'sample_{scenario_id}.pkl')
                with open(output_file, 'wb') as f:
                    pickle.dump(info, f)                    
                self.pbar.update(1)
            self.pbar.close()
            
        return ret_info


def multiprocessing(data_files):
    processor = DataProcess([data_files]) 
    ret_info=processor.process_data(save_path, viz=True)
    return ret_info

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing 1')
    parser.add_argument('--load_path',default="./data/raw", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="./data/preprocessed",type=str, help='path to save processed data')
    parser.add_argument('--use_multiprocessing', help='if use multiprocessing', default=True)
    
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    if args.use_multiprocessing:
        with Pool() as p:
            results = p.map(multiprocessing, data_files)
            ret_info = [item for sublist in results for item in sublist]
            test_filename = os.path.join('./data/preprocessed', 'processed_scenarios_20s.pkl')
            with open(test_filename, 'wb') as f:
                pickle.dump(ret_info, f)
    else:
        processor = DataProcess(data_files) 
        processor.process_data(save_path, viz=True)
        print('Done!')
