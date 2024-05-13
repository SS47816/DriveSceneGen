import glob
import os
import argparse
# import multiprocessing

import tensorflow as tf
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from waymo_open_dataset.protos import scenario_pb2


from multiprocessing import Pool
from DriveSceneGen.utils.datasets.waymo.data_utils import *
from DriveSceneGen.utils.datasets.waymo.waymo_types import *

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')

# Data process
class DataProcessor(object):
    def __init__(self, files):
        self.data_files = files
        
    def build_map(self, map_features, dynamic_map_states):
        # parsed_data.map_features, parsed_data.dynamic_map_states
        # each scenario has a set of map_features
        self.traffic_signals = dynamic_map_states
        self.lane_polylines = {}
        self.polylines = []
        self.crosswalks = {}
        self.lanes = {}
        self.crosswalks_p = {}
        self.road_polylines = {}
        self.speed_bump = {}
        self.driveway = {}
        self.stop_sign = {}
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
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:], direction), axis=-1)
                self.lane_polylines[map_id]=cur_polyline[:, :7]
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
    
    
    def build_graph(self, centerlines: dict, plot: bool = False) -> nx.Graph:
        graph = nx.Graph()
        
        edges = []
        nodes = []
        for centerline in list(centerlines.values()):
            if centerline.shape[0] <= 1:
                continue

            dx = np.diff(centerline[:, 0])
            dy = np.diff(centerline[:, 1])
            ds = np.hypot(dx, dy)
            s = np.cumsum(ds)
            path = list(zip(centerline.T[0], centerline.T[1]))
            n1 = path[0]
            n2 = path[-1]
            
            n1_yaw = np.arctan2(dy[0], dx[0])
            n2_yaw = np.arctan2(dy[-1], dx[-1])
            
            edges.append((n1, n2, {'path': path, 'dist': s[-1]}))
            nodes.append((n1, {'yaw': n1_yaw, 'type': 'exit'}))
            nodes.append((n2, {'yaw': n2_yaw, 'type': 'exit'}))
            
        graph.add_edges_from(edges)
        graph.add_nodes_from(nodes)

        if plot:
            fig, ax = plt.subplots()
            for n1, n2 in list(graph.edges):
                e = graph[n1][n2]
                points = np.array(e['path'])
                # print(f"n1_yaw = {graph.nodes[n1]['yaw']}, n2_yaw = {graph.nodes[n2]['yaw']}, dist = {e['dist']}")
                ax.plot(points[:, 0], points[:, 1])
            plt.show()
        
        return graph
    
    
    def process_data(self, save_path: str, viz: bool = False):  
        ret_info = []
        count = -1
        
        for index, data_file in enumerate(self.data_files):
            if count > 5000:
                break
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}") 

            for data in dataset:
                count += 1
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())   
                # scenario_id = parsed_data.scenario_id   #  2fcc1e81956cdf14
                scenario_id = count
                timestamps_seconds = list(parsed_data.timestamps_seconds)
                timestep = parsed_data.current_time_index
                sdc_id = parsed_data.sdc_track_index    # index `for ego vehicle
                track_to_predict = [cur_pred.track_index for cur_pred in parsed_data.tracks_to_predict]
                time_len = len(parsed_data.tracks[sdc_id].states) # time length
                
                self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)
                self.decode_tracks_from_proto(parsed_data.tracks)  
                
                info = {}
                info['scenario_id'] = scenario_id
                # info['lanes_info'] = self.lanes
                # info['dynamic_info'] = self.traffic_signals
                # info['lane'] = self.lane_polylines # centerlines
                # info['road_polylines'] = self.road_polylines # include both road markings and road boundaries
                # info['crosswalk'] = self.crosswalks_p
                # info['speed_bump'] = self.speed_bump
                # info['drive_way'] = self.driveway
                # info['stop_sign'] = self.stop_sign
                
                info['sdc_track_index'] = sdc_id
                info['tracks_info'] = self.track_infos # (num_objects, num_timestamp, 10)
                info['predict_list'] = track_to_predict
                
                info['lane'] = list(self.lane_polylines.values()) # list[np.ndarray] [M, [P, F=7]]
                info['all_agent'] = np.array(list(self.track_infos['trajs']))[:, :, :10] # np.ndarray [N, T=91, F=10]
                
                # print(f"all_polylines: {info['all_polylines'].shape}")
                # print(f"all_agent: {info['all_agent'].shape}")
                
                try:
                    # graph = self.build_graph(self.lane_polylines, plot=False)
                    # graph_file = os.path.join(save_path, 'graph', f'{scenario_id}_graph.pickle')
                    # with open(graph_file, 'wb') as f:
                    #     pickle.dump(graph, f)
                    
                    # # print(f"tracks_shape = {self.track_infos['trajs'].shape}")
                    # track_file = os.path.join(save_path, 'track', f'{scenario_id}_track.pickle')
                    # with open(track_file, 'wb') as f:
                    #     pickle.dump(self.track_infos, f)
                        
                    track_file = os.path.join(save_path, 'scenario', f'{scenario_id}.pkl')
                    with open(track_file, 'wb') as f:
                        pickle.dump(info, f)
                    
                except Exception as e:
                    print(f"At {scenario_id}, {e}")
                    self.pbar.update(1)
                    continue
                
                self.pbar.update(1)
                
            self.pbar.close()
            
        return ret_info


def multiprocessing(data_files: list, save_path: str, viz: bool = False):
    processor = DataProcessor([data_files]) 
    ret_info = processor.process_data(save_path, viz=True)
    return ret_info


# def multiprocessing_func(data_files: list, save_path: str, n_proc: int, proc_id: int):
#     processor = DataProcessor([data_files]) 
#     ret_info = processor.process_data(save_path, viz=True)
#     return 


# def chunks(input, n):
#     """Yields successive n-sized chunks of input"""
#     for i in range(0, len(input), n):
#         yield input[i : i + n]
        

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--load_path', default="/media/shuo/Chocolate/Waymo1.2/training", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="/media/shuo/Cappuccino/DriveSceneGen/waymo1.2", type=str, help='path to save processed data')
    parser.add_argument('--use_multiprocessing', help='if use multiprocessing', default=False)
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path + '/*')
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    if args.use_multiprocessing:
        with Pool() as p:
            results = p.map(multiprocessing, data_files, save_path)
            ret_info = [item for sublist in results for item in sublist]
            test_filename = os.path.join('/media/shuo/Dataset/waymo1.2/', 'processed_scenarios_20s.pkl')
            with open(test_filename, 'wb') as f:
                pickle.dump(ret_info, f)
    else:
        processor = DataProcessor(data_files) 
        processor.process_data(save_path, viz=True)
        print('Done!')
    
    # n_proc = 4
    
    # ## split the input files into n_proc chunks
    # chunked_files = list(chunks(data_files, int(len(data_files) / n_proc) + 1))

    # # Initialize the parallel processes list
    # processes = []
    # for proc_id in np.arange(n_proc):
    #     """Execute the target function on the n_proc target processors using the splitted input"""
    #     p = multiprocessing.Process(
    #         target=multiprocessing_func,
    #         args=(chunked_files[proc_id], save_path, n_proc, proc_id),
    #     )
    #     processes.append(p)
    #     p.start()
    # for process in processes:
    #     process.join()

    # print(f"Process finished!!!, results saved to: {save_path}")

