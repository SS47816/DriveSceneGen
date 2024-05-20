import matplotlib.pyplot as plt
import numpy as np

import torch
import io
from PIL import Image
from torchvision import transforms
from DriveSceneGen.utils.datasets.map_processing import segment_points_to_polylines
from DriveSceneGen.utils.datasets.map_processing import dxdy_normalization
from scipy import interpolate
from collections import defaultdict
import bisect

"""
# interpolate and padding the vector map 
# to tensor map that has the same size for all scenarios

#UNSOLVE:
# 1.filte polylines only in the retangle desired size area


# 1. interpolate the vector map
# 2. padding the vector map to the same size
# 3. save the vector map to tensor map
"""

def generate_desired_type_polylines_list(all_points, point_sampled_interval=1.0, 
                                          points_break_dist_thresh=1.0,
                                          desired_map_type_list=[2],
                                          filtering: bool=False, 
                                          filter_distance:int=40):
    """
    Args:
        all_points (num_points, 8): [x, y, z, dir_x, dir_y, dir_z, global_type, theta]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 8)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    too_less_polylines=False
    point_dim = all_points.shape[-1]
    sampled_points = all_points[::point_sampled_interval]
    
    polylines_list = segment_points_to_polylines(sampled_points, points_break_dist_thresh)
    desired_polylines_list=[]
    filtered_polylines_list=[]
    
    for k in range(len(polylines_list)):
        polyline=polylines_list[k]
        if len(polyline) <= 0:
            continue
        
        if polyline[0,6] in desired_map_type_list:
            desired_polylines_list.append(polyline)
            
            # filter points out of fields of view
            if filtering:
                filtered_polyline=polyline  
                delete_idxs=[]
                for i in range(len(polyline)):
                    # filter_distance=10
                    if np.abs(polyline[i,0])>filter_distance or np.abs(polyline[i,1])>filter_distance:
                        delete_idxs.append(i)
                        
                filtered_polyline=np.delete(polyline, delete_idxs,axis=0)  
                
                # if current polyline has no waypoint in the desired area  
                # do not add it to the polyline list 
                if filtered_polyline.shape[0]==0:
                   continue
                    
            filtered_polylines_list.append(filtered_polyline)  
        else:
            continue
        
    # if there is no polyline belongs to desired type
    # skip this scenario
    if len(polylines_list)==0:
        too_less_polylines=True
    if len(desired_polylines_list)==0:
       desired_polylines_list=polylines_list
       too_less_polylines=True
    
    # if there is no no polyline in desired area
    # skip this scenario
    # print(len(filtered_polylines_list))
    filtered_polylines_list=cut_polyline_for_end_point(filtered_polylines_list,filter_distance)
    # filtered_polylines_list=cal_curvature_and_cut(filtered_polylines_list)  
    
    filtered_polylines_list=create_path_from_polylines(filtered_polylines_list,filter_distance)
    filtered_polylines_graph=polylines_list_to_graph(filtered_polylines_list)
    return filtered_polylines_list, too_less_polylines

def cut_polyline_for_end_point(polylines_list,filter_distance:int=40):
    """
    1. classfy which end point potentially needs to be used as cut reference
        for a polyline end point, 
            1. if it is not at the edge  
            2. also it has no connectors polyline
        it means this end point position has a long polyline that need to be cutted
    
    2. find this end point neighbor points, their belongings and cutting index
        how to find the neighbor points?
            1. build a dict of points, with key: polyline index + point index
            2. build a distance matrix of all points
    3. cut the polyline into multiple polylines, at the cutting index
    
    """
    # build a dict of polylines
    polylines_dict={}
    key=np.arange(len(polylines_list))
    for i in range(len(polylines_list)):
        polylines_dict[key[i]]=polylines_list[i]
    
      
    split_dict=defaultdict(list)# key is target poly index, value is split ptrs list   
    delete_keys=[]
    polylines_splited=[]
    for k, poly_k in  polylines_dict.items():
        ref_points_idx=[0,-1]
        
        for start_end_i in ref_points_idx:
            # if the endpoint is not at the edge
            if (abs(abs(poly_k[start_end_i,0])-abs(filter_distance))>1 and abs(abs(poly_k[start_end_i,1])-abs(filter_distance)))>1:
                    #  or (abs(abs(poly_k[0,0])-abs(filter_distance))>1 and abs(abs(poly_k[0,1])-abs(filter_distance))>1):
                
                # if the end point also has no successor polyline
                # successor definition, successor start point should close to 
                # current polyline end point 
                connectors=[]
                for s,poly_conn in polylines_dict.items():
                    counter_i=[i for i in ref_points_idx if i!=start_end_i][0]
                    if np.linalg.norm(poly_k[start_end_i,:2]-poly_conn[counter_i,:2])<2:
                        # or np.linalg.norm(poly_k[0,:2]-poly_conn[-1,:2])<1:
                        connectors.append(s)
                        
                if len(connectors)==0:
                    
                    # calculate current end point distance to all points    
                    for poly_cc_i, poly_cut_candidate in polylines_dict.items():
                        ptr_candidates_list={} # key is distance, value is the ptr_index
                        distances_list=[]
                        for ptr_index in range(len(poly_cut_candidate)):
                            ptr=poly_cut_candidate[ptr_index]
                            
                            ptr_distance = abs(np.linalg.norm(poly_k[start_end_i,:2]-ptr[:2]))
                            
                            # ptr_distance_start<=1.5 or 
                            
                            if ptr_distance<=1.5 and poly_cc_i !=k and len(poly_cut_candidate)>=4:
                                ptr_candidates_list[ptr_distance]=ptr_index
                             
                                bisect.insort_left(distances_list,ptr_distance)
                                # save the split target index and split point 
                                # in a dict
            
                        # split point is the closet one
                        while len(distances_list)>0:
                            split_index=ptr_candidates_list[distances_list[0]]     
                            # if the split point is not close to 
                            # original endpoints
                            if split_index> 3 and split_index < (len(poly_cut_candidate)-3):
                                split_dict[poly_cc_i].append(split_index)
            
                                break
                            
                            # if current split does not work
                            # delete the closet one and continue
                            else: 
                                del distances_list[0]
                                break    
                                
            
    for split_target_key,split_ptr in split_dict.items():
        
        split_target=polylines_dict[split_target_key]
        split_ptr=list(set(split_ptr))
        
        split_result=np.split(split_target,split_ptr,axis=0)
        
        delete_keys.append(split_target_key)
        polylines_splited.extend(split_result)
                   
                        
    # replace the original polyline with splited polylines
    delete_keys=list(set(delete_keys))
    if len(delete_keys)>0:
        sorted_delete_keys=sorted(delete_keys)
        sorted_delete_keys.reverse()
        for delete_key in sorted_delete_keys:
            del polylines_list[delete_key]
            
        
        polylines_splited=[poly for poly in polylines_splited if len(poly)>=3]
        polylines_list.extend(polylines_splited)
        
    return polylines_list


def cal_curvature_and_cut(polylines_list: np.ndarray):
    """
    1. input is a list of polylines
    2. extract the velocity of each waypoint
    3. speed is a constant 0.5
    4. calculate the curvature of each waypoint
    """
    cutted_polylines=[]
    for polyline in polylines_list:
        if len(polyline)>=5:
            x_t=polyline[:,3]
            y_t=polyline[:,4]
            speed=0.5
    
            
            # calculate the curvature
            ss_t = np.gradient(speed)
            xx_t = np.gradient(x_t)
            yy_t = np.gradient(y_t)

            curvature = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
            d_curvature = np.gradient(curvature)
            dd_curvature = np.gradient(d_curvature)
            cut_index=[]
            
            for index in range(len(d_curvature)):
                if dd_curvature[index]>50e-4:
                    cut_index.append(index)
            
            # single polyline may be cutted into multiple polylines
            # polylines_splited is a list of cutted polylines       
            polylines_splited=np.split(polyline,cut_index,axis=0)
            
            for polyline in polylines_splited:
                if len(polyline)>0:
                    cutted_polylines.append(polyline) 
        else:
            cutted_polylines.append(polyline)
        
    return cutted_polylines
        
        
def create_path_from_polylines(polylines_list: np.ndarray,filter_distance:int=40): 
    """
    1. find root polylines(polyline that starts at the edge of the map)
    2. find leaf polylines(polyline that ends at the edge of the map)
    3. find the path from root to leaf
        1. connect the polyline which start point is 
            concident with the end point of current polyline
    
    """ 
    # build a dict of polylines
    polylines_dict={}
    key=np.arange(len(polylines_list))
    for i in range(len(polylines_list)):
        polylines_dict[key[i]]=polylines_list[i]
        
        
    root_vertices={}
    leaf_vertices={}
    for k, poly_k in  polylines_dict.items():
  
        if abs(abs(poly_k[0,0])-abs(filter_distance))<0.5 or abs(abs(poly_k[0,1])-abs(filter_distance))<0.5:
            root_vertices[k]=poly_k
            
        if abs(abs(poly_k[-1,0])-abs(filter_distance))<0.5 or abs(abs(poly_k[-1,1])-abs(filter_distance))<0.5:
            leaf_vertices[k]=poly_k
    
    # Initialize visited dictionary
    visited = {k: False for k in polylines_dict.keys()}
    
    # Iterate through root polylines and search for paths to leaf polylines
    final_paths_keys=[]
    for root_key in root_vertices.keys():
        path = []
        depth_first_search(polylines_dict, root_key, leaf_vertices,visited, path,final_paths_keys)
    
    final_paths =[]

    for single_path_keys in final_paths_keys:
        single_path=np.concatenate([polylines_dict[i] for i in single_path_keys],axis= 0)
        final_paths.append(single_path)
    return final_paths
     
def depth_first_search(polylines_dict, current_polyline_key, leaf_vertices,visited, path,paths_list):
    visited[current_polyline_key] = True
    path.append(current_polyline_key)
    
    if current_polyline_key in leaf_vertices:
        # Found a path from root to leaf, add it to final_paths_keys
        # print("Found a path:", path)
        paths_list.append(path.copy())
    else:
        current_polyline = polylines_dict[current_polyline_key]
        current_endpoint = current_polyline[-1,:2]  # The last point of the current polyline
        current_direction = current_polyline[-1,4]/current_polyline[-1,3]  # The last point of the current polyline
        
        
        condidates_keys = []
        for next_polyline_key, next_polyline in polylines_dict.items():
            if not visited[next_polyline_key]:
                # Check if the current polyline's endpoint matches the next polyline's start point
                # Check if there are multiple polylines that match the current polyline's endpoint
                next_polyline_direction = next_polyline[0,4]/next_polyline[0,3]
                if  np.linalg.norm(current_endpoint-next_polyline[0,:2])<=0.5:# and abs(current_direction-next_polyline_direction)<0.314:
                    condidates_keys.append(next_polyline_key)
                
        condidates_keys= list(set(condidates_keys))
        for candidate in condidates_keys:
            depth_first_search(polylines_dict, candidate, leaf_vertices,visited, path,paths_list)
    
    # Backtrack
    visited[current_polyline_key] = False
    path.pop()

def polylines_list_to_graph(polylines_list):
    """
        1. build a dict of all points as vertices  
            key: polyline_index + point_index
            value: point coordinate 
            
        2.  path will be transferred a list of point_dict keys
        
        3. check redundant of points
            build a list called redundant_list to record the redundant points index
            in the list, each element is a list of redundant points index
        
        4. the unique index of redundant points will be the first element of the list
        5. in the path list, replace the redundant points index with the unique index

    """
    # build a dict of polylines
    polylines_dict={}
    key=np.arange(len(polylines_list))
    for i in range(len(polylines_list)):
        polylines_dict[key[i]]=polylines_list[i]
        # plt.plot(polylines_list[i][:,0],polylines_list[i][:,1])
    points_dict={}
    polys_key_list=[] # list of polys array(array of point keys)
    for poly_i, poly in polylines_dict.items():
        poly_key=[]
        for ptr_i in range(len(poly)):
            ptr_key=str(poly_i)+'_'+str(ptr_i)
            points_dict[ptr_key]=poly[ptr_i]
            poly_key.append(ptr_key)
        poly_key=np.array(poly_key)
        polys_key_list.append(poly_key)                   
    
    # check redundant points
    # pos_key_dict is a dict to record points positions 
    # and its corresponding keys(both unique and redundant) 
    
    # pos's value is a list of keys
    pos_key_dict =defaultdict(list)
    
    for key, pos in points_dict.items():
        pos_str=str(pos[0])+'_'+str(pos[1]) # pos x and y
        
        if pos_str in pos_key_dict:
            pos_key_dict[pos_str].append(key)
            
        else:
            pos_key_dict[pos_str].append(key) 
     
    # delete positions corresponding to unique key from pos_key_dict
    unique_pos=[]        
    for pos_str, keys in pos_key_dict.items():
        if len(keys)==1:
            unique_pos.append(pos_str)    
   
    for pos_str in unique_pos:
        del pos_key_dict[pos_str] 
        
        
    # replace redundant keys with the first key in the list,
    # make the change in the polylines_key_list
    points_dict_del_keys=[]
    for poly in polys_key_list:
        for i in range(poly.shape[0]):
            ptr_key=poly[i]
            for pos_str, redundant_keys in pos_key_dict.items():
                if ptr_key in redundant_keys and ptr_key!=redundant_keys[0]:
                    poly[i]=redundant_keys[0]
                    
                    points_dict_del_keys.append(ptr_key)
                  
    # clean the redundant keys in the points list
    for key in points_dict_del_keys:
        del points_dict[key]   
    
    paths_vertices=points_dict
    paths_edges=polys_key_list
    
    # plot the vertices
    vertices=np.array(list(paths_vertices.values()))
    # plt.scatter(vertices[:,0],vertices[:,1])
    
    # plt.show()
    
    path_graph=[paths_vertices,paths_edges]            
    return path_graph

  
                
def polyline_list_interpolation(polylines_list, desired_polyline_row_length=128,average_distance_interpolate=False):
    delete_keys = []
    
    polylines_interpolated_dict = {}
    polylines_interpolated_dict_decrease_channel = {}
    
    polylines_interpolated_list =[]
    polylines_interpolated_list_decrease_channel =[]
    
    for key,polyline in enumerate(polylines_list):
              
        if len(polyline) < 3:
            # polylines_list.remove(key)
            delete_keys.append(key)
            continue

        # Linear length along the line:
        distance_xyz = np.cumsum(
            np.sqrt(np.sum(np.diff(polyline[:, 0:3], axis=0)**2, axis=1)))
        distance_xyz = np.insert(distance_xyz, 0, 0)/distance_xyz[-1]

        # distance for point(dx,dy,dz)
        distance_dxdydz = np.cumsum(
            np.sqrt(np.sum(np.diff(polyline[:, 3:6], axis=0)**2, axis=1)))
        distance_dxdydz = np.insert(distance_dxdydz, 0, 0)/distance_dxdydz[-1]

        # distance for theta
        # row_indice = np.arange(0, polyline.shape[0])
        # theta_2d_points = np.array([row_indice, polyline[:, 6]]).T

        # distance_theta = np.cumsum(
            # np.sqrt(np.sum(np.diff(theta_2d_points, axis=0)**2, axis=1)))
        # distance_theta = np.insert(distance_theta, 0, 0)/distance_theta[-1]

        # generate new x 
        # interpolate for fix number of points
        if average_distance_interpolate==False:
            x_new = np.linspace(distance_xyz[0], distance_xyz[-1], num=desired_polyline_row_length)
            dx_new = np.linspace(distance_dxdydz[0], distance_dxdydz[-1], num=desired_polyline_row_length)
            # theta_new = np.linspace(distance_theta[0], distance_theta[-1], num=desired_polyline_row_length)
        else:
            # interpolate for fix average distance
            x_new = np.linspace(distance_xyz[0], distance_xyz[-1], num=int(distance_xyz[-1]/0.5))
            dx_new = np.linspace(distance_dxdydz[0], distance_dxdydz[-1], num=int(distance_dxdydz[-1]/0.5))
        
        # avoid inf slope in interpolation
        for i in range(len(polyline)):
            # if dx is too small, add 0.01 to avoid inf slope
            if polyline[i,3]<=0.1:
                polyline[i,0]= polyline[i,0]+0.1
        # start interpolation
        
        interpolator_xyz = interpolate.interp1d(
            distance_xyz, polyline[:, 0:3], axis=0)
        interpolator_dxdydz = interpolate.interp1d(
            distance_dxdydz, polyline[:, 3:6], axis=0)
        # interpolator_theta = interpolate.interp1d(
            # distance_theta, theta_2d_points, axis=0)

        xyz_interpolated = interpolator_xyz(x_new)
        dxdydz_interpolated = interpolator_dxdydz(dx_new)
        # theta_interpolated = interpolator_theta(theta_new)

        # polyline type
        polyline_type = np.ones([xyz_interpolated.shape[0], 1])*polyline[0, 6]

        # assemble polyline
        polyline_interpolated = np.concatenate(
            (xyz_interpolated, dxdydz_interpolated, polyline_type),axis=1)#, theta_interpolated[:, 1].reshape([-1, 1])), axis=1)

        # assemble polyline with only [x,,y,dx,dy,type]
        polyline_decrease_channel = np.concatenate(
            (xyz_interpolated[:, 0:2], dxdydz_interpolated[:, 0:2],polyline_type), axis=1)
        # polyline_interpolated[:,6]=polyline_type[:,0]

        # write back to polylines
        polylines_interpolated_dict[key] = polyline_interpolated
        # polyline_interpolated[:,6]=polyline_type[:,0]

        # write back to polylines
        polylines_interpolated_dict_decrease_channel[key] = polyline_decrease_channel
        polylines_interpolated_list_decrease_channel.append(polyline_decrease_channel)
        polylines_interpolated_list.append(polyline_interpolated)
        
        
    for delete_key in delete_keys:
        if delete_key in polylines_interpolated_dict:
            del polylines_interpolated_dict[delete_key]
            del polylines_interpolated_dict_decrease_channel[delete_key]
            polylines_interpolated_list_decrease_channel.remove(delete_key)
            polylines_interpolated_list.remove(delete_key)
           
    return polylines_interpolated_list
 

def vector_to_same_size_tensor(
    scenario_info: dict, 
    des_column_size: int=256,
    des_row_size: int=256,
    img_res: tuple = (512, 512), 
    dpi: int = 200, 
    map_range: float = 100.0,
    average_distance_interpolate:bool=False,
    ) -> None:
    """
    this function will
    1. call `generate_desired_type_polylines_list` to get the desired polylines list
        with the desired polyline type, 
        center is ego vehicle,
        and filter polyline in desired area.
    
    2. call `polyline_list_interpolation` to interpolate single polyline to the same size
    
    3. padding the polylines list to the same length
    """

    too_less_polylines=False
    
    # 2w dataset
    # all_points = scenario_info['all_polylines']  # [M, 7] [x,y,z,ori_x,ori_y,ori_z,type,theta]
    # obj_trajs_full = scenario_info['all_agent']
    # sdc_track_index = scenario_info['sdc_track_index']
    
    # #7w dataset
    all_polylines=scenario_info['lane']
    all_points=np.array([v[:,:7] for v in all_polylines.values()],dtype='object')
    all_points=np.vstack(all_points)
    sdc_track_index = scenario_info['sdc_track_index']
    obj_trajs_full = scenario_info['tracks_info']['trajs']
    
    ego_position = np.array(obj_trajs_full[sdc_track_index][10][:2])  # [x,y]
    ego_theta = np.array(obj_trajs_full[sdc_track_index][10][6])  # [x,y]
    
    # Rotate(optional) and coordinate transformation 
    # the map following the ego vehicle's pose
    all_points[:, :2]=all_points[:, :2]-ego_position
    # extract centerlines from all polylines
    desired_map_type_list=[2]
    filtered_polylines_list,too_less_polylines=generate_desired_type_polylines_list(
                                                all_points, point_sampled_interval=1,  
                                                points_break_dist_thresh=1.0,
                                                desired_map_type_list=desired_map_type_list,
                                                filtering=True,filter_distance=map_range)
    
    if too_less_polylines==True:
        polylines_with_mask_tensor=torch.zeros((des_row_size,des_column_size,8))
    
    else:
        # interpolated the polylines to the same size
        interpolated_polylines_list=polyline_list_interpolation(filtered_polylines_list, 
                                                                desired_polyline_row_length=des_column_size,
                                                                average_distance_interpolate=average_distance_interpolate)
        
        # interpolated_polylines_list=filtered_polylines_list
            
            
        ## padiing the polylines list to the same size
        if len(interpolated_polylines_list)>0:
            if len(interpolated_polylines_list) >=des_row_size:
                interpolated_polylines_list_padding=interpolated_polylines_list[:des_row_size]
                polylines_tensor=torch.Tensor(np.stack(interpolated_polylines_list))
                
                
            elif len(interpolated_polylines_list)<des_row_size:
                padding_row=np.ones((des_column_size,interpolated_polylines_list[0].shape[1]))*0.2
                padding=[padding_row]*(des_row_size-len(interpolated_polylines_list))
                
                interpolated_polylines_list_padding=interpolated_polylines_list+padding
                
                # print(len(polylines_list))
                # create padding indicating masks                               
            all_masks=np.full((des_row_size,des_column_size),False)#0.2,dtype=np.float32)
            all_masks[:len(interpolated_polylines_list),:]=True

            polylines_tensor=torch.Tensor(np.stack(interpolated_polylines_list_padding))
            mask_tensor=torch.Tensor(np.stack(all_masks))
            
            #     print(polylines_tensor.shape)
            polylines_with_mask_tensor=torch.cat((polylines_tensor,mask_tensor.unsqueeze(2)),dim=2)
        else:
            polylines_with_mask_tensor=torch.zeros((des_row_size,des_column_size,8))
            too_less_polylines=True
    return polylines_with_mask_tensor,too_less_polylines

def tensor_back_to_list(polyines_with_mask_tensor):
    polylines_list=[]
    masks=np.array(polyines_with_mask_tensor[:,:,-1],dtype=bool)
    
    for i in range(polyines_with_mask_tensor.shape[0]):
       polyline=np.array(polyines_with_mask_tensor[i,:,0:8])
       polylines_list.append(polyline)
       
    return polylines_list, masks


## extract each pixel spatial coordinate from the fig_tensor
def extract_spatial_coordinates(img):
    size=img.shape[1] #original plot size is 512x512, but we will resize to 256
    coord_bias=size/2
    coord_x=(np.linspace(-coord_bias,coord_bias,size)/size).reshape(1,-1) # 256
    coord_x=np.repeat(coord_x,size,axis=0) # 256x256, each row is the same
    
    coord_y=(np.linspace(-coord_bias,coord_bias,size)/size).reshape(-1,1) # 256
    coord_y=np.repeat(coord_y,size,axis=1) # 256x256, each row is the same
    
    
    coordinates=np.stack((coord_x,coord_y),axis=0) # 2x256x256
    return coordinates

def histogram_find_from_dataset(img):
    dx = img[0,:,:].to('cpu').numpy()
    dy = img[1,:,:].to('cpu').numpy()

    # flatten dx and dy to 1D array
    dx_gray_values = dx.flatten()
    dy_gray_values = dy.flatten()

    # compute the histogram
    dx_histogram, dxbins = np.histogram(dx_gray_values, bins=256, range=(0, 1))
    dy_histogram, dybins = np.histogram(dy_gray_values, bins=256, range=(0, 1))

    # find the gray value with the most pixels
    dx_max_gray_value = dxbins[np.argmax(dx_histogram)]
    dy_max_gray_value = dybins[np.argmax(dy_histogram)]

    # # plot the histogram
    # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    # ax0.bar(dxbins[:-1], dx_histogram, width=1/256, align='edge', color='gray')
    # ax0.set_title('dx')

    # ax1.bar(dybins[:-1], dy_histogram, width=1/256, align='edge', color='gray')
    # ax1.set_title('dy')

    # ax2.imshow(dx, cmap='gray')
    # ax3.imshow(dy, cmap='gray')

    # plt.tight_layout()

    # # print the gray value with the most pixels
    # print('Max gray value in dx:', dx_max_gray_value)
    # print('Max gray value in dy:', dy_max_gray_value)

    # plt.show()

    return dx_max_gray_value, dy_max_gray_value

########### start ploting ################
# checking the feasibility of the map
def plotting_vector_map(polylines_list, all_masks, img_res: tuple = (512, 512), 
                        dpi: int = 200, map_range: float = 100.0,dx_dy_exist: bool=False,
                        des_column_size:int=256,des_row_size:int=256) -> None:
    
    if dx_dy_exist==True:
                 # normalized polyline dx dy for visualization
                 polylines_list=dxdy_normalization(polylines_list,des_row_size)
                 normalized_polyline={}
                 
    fig, ax = plt.subplots(1, 1, figsize=(
        img_res[0]/dpi, img_res[1]/dpi), dpi=dpi)
    
    for polyline, masks in zip(polylines_list, all_masks):
        map_type = polyline[0, -1]
        if dx_dy_exist==True:
            normalized_polyline=polyline
        
            # 
            normalized_polyline[masks, 3:5]=np.squeeze(polyline[masks][:, 3:5])

        
            colors_array_dx=np.zeros((normalized_polyline[masks].shape[0],3))
            colors_array_dy=np.zeros((normalized_polyline[masks].shape[0],3))
            colors_array=np.zeros((normalized_polyline[masks].shape[0],3))
            # print the waypoint color as the value of dx dy
            for index, waypoint in enumerate(normalized_polyline[masks]):
                color_waypoint_dx=np.array([waypoint[3],0.0,0.0])
                color_waypoint_dy=np.array([0.0,waypoint[4],0.0])
                color_waypoint = np.array([waypoint[3], waypoint[4], 0.0])
            
                colors_array_dx[index]=color_waypoint_dx
                colors_array_dy[index]=color_waypoint_dy
                colors_array[index]=color_waypoint
            
        # if map_type > 1 and map_type < 3:
            #ax.scatter(polyline[masks][:, 0], polyline[masks][:, 1], c=colors_array_dx, s=1)
            c=colors_array
            gap=1
            # ax.scatter(polyline[masks][:, 0], polyline[masks][:, 1], c=colors_array, s=0.05, marker='.')
            polyline_len=polyline[masks].shape[0]
            for i in range(polyline_len-1):
                ax.plot(polyline[masks][i*gap:(i+2)*gap:gap, 0], polyline[masks][i*gap:(i+2)*gap:gap, 1],linewidth=1.5, color=colors_array[i*gap,:])
            
        else:
            ax.scatter(polyline[masks][1:, 0], polyline[masks][1:, 1], color= np.array([0.0,0.0,0.0]), s=1, marker='D')   
            # ax.plot(polyline[masks][1:, 0], polyline[masks][1:, 1], color= np.array([0.0,0.0,0.0]))
        # plot control points
        # ax.scatter(polyline[masks][0, 0], polyline[masks][0, 1], color= np.array([0.9,0.0,0.0]), s=0.2, marker='o')   
      
    # ax.axis('off')            
    ax.set_facecolor(np.array([0.5,0.5,0.5]))
    # ax.set_aspect('equal')
    ax.axis('equal')
    # 
    map_range=map_range-1
    ax.set(xlim=(-map_range,  map_range), ylim=(-map_range, map_range))
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()
    # Save the figure to a buffer
    buffer_scatter = io.BytesIO()
    plt.savefig(buffer_scatter, format="png")
    plt.close()
    
    # read from the buffer
    buffer_scatter.seek(0)
    image_scatter = Image.open(buffer_scatter)
    image_scatter = image_scatter.convert('RGB')
    
    # transfer the image to array, then to tensor
    transform=transforms.ToTensor()
    fig_2_tensor=transform(image_scatter)
    fig_2_tensor=fig_2_tensor[:2,:,:]
    
    
    # print(fig_2_tensor.shape)
    # fig_tensor=torch.cat((fig_1_tensor,fig_2_tensor),dim=0)
    # print(fig_tensor.shape)
    
    # resize to 256x256
    transform=transforms.Resize((256, 256))#,antialias=False)
    fig_2_tensor=transform(fig_2_tensor)
    
    # add coordinates to the tensor
    coordinates=extract_spatial_coordinates(fig_2_tensor)
    coordinates_tensor=torch.tensor(coordinates)
    # print(coordinates_tensor.shape)
    fig_coord_tensor=torch.cat((coordinates_tensor,fig_2_tensor),dim=0)  # x,y,fig_2_tensor
    return fig_coord_tensor
