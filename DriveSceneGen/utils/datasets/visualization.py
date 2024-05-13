import argparse
import glob
import pickle
import numpy as np
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
from torchvision import transforms
from PIL import Image
from shapely.geometry import LineString, Polygon,MultiLineString
from shapely.affinity import rotate


def polygon_completion(polygon):
    polyline_x = []
    polyline_y = []

    for i in range(len(polygon)):
        if i+1 < len(polygon):
            next = i+1
        else:
            next = 0
        dist_x = polygon[next, 0] - polygon[i, 0]
        dist_y = polygon[next, 1] - polygon[i, 1]
        dist = np.linalg.norm([dist_x, dist_y])
        interp_num = np.ceil(dist) * 2
        interp_index = np.arange(2 + interp_num)
        point_x = np.interp(
            interp_index, [0, interp_index[-1]], [polygon[i, 0], polygon[next, 0]]).tolist()
        point_y = np.interp(
            interp_index, [0, interp_index[-1]], [polygon[i, 1], polygon[next, 1]]).tolist()
        polyline_x.extend(point_x[:-1])
        polyline_y.extend(point_y[:-1])

    return np.array([polyline_x, polyline_y]).T


def plot_static_map(scenario_info: dict) -> None:
    ax = plt.gca()
    # Unpack map information
    # all_polylines = scenario_info['all_polylines']   # [N_l,7] [x,y,z,ori_x,ori_y,ori_z,type,theta]
    lane_polylines = scenario_info['lane']
    road_polylines = scenario_info['road_polylines']
    if 'crosswalk' in scenario_info.keys():
        crosswalks_p = scenario_info['crosswalk']
    if 'speed_bump' in scenario_info.keys():
        speed_bump = scenario_info['speed_bump']
    if 'drive_way' in scenario_info.keys():
        driveway = scenario_info['drive_way']
    if 'stop_sign' in scenario_info.keys():
        stop_sign = scenario_info['stop_sign']

    # [x,y,theta,curvature,one of [speedlimit, trafficlight, crosswalks] ]
    # 0 is red light, 1 is crosswalk, other is speed_limit

    # Visualize Lane Centerlines
    for key, polyline in lane_polylines.items():
        map_type = polyline[0, 6]
        if map_type == 1 or map_type == 2 or map_type == 3:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'g', linestyle='solid', linewidth=1)

    # Visualize Lane Markings
    for key, polyline in road_polylines.items():
        map_type = polyline[0, 6]
        if map_type == 6:
            ax.plot(polyline[:, 0], polyline[:, 1], 'w',
                    linestyle='dashed', linewidth=1)
        elif map_type == 7:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'w', linestyle='solid', linewidth=1)
        elif map_type == 8:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'w', linestyle='solid', linewidth=1)
        elif map_type == 9:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'xkcd:yellow', linestyle='dashed', linewidth=1)
        elif map_type == 10:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'xkcd:yellow', linestyle='dashed', linewidth=1)
        elif map_type == 11:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'xkcd:yellow', linestyle='solid', linewidth=1)
        elif map_type == 12:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'xkcd:yellow', linestyle='solid', linewidth=1)
        elif map_type == 13:
            ax.plot(polyline[:, 0], polyline[:, 1],
                    'xkcd:yellow', linestyle='dotted', linewidth=1)
        elif map_type == 15:
            ax.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
        elif map_type == 16:
            ax.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)

    # Visualize Special Road Elements
    if 'stop_sign' in scenario_info.keys():
        for key, polyline in stop_sign.items():
            map_type = polyline[0, 6]
            if map_type == 17:
                if len(polyline) < 2:
                    ax.add_patch(plt.Circle(polyline[0][:2], 2, color='r'))
                else:
                    for pol in polyline:
                        ax.add_patch(plt.Circle(pol[:2], 2, color='r'))

        for key, polyline in crosswalks_p.items():
            map_type = polyline[0, 6]
            if map_type == 18:
                polyline = polygon_completion(polyline).astype(np.float32)
                ax.plot(polyline[:, 0], polyline[:, 1], 'b', linewidth=1)

        for key, polyline in speed_bump.items():
            map_type = polyline[0, 6]
            if map_type == 19:
                polyline = polygon_completion(polyline).astype(np.float32)
                ax.plot(polyline[:, 0], polyline[:, 1],
                        'xkcd:orange', linewidth=1)

        for key, polyline in driveway.items():
            map_type = polyline[0, 6]
            if map_type == 20:
                polyline = polygon_completion(polyline).astype(np.float32)
                ax.plot(polyline[:, 0], polyline[:, 1],
                        'xkcd:orange', linewidth=1)

        # viz_ref_line
        #ax.plot(ref_path[:, 0], ref_path[:, 1], 'y', linewidth=2, zorder=4)


def plot_dynamic_objects(scenario_info: dict, t_step: int = 11) -> None:
    sdc_track_index = scenario_info['sdc_track_index']
    all_trajs = scenario_info['tracks_info']['trajs']

    # [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid, object_type]
    def plot_object_trajectory(t_step: int, trajectory: np.ndarray, is_ego: bool = False) -> None:
        ax = plt.gca()
        history = trajectory[:t_step]
        future = trajectory[t_step:]
        history_mask = history[:, 9] > 0
        future_mask = future[:, 9] > 0

        if future[0, 9] == 0:
            return

        if is_ego:
            history_color = 'mistyrose'
            future_color = 'tomato'
        elif trajectory[0, 10] == 1:
            history_color = 'violet'
            future_color = 'magenta'
        elif trajectory[0, 10] == 2:
            history_color = 'lightskyblue'
            future_color = 'deepskyblue'
        elif trajectory[0, 10] == 3:
            history_color = 'springgreen'
            future_color = 'lime'

        ax.plot(history[history_mask][::5, 0], history[history_mask][::5, 1],
                linewidth=2, color=history_color, marker='*', markersize=2, zorder=4)
        ax.plot(future[future_mask][::5, 0], future[future_mask][::5, 1],
                linewidth=2, color=future_color, marker='.', markersize=6, zorder=4)
        rect = plt.Rectangle((future[0, 0] - future[0, 3]/2, future[0, 1] - future[0, 4]/2), future[0, 3], future[0, 4], linewidth=2, color=future_color, alpha=0.6, zorder=5,
                             transform=mpl.transforms.Affine2D().rotate_around(*(future[0, 0], future[0, 1]), future[0, 6]) + ax.transData)
        ax.add_patch(rect)

    # Visualize all trajectories (and their bounding boxes)
    for i, traj in enumerate(all_trajs):
        plot_object_trajectory(t_step, traj, i == sdc_track_index)
        
def plot_dynamic_objects_v2(scenario_info: dict, 
                            t_step: int = 11, 
                            dpi: int = 200, 
                            img_res=[512,512], 
                            view_range: float = 100.0 , 
                            direction_lines:list =None ) -> None:
    sdc_track_index = scenario_info['sdc_track_index']

    obj_trajs_full = scenario_info['tracks_info']['trajs']
    obj_mask=scenario_info['tracks_info']['trajs'][:,:,10]==1
    
    
    # obj_mask=scenario_info['all_agent'][:,:,10]==1
    # obj_trajs_full = scenario_info['all_agent']
    
    # [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, object_type]
    
    n_object, t_end, n_feature = scenario_info['tracks_info']['trajs'].shape
    # n_object, t_end, n_feature = scenario_info['all_agent'].shape
    t_end=t_step+10
    ego_position = np.array(obj_trajs_full[sdc_track_index][1][:2])  # [x,y]
    ego_theta = np.array(obj_trajs_full[sdc_track_index][1][6])  
    
    obj_mask=obj_trajs_full[:,:,10]==1
    obj_trajs_full[:, :, :2]=obj_trajs_full[:, :, :2]-ego_position
    obj_trajs_full=obj_trajs_full[obj_mask,:].reshape([-1,obj_mask.shape[1],n_feature])
    # obj_trajs_full = transform_scenario(obj_trajs_full, ego_position,ego_theta)
    
    def plot_object_trajectory(
        t_step: int, 
        trajectory: np.ndarray, 
        is_ego: bool = False,
        plot_rectangle: bool = False
        ) -> None:

        ax = plt.gca()
        history = trajectory[:t_step] # [N,10]
        future = trajectory[t_step:t_end]
        history_mask = history[:, 9] > 0
        future_mask = future[:, 9] > 0

        # calcuate the velocity
        future_offseted = np.roll(future, shift=1, axis=0)
        buffer_points = np.concatenate(
            (future[:, 0:2], future_offseted[:, 0:2]), axis=-1
        )  # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]
        velocity = np.array(buffer_points[:, 0:2] - buffer_points[:, 2:4])
        
        # maximum velocity for normalization is set to 20m/s, 
        # normalize the velocity from abs[-20m/s, 20m/s] to [0,1]
     
        # calcuate the velocity scalar
        velocity_scalar = np.linalg.norm(velocity, axis=1)

        for i in range(len(velocity_scalar)):
            
            velocity_scalar[i] = velocity_scalar[i]/60 + 0.5
        
        
        # for plot, only plot single time step
        plot_mask=np.zeros_like(future_mask)
        plot_mask[0]=True       
        future_mask=plot_mask      
        if future[0, 9] == 0:
            return

        history_color = np.zeros([history.shape[0],3]) # [N,3]
        future_color = np.zeros([future.shape[0],3])
        velocity_color = np.zeros([future.shape[0],3])
        
        
        for idx, row in enumerate(history):
            history_color[idx,2] = 1-np.abs(idx-t_step)/t_step
        for idx, row in enumerate(future):  
            future_color[idx,2] = np.abs((t_end-t_step)-idx)/(t_end-t_step)
            velocity_color[idx,2] = velocity_scalar[idx]
        
        ## plot traj and rectangle seperately
        gap=1
        
        # plot the future trajs
        line_list=direction_lines[1:,:,:].tolist()
        lines=MultiLineString(line_list)
        for i in range(len(future[future_mask][::gap])):
   
            ## step1  verify whether the rectangle is on the direction line
            polygon_shapely=((future[future_mask][i*gap, 0] - future[future_mask][i*gap, 3]/2, future[future_mask][i*gap, 1] - future[future_mask][i*gap, 4]/2),
                            (future[future_mask][i*gap, 0] - future[future_mask][i*gap, 3]/2, future[future_mask][i*gap, 1] + future[future_mask][i*gap, 4]/2),
                            (future[future_mask][i*gap, 0] + future[future_mask][i*gap, 3]/2, future[future_mask][i*gap, 1] + future[future_mask][i*gap, 4]/2),
                            (future[future_mask][i*gap, 0] + future[future_mask][i*gap, 3]/2, future[future_mask][i*gap, 1] - future[future_mask][i*gap, 4]/2),
                            (future[future_mask][i*gap, 0] - future[future_mask][i*gap, 3]/2, future[future_mask][i*gap, 1] - future[future_mask][i*gap, 4]/2),
                            )
            
            rectangle_shapely = Polygon(polygon_shapely)
            rotated_rectangle_shapely = rotate(rectangle_shapely, future[future_mask][i*gap, 6],use_radians=True)
            
            if not lines.is_valid:
                continue
                print("there is not valid in lines_shapely.bounds")
            if not rectangle_shapely.is_valid:
                print("there is not valid in rectangle_shapely.areas")
                print(future[future_mask][i*gap, :])          
            if np.isnan(lines.length):
                print("there is nan in lines_shapely.length")
            # [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, object_type]
            try:
                lines.intersects(rotated_rectangle_shapely)
            except BaseException:
                # print("there is not valid in lines_shapely.intersects")
                continue 
            else:   
                if lines.intersects(rotated_rectangle_shapely):
                    rect=plt.Rectangle((future[future_mask][i*gap, 0] - future[future_mask][i*gap, 3]/2, future[future_mask][i*gap, 1] - future[future_mask][i*gap, 4]/2)
                                    , future[future_mask][i*gap, 3],future[future_mask][i*gap, 4]
                                    , linewidth=1 #alpha=velocity_color[i*gap+1,2]
                                    , facecolor=velocity_color[i*gap+1,:]
                                    , edgecolor=velocity_color[i*gap+1,:]#,zorder=3
                                    , transform=mpl.transforms.Affine2D().rotate_around(*(future[future_mask][i*gap, 0], future[future_mask][i*gap, 1]), future[future_mask][i*gap, 6]) + ax.transData)
                    ax.add_patch(rect)
            
            
        ax.set_aspect('equal')
        ax.axis('equal')
        ax.set_facecolor(np.array([0.0,0.0,0.0]))
        ax.set(xlim=(-view_range,  view_range), ylim=(-view_range, view_range))
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
   
        
    
    # Visualize all bounding boxes)
    fig, ax = plt.subplots(1, 1, figsize=(
        img_res[0]/dpi, img_res[1]/dpi), dpi=dpi)
    for i, traj in enumerate(obj_trajs_full):
        plot_object_trajectory(t_step, traj, i == sdc_track_index)
    # plt.show()
    # save the traj image  
    buffer_scatter = io.BytesIO()
    plt.savefig(buffer_scatter, format="png")
    plt.close()  
    
    buffer_scatter.seek(0)
    image_traj = Image.open(buffer_scatter)
    image_traj = image_traj.convert('RGB') 
        
    # save the rectangle image
    buffer_scatter = io.BytesIO()
    plt.savefig(buffer_scatter, format="png")
    plt.close()  
    
    ### transfer images to tensor
    to_tensor=transforms.ToTensor()
    traj_tensor=to_tensor(image_traj)
    traj_tensor=traj_tensor[[2],:,:]
    
    return traj_tensor

def animate_scenario(t_step: int, t_res: float, t_start: int, scenario_info: dict) -> None:
    """
    Function for visualizing an animated scenario
    """
    # Set Axes
    ax = plt.gca()
    ax.clear()
    ax.set_title(f'Simulation Time = {(t_step - t_start)*t_res:.1f} s')
    ax.set_facecolor('xkcd:grey')
    ax.margins(0)
    ax.set_aspect('equal')
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    # Plot the static map layer
    plot_static_map(scenario_info)
    # Plot the dynamic object layer
    plot_dynamic_objects(scenario_info, t_step)


def visualize_scenario(scenario_info: dict, t_start: int = 10, t_steps: int = 0, t_res: float = 0.1):
    """
    Function for visualizing an animated scenario
    """
    # [N_0, 91, 11]
    n_object, t_end, n_feature = scenario_info['tracks_info']['trajs'].shape
    
    if t_steps > 0 and t_start + t_steps <= t_end:
        t_end = t_start + t_steps

    # Create an animation
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, partial(animate_scenario, t_res=t_res, t_start=t_start,
                                  scenario_info=scenario_info), frames=np.arange(t_start, t_end, 1))

    plt.tight_layout()
    plt.show()
    plt.close()

    # ani.save("./media/sim_demo.mp4")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--load_path', default="./data/waymo/processed_scenarios_training",
                        type=str, help='path to dataset files')
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')

    for i, file in enumerate(data_files):
        with open(file, 'rb') as f:
            scenario_info = pickle.load(f)
        visualize_scenario(scenario_info)
        # break
