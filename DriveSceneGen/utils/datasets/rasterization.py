import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from DriveSceneGen.utils.datasets.map_processing import (dxdy_normalization,
                                  generate_batch_polylines_from_map,
                                  transform_scenario)
from DriveSceneGen.utils.datasets.visualization import plot_dynamic_objects_v2
from DriveSceneGen.utils.datasets.vector_normalization import generate_desired_type_polylines_list

def rasterize_static_map(
    scenario_info: dict,
    img_res: tuple = (512, 512),
    dpi: int = 200,
    map_range: float = 100.0,
    des_row_size: int = 100,
    output_img_path: str = None,
    with_agent = False,
    scatter_as_line = True,
    resize = False,
    scatter_size = 1.5,
    save_imgs = False,
    save_png_polys = False
):

    # dataset
    all_polylines = scenario_info["lane"]
    all_points = np.array([v[:, :7] for v in all_polylines.values()], dtype=object)
    all_points = np.vstack(all_points)
    sdc_track_index = scenario_info["sdc_track_index"]
    obj_trajs_full = scenario_info["tracks_info"]["trajs"]

    ego_position = np.array(obj_trajs_full[sdc_track_index][10][:2])  # [x,y]
    ego_theta = np.array(obj_trajs_full[sdc_track_index][10][6])  # [theta]

    all_polylines, all_masks = generate_batch_polylines_from_map(
        all_points,
        point_sampled_interval=1,
        vector_break_dist_thresh=1.0,
        num_points_each_polyline=100,
    )

    
    # Rotate the map following the ego vehicle's orientation
    all_polylines = transform_scenario(all_polylines, ego_position, ego_theta)

    # normalized polyline dx dy for visualization
    all_polylines = dxdy_normalization(all_polylines, des_row_size=des_row_size)

    ######################### Plot the scatter of dx,dy #########################
    # plot all kinds of polylines dx dy
    # Create a figure according to the specified resolution
    fig, ax = plt.subplots(1, 1, figsize=(img_res[0] / dpi, img_res[1] / dpi), dpi=dpi)

    direction_lines = np.empty((1, 100, 2))
    desired_type_polylines_list=[]
    for polyline, masks in zip(all_polylines, all_masks):
        gap = 1

        map_type = polyline[0, 6]
        
        if map_type > 1 and map_type < 3:
            normalized_polyline = polyline
            normalized_polyline[masks, 3:5] = np.squeeze(polyline[masks][:, 3:5])

            colors_array_dx = np.zeros((normalized_polyline[masks].shape[0], 3))
            colors_array_dy = np.zeros((normalized_polyline[masks].shape[0], 3))
            colors_array = np.zeros((normalized_polyline[masks].shape[0], 3))
            # print the waypoint color as the value of dx dy
            for index, waypoint in enumerate(normalized_polyline[masks]):
                color_waypoint_dx = np.array([waypoint[3], 0.0, 0.0])
                color_waypoint_dy = np.array([0.0, waypoint[4], 0.0])
                color_waypoint = np.array([waypoint[3], waypoint[4], 0.0])

                colors_array_dx[index] = color_waypoint_dx
                colors_array_dy[index] = color_waypoint_dy
                colors_array[index] = color_waypoint

            
            if scatter_as_line:
                # scatter as line
                ax.scatter(
                    polyline[masks][:, 0],
                    polyline[masks][:, 1],
                    c=colors_array,
                    s=scatter_size,
                    marker="D",
                )
            else:
                # plot the line
                polyline_len=polyline[masks][::gap].shape[0]
                for i in range(polyline_len-1):
                    ax.plot(polyline[masks][i*gap:(i+2)*gap:gap, 0], 
                            polyline[masks][i*gap:(i+2)*gap:gap, 1],
                            linewidth=1.5, 
                            color=colors_array[i*gap,:]) 

            if (
                polyline[masks][::gap, 0:2].reshape(1, -1, 2).shape[1]
                == direction_lines.shape[1]
            ):
                direction_lines = np.append(
                    direction_lines,
                    polyline[masks][::gap, 0:2].reshape(1, -1, 2),
                    axis=0,
                )
            desired_type_polylines_list.append(polyline[masks])
            
    ax.set_facecolor(np.array([0.5, 0.5, 0.5]))
    ax.axis("equal")
    ax.set(xlim=(-map_range, map_range), ylim=(-map_range, map_range))
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()
    if save_imgs:
        plt.savefig(output_img_path)
        plt.close()
        
    elif save_png_polys:
        buffer_scatter = io.BytesIO() 
        plt.savefig(buffer_scatter, format="png")
        plt.close()

        # read from the buffer
        buffer_scatter.seek(0)
        image_scatter = Image.open(buffer_scatter)
        # image_scatter.show()
        # assume polylines to paths
        desired_map_type_list=[2]
        all_points[:, :2]=all_points[:, :2]-ego_position
        paths_list,too_less_polylines=generate_desired_type_polylines_list(
                                                all_points, point_sampled_interval=1,  
                                                points_break_dist_thresh=1.0,
                                                desired_map_type_list=desired_map_type_list,
                                                filtering=True,filter_distance=map_range)
        
        
        output_dict={}
        output_dict['fig_png']=image_scatter # 512x512x3
        output_dict['paths_list']=paths_list # 128x10x4 x,y,z, type    
        return output_dict,too_less_polylines
    else:
        # plt.show()
        # Save the figure to a buffer
        buffer_scatter = io.BytesIO()
        plt.savefig(buffer_scatter, format="png")
        plt.close()

        # read from the buffer
        buffer_scatter.seek(0)
        image_scatter = Image.open(buffer_scatter)
        image_scatter = image_scatter.convert("RGB")
        # transfer the image to array, then to tensor
        transform = transforms.ToTensor()
        fig_2_tensor = transform(image_scatter)
        
        # resize to 256x256
        if resize:
            transform=transforms.Resize((256, 256),antialias=True)
            fig_2_tensor=transform(fig_2_tensor)

        #plot dynamic objects
        if with_agent:
            fig_2_tensor = fig_2_tensor[:2, :, :]
            t_step = 1
            traj_tensor = plot_dynamic_objects_v2(
                scenario_info, 
                t_step, dpi, 
                img_res,
                map_range, 
                direction_lines
            )
            fig_tensor = torch.cat((fig_2_tensor, traj_tensor), dim=0)
        else:
            fig_tensor = fig_2_tensor
        # return fig_tensor [256,256,3]
        fig_tensor = fig_tensor.permute(1, 2, 0)
        return fig_tensor
