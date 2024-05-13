import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler


def filter_polylines_by_distance(polylines: dict, center: np.ndarray, thresh_dist=100):
    """
    Filter all polylines (points) on the map based on their distance to the ego vehicle's inital location

    ---
    Args:
        scenario_info (`dict`): A Dictionary containign ALL the data about the scenario that was previously unpacked from the waymo dataset
        center (`np.ndarray`): A 2 x 1 vector containing the x and y coordinates of the center from which the range is calculated
        thresh_dist (`float`): The distance threshold beyond which all points will be filtered out
    ---
    Returns:
        filtered_polylines: (dict)
    """
    filtered_polylines = {}
    for unique_id, polyline in polylines.items():
        polyline_xy = np.array(polyline)[:, :2]  # Get the x, y coordinates
        # Compute Euclidean distance
        distances = np.sqrt(np.sum((polyline_xy - center) ** 2, axis=1))

        # Filter points in polyline within threshold distance
        filtered_polyline = polyline[distances <= thresh_dist]
        if len(filtered_polyline) > 0:
            filtered_polylines[unique_id] = filtered_polyline
    return filtered_polylines


def segment_points_to_polylines(points: np.ndarray, dist_thresh: float = 1.0):
    """
    Segment a list of points into polylines based on the distance between them.

    ---
    Args:
        points (num_points, 8): [x, y, z, dir_x, dir_y, dir_z, global_type, theta]
    ---
    Returns:
        polylines: (num_polylines, num_points_each_polyline, 8)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """

    points_offseted = np.roll(points, shift=1, axis=0)
    buffer_points = np.concatenate(
        (points[:, 0:2], points_offseted[:, 0:2]), axis=-1
    )  # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]
    difference = np.array(buffer_points[:, 0:2] - buffer_points[:, 2:4])
    break_idxs = (np.linalg.norm(difference, axis=-1) > dist_thresh).nonzero()[0]
    polylines = np.array_split(points, break_idxs, axis=0)
    
    for polyline in polylines:
        try:
            polyline[0, 3:5] = polyline[1, 3:5]
        except IndexError:
            continue
    return polylines

def generate_batch_polylines_from_map(
    polylines,
    point_sampled_interval=1,
    vector_break_dist_thresh=1.0,
    num_points_each_polyline=20,
):
    """
    Args:
        polylines (num_points, 8): [x, y, z, dir_x, dir_y, dir_z, global_type, theta]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 8)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """

    point_dim = polylines.shape[-1]
    sampled_points = polylines[::point_sampled_interval]

    polylines_list = segment_points_to_polylines(
        sampled_points, vector_break_dist_thresh
    )

    ret_polylines = []
    ret_polylines_valid = []
    ret_polylines_mask = []

    for k in range(len(polylines_list)):
        if len(polylines_list[k]) <= 0:
            continue
        for idx in range(0, len(polylines_list[k]), num_points_each_polyline):
            new_polyline = polylines_list[k][idx : idx + num_points_each_polyline]

            cur_polyline = np.zeros(
                (num_points_each_polyline, point_dim), dtype=np.float32
            )
            cur_valid = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_mask = np.full((num_points_each_polyline), False, dtype=bool)

            cur_polyline[: len(new_polyline)] = new_polyline
            cur_valid[: len(new_polyline)] = 1
            cur_mask[: len(new_polyline)] = True

            ret_polylines.append(cur_polyline)
            ret_polylines_valid.append(cur_valid)
            ret_polylines_mask.append(cur_mask)

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_valid = np.stack(ret_polylines_valid, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    # print(ret_polylines.shape, ret_polylines_valid.shape, ret_polylines_mask.shape)
    ret_polylines_features = np.concatenate(
        (ret_polylines, ret_polylines_valid[:, :, np.newaxis]), axis=2
    )

    return ret_polylines_features, ret_polylines_mask


def polyline_interpolation(polylines):
    delete_keys = []
    polylines_interpolated = {}
    polylines_interpolated_decrease_channel = {}
    for key, polyline in polylines.items():
        if len(polyline) < 3:
            delete_keys.append(key)
            continue

        # Linear length along the line:
        distance_xyz = np.cumsum(
            np.sqrt(np.sum(np.diff(polyline[:, 0:3], axis=0) ** 2, axis=1))
        )
        distance_xyz = np.insert(distance_xyz, 0, 0) / distance_xyz[-1]

        # distance for point(dx,dy,dz)
        distance_dxdydz = np.cumsum(
            np.sqrt(np.sum(np.diff(polyline[:, 3:6], axis=0) ** 2, axis=1))
        )
        distance_dxdydz = np.insert(distance_dxdydz, 0, 0) / distance_dxdydz[-1]

        # distance for theta
        row_indice = np.arange(0, polyline.shape[0])
        theta_2d_points = np.array([row_indice, polyline[:, 7]]).T

        distance_theta = np.cumsum(
            np.sqrt(np.sum(np.diff(theta_2d_points, axis=0) ** 2, axis=1))
        )
        distance_theta = np.insert(distance_theta, 0, 0) / distance_theta[-1]

        # generate new x

        x_new = np.linspace(distance_xyz[0], distance_xyz[-1], num=128)

        dx_new = np.linspace(distance_dxdydz[0], distance_dxdydz[-1], num=128)

        theta_new = np.linspace(distance_theta[0], distance_theta[-1], num=128)

        # start interpolation
        interpolator_xyz = interpolate.interp1d(distance_xyz, polyline[:, 0:3], axis=0)
        interpolator_dxdydz = interpolate.interp1d(
            distance_dxdydz, polyline[:, 3:6], axis=0
        )
        interpolator_theta = interpolate.interp1d(
            distance_theta, theta_2d_points, axis=0
        )

        xyz_interpolated = interpolator_xyz(x_new)
        dxdydz_interpolated = interpolator_dxdydz(dx_new)
        theta_interpolated = interpolator_theta(theta_new)

        # polyline type
        polyline_type = np.ones([xyz_interpolated.shape[0], 1]) * polyline[0, 6]

        # assemble polyline
        polyline_interpolated = np.concatenate(
            (xyz_interpolated, dxdydz_interpolated), axis=1
        )
        polyline_interpolated = np.concatenate(
            (polyline_interpolated, polyline_type), axis=1
        )
        polyline_interpolated = np.concatenate(
            (polyline_interpolated, theta_interpolated[:, 1].reshape([-1, 1])), axis=1
        )

        # assemble polyline with only [x,,y,type]
        polyline_decrease_channel = np.concatenate(
            (xyz_interpolated[:, 0:2], polyline_type), axis=1
        )

        # polyline_interpolated[:,6]=polyline_type[:,0]

        # write back to polylines
        polylines_interpolated[key] = polyline_interpolated
        # polyline_interpolated[:,6]=polyline_type[:,0]

        # write back to polylines
        polylines_interpolated_decrease_channel[key] = polyline_decrease_channel

    for key in delete_keys:
        if key in polylines_interpolated:
            del polylines_interpolated[key]
            del polylines_interpolated_decrease_channel[key]

    return polylines_interpolated, polylines_interpolated_decrease_channel


def dxdy_normalization(polylines_list, des_row_size=128):
    """
    normalization dx,dy to plot dirction as a channel
    Args:
        polylines_list: all polylines in one scenario

    Returns:
        normalized_polylines_list: all polylines in one scenario with normalized dx,dy
    """
    polylines_array = np.array(polylines_list)
    normalized_polylines_array = polylines_array.copy()

    scaler = MinMaxScaler(feature_range=(0, 0.99))
    # print( normalized_polylines_array[:,:,3:5].shape)
    normalized_polylines_array[:, :, 3:5] = scaler.fit_transform(
        polylines_array[:, :, 3:5].reshape(-1, 2)).reshape(
                                                        [-1, 100, 2]
                                                        )  # [des_row_size,-1,2]
    normalized_polylines_list = []

    for polyline in normalized_polylines_array:
        normalized_polylines_list.append(polyline)

    return normalized_polylines_list


def transform_scenario(all_polylines, ego_position, ego_theta):
    """
    Rotate the map following the ego vehicle's orientation

    Args:
        all_polylines: all polylines in one scenario
        all_masks: all masks in one scenario
    Returns:
        return_rotated_polylines: all polylines in one scenario with rotation
    """
    ego_theta = ego_theta  # -0.5*np.pi
    rotation_matrix = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    return_rotated_polylines = []

    all_polylines_array = np.array(all_polylines)

    all_polylines_array[:, :, :2] = all_polylines_array[:, :, :2] - ego_position

    # all_polylines_array[:, :, :2] = np.dot(all_polylines_array[:, :, :2],rotation_matrix)# Get the x, y coordinates

    # if all_polylines_array.shape[-1]>=6:
    #     all_polylines_array[:, :, 3:5] = np.dot(all_polylines_array[:, :, 3:5],rotation_matrix)

    for polyline in all_polylines_array:
        cutted_polyline = polyline

        return_rotated_polylines.append(polyline)

    # for polyline, masks in zip(all_polylines, all_masks):

    #     # step1: center is ego position
    #     polyline_xy_tranlated=np.array(polyline)[:, :2]-ego_position

    #     # step2: rotate the map
    #     polyline_xy_rotated = np.dot(polyline_xy_tranlated,rotation_matrix)# Get the x, y coordinates
    #     polyline_dxdy_rotated = np.dot(np.array(polyline)[:, 3:5],rotation_matrix)

    #     polyline[:, :2] = polyline_xy_rotated
    #     polyline[:, 3:5] = polyline_dxdy_rotated

    #     return_rotated_polylines.append(polyline)

    return return_rotated_polylines

def preprocess_static_map(
    scenario_info: dict,
    keep_types: list = [
        "lane",
        "road_polylines",
        "crosswalk",
        "speed_bump",
        "drive_way",
        "stop_sign",
    ],
    thresh_dist: float = 100.0,
):
    """
    Filter all polylines (points) on the map based on their distance to the ego vehicle's inital location, and cast the preprocessed map data to a tensor

    ---
    Args:
        scenario_info (`dict`): A Dictionary containign ALL the data about the scenario that was previously unpacked from the waymo dataset
        thresh_dist (`float`): The distance threshold beyond which all points will be filtered out
    ---
    Returns:
        filtered_scenario_info (`dict`): The preprocessed scenario information
    """
    all_map_keys = [
        "lane",
        "road_polylines",
        "crosswalk",
        "speed_bump",
        "drive_way",
        "stop_sign",
    ]
    sdc_track_index = scenario_info["sdc_track_index"]
    obj_trajs_full = scenario_info["track_infos"]["trajs"]
    ego_position = np.array(obj_trajs_full[sdc_track_index][10][:2])  # [x,y]
    ego_theta = np.array(obj_trajs_full[sdc_track_index][10][6])  # [x,y]
    # polyline_type_object_keys = [
    #    'lane', 'road_polylines', 'crosswalk', 'speed_bump', 'drive_way', 'stop_sign']
    simple_polyline_type_object_keys = [
        "lane",
        "road_polylines",
    ]  # , 'crosswalk', 'speed_bump', 'drive_way', 'stop_sign']

    ###################### Preprocessing polylines ###########################
    for key in scenario_info.keys():
        if key in all_map_keys:
            if key in keep_types:
                # Step 1: Filter all points outside the 100 m range
                scenario_info[key] = filter_polylines_by_distance(
                    scenario_info[key], ego_position, thresh_dist
                )
                # Step 2: Rotate the map following the ego vehicle's orientation

                scenario_info[key] = transform_scenario(scenario_info[key], ego_theta)

            else:
                scenario_info[key] = {}

    return scenario_info


def preprocess_static_map_polyline_to_row(
    scenario_info: dict, thresh_dist: float = 100.0
):
    """
    Filter all polylines (points) on the map based on their distance to the ego vehicle's inital location, and cast the preprocessed map data to a tensor

    ---
    Args:
        scenario_info (`dict`): A Dictionary containign ALL the data about the scenario that was previously unpacked from the waymo dataset
        thresh_dist (`float`): The distance threshold beyond which all points will be filtered out
    ---
    Returns:
        processed_scenario_info (`dict`): The preprocessed scenario information
        dictinary of polylines (`dict`): The preprocessed polylines
    """

    sdc_track_index = scenario_info["sdc_track_index"]
    obj_trajs_full = scenario_info["tracks_info"]["trajs"]
    ego_position = np.array(obj_trajs_full[sdc_track_index][10][:2])  # [x,y]

    polyline_type_object_keys = ["lane", "road_polylines"]

    ###################### Preprocessing polylines ###########################
    processed_scenario_info = {}
    for key in polyline_type_object_keys:
        # Step 1: Filter all points outside the 100 m range
        filtered_polylines = filter_polylines_by_distance(
            scenario_info[key], ego_position, thresh_dist
        )

        # Step 2: interpolation
        (
            filtered_interpolated_full_channel_polylines,
            filtered_interpolated_polylines,
        ) = polyline_interpolation(filtered_polylines)

        processed_scenario_info[key] = filtered_interpolated_polylines

    lane_polylines = processed_scenario_info["lane"]
    road_polylines = processed_scenario_info["road_polylines"]

    return lane_polylines, road_polylines
