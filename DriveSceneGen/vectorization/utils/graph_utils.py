import random

import numpy as np
import networkx as nx
from PIL import Image


def distance(point1: tuple, point2: tuple) -> float:
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])


def are_close_enough(point1: tuple, point2: tuple, thresh: float=1.0) -> bool:
    if distance(point1, point2) > thresh:
        return False
    else:
        return True


def normalize_angle_rad(angle) -> float:
    # normalize angle to (-pi, pi]
    while angle > np.pi:
        angle = angle - 2*np.pi
    while angle <= -np.pi:
        angle = angle + 2*np.pi
    return angle


def normalize_dx_dy(dx, dy) -> tuple:
    norm = np.hypot(dx, dy)
    try:
        dx_norm = dx/norm
        dy_norm = dy/norm
    except RuntimeWarning:
        print(f'dx = {dx} dy = {dy} norm = {norm} dx_norm = {dx_norm} dy_norm = {dy_norm}')
    
    return dx_norm, dy_norm


def calc_path_yaw_diff(yaw1: float, yaw2: float) -> float:
    return np.fabs(normalize_angle_rad(yaw1 - yaw2 - np.pi))


def correct_path_direction(path: list, n1: tuple, n2: tuple) -> list:
    if distance(n1, path[0]) <= distance(n2, path[0]):
        return path
    else: 
        path_T = np.array(path, dtype=float).T
        if path_T.shape[0] > 2:
            xs = path_T[0, ::-1]
            ys = path_T[1, ::-1]
            yaws = path_T[2, ::-1] + np.pi
            ks = path_T[-2, ::-1]
            s = path_T[-1, :]
            return list(zip(xs, ys, yaws, ks, s))
        else:
            return path[::-1]


def join_paths(path1, path2):
    if len(path2) > 0:
        path2 = path2[1:]
        return path1 + [(*(pt[:-1]), path1[-1][-1] + pt[-1]) for pt in path2]
    else:
        return path1
    

def connect_small_gaps(graph: nx.Graph, nodes: list, thresh: int=4) -> nx.Graph:
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes[i+1:]):
            dist = np.hypot(n1[0] - n2[0], n1[1] - n2[1])
            if dist <= thresh:
                n1_neighbours = list(graph.edges(n1, keys=True))
                n2_neighbours = list(graph.edges(n2, keys=True))
                if n1_neighbours and n2_neighbours:
                    n1, n1_neighbour, k1 = n1_neighbours[0]
                    n2, n2_neighbour, k2 = n2_neighbours[0]
                    
                    e1 = graph[n1][n1_neighbour][k1]
                    e1_path = correct_path_direction(e1['path'], n1_neighbour, n1)
                    e2 = graph[n2][n2_neighbour][k2]
                    e2_path = correct_path_direction(e2['path'], n2, n2_neighbour)

                    new_path = e1_path + e2_path
                    graph.add_edge(n1_neighbour, n2_neighbour, path=new_path, d=len(new_path)-1)
                    graph.remove_node(n1)
                    graph.remove_node(n2)
                    break

    return graph


def estimate_path_yaws(path: list, local_length: int=10) -> tuple:
    path_np = np.array(path)

    if path_np.shape[0] > local_length:
        front_delta = path_np[local_length-1] - path_np[0]
        rear_delta = path_np[-1] - path_np[-local_length]
    else:
        front_delta = path_np[-1] - path_np[0]
        rear_delta = front_delta

    front_delta = normalize_dx_dy(front_delta[0], front_delta[1])
    rear_delta = normalize_dx_dy(rear_delta[0], rear_delta[1])
    front_yaw = np.arctan2(front_delta[1], front_delta[0])  
    rear_yaw = np.arctan2(rear_delta[1], rear_delta[0])

    return front_yaw, front_delta, rear_yaw, rear_delta


def find_node_directions(graph: nx.Graph, nodes_terminal: list, img_color: Image) -> tuple:
    inlets = []
    outlets = []
    for n1 in nodes_terminal:
        n1, n2, k = list(graph.edges(n1, keys=True))[0]
        dx, dy = normalize_dx_dy(n2[0] - n1[0], n2[1] - n1[1])
        node_angle = np.rad2deg(np.arctan2(dy, dx))

        n1_color = img_color.getpixel(n1)
        color_dx, color_dy = normalize_dx_dy(n1_color[0] - 128, 128 - n1_color[1])
        color_angle = np.rad2deg(np.arctan2(color_dy, color_dx))
        
        angle_diff = np.fabs(color_angle - node_angle)

        if angle_diff < 90.0:
            direction = 1 # inlet
            inlets.append((n1[0], n1[1], dx, dy, color_dx, color_dy, direction))
        else: 
            direction = 0 # outlet
            dx = -dx
            dy = -dy
            outlets.append((n1[0], n1[1], dx, dy, color_dx, color_dy, direction))

    return np.array(inlets), np.array(outlets)


def get_edges_between_nodes(graph: nx.Graph, n1, n2) -> list:
    edges = [edge for edge in graph.edges(n1, keys=True) if edge[1] == n2]
    return edges


def trace_route(graph: nx.Graph, route: list) -> list:
    waypoints = []
    for i in range(len(route) - 1):
        n1 = route[i]
        n2 = route[i+1]
        edges = get_edges_between_nodes(graph, n1, n2)
        n1, n2, k = edges[0]
        e = graph[n1][n2][k]
        points = e['path']
        waypoints = waypoints + correct_path_direction(points, n1, n2)

    return waypoints


def downsample_path(path: np.ndarray, ratio: int=2) -> np.ndarray:
    if path.shape[0] > ratio:
        new_path = path[::ratio]
        if path.shape[0] % ratio > ratio/2:
            new_path = np.append(new_path, [path[-1]], axis=0)
        else:
            new_path[-1] = path[-1]
        # new_path[-1] = path[-1]
        return new_path
    elif path.shape[0] == 0:
        return np.array([])
    else:
        return np.take(path, [1, -1], axis=0)
    
    
def random_color():
    return '#{:02X}{:02X}{:02X}'.format(random.randint(30, 220), random.randint(30, 220), random.randint(30, 220))


def generate_random_colors(n):
    return [random_color() for _ in range(n)]


def graph_to_polylines(g: nx.Graph) -> list:
    polylines = []

    if isinstance(g, nx.MultiGraph) or isinstance(g, nx.MultiDiGraph):
        for (n1, n2, k) in g.edges(keys=True):
            edge = g[n1][n2][k]
            path = edge['path']
            coords = np.array(path)
            polylines.append(coords)
    else:
        for (n1, n2) in g.edges():
            edge = g[n1][n2]
            path = edge['path']
            coords = np.array(path)
            polylines.append(coords)

    return polylines


def tansform_to_world_frame(polyline: np.ndarray, center: tuple, scale: float) -> np.ndarray:
    # polyline: [x, y, yaw, k, s]
    polyline[:, 0] = polyline[:, 0] * scale - center[0] # x
    polyline[:, 1] = center[1] - polyline[:, 1] * scale # y
    polyline[:, 2] = -polyline[:, 2]                    # yaw
    polyline[:, 3] = polyline[:, 3] / scale             # k
    polyline[:, 4] = polyline[:, 4] * scale             # s
    return polyline


def polylines_to_world_frame(polylines: list, img_shape: tuple, map_range: float=80) -> list:
    scale = map_range / img_shape[0] # m/pixel
    center = (img_shape[0]/2 * scale, img_shape[1]/2 * scale)
    return [tansform_to_world_frame(polyline, center, scale) for polyline in polylines]


def polylines_to_output(polylines: list) -> list:
    """
    Parameters
    ---
    polylines: `list` [polyline1, polyline2, ...]
        polyline: `np.ndarray` [[x, y, yaw, k, s], [x, y, yaw, k, s], ...]

    Returns
    ---
    lanes: `list` [lane1, lane2, ...]
        lane: `list` [point1, point2, ...] (follow the sequence of the traffic flow)
            point: `list` [x, y, z, dx, dy, dz]
    """
    lanes = []
    for polyline in polylines:
        dx = np.cos(polyline[:, 2])
        dy = np.sin(polyline[:, 2])
        zeros = np.zeros_like(dx)
        lane = np.stack((polyline[:, 0], polyline[:, 1], zeros, dx, dy, zeros), axis=-1)
        lanes.append(lane)

    return lanes