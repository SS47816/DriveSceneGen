import copy
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from DriveSceneGen.utils.io import get_logger
from DriveSceneGen.vectorization.utils import image_utils, graph_utils
from DriveSceneGen.vectorization.curve import bezier_curve, cubic_spline
from DriveSceneGen.vectorization.graph import extract_network

logger = get_logger('image_to_polylines', logging.WARNING)


def image_to_graph(img: Image) -> tuple:
    rgb = (255, 255, 255)
    px = extract_network.find_color(img, rgb).T
    return extract_network.extract_network(px, min_distance=4)


def determine_node_direction(graph: nx.Graph, img_color: Image, n1, n2) -> tuple:
    e = graph[n1][n2][0]
    path = graph_utils.correct_path_direction(e['path'], n1, n2)
    n1_yaw, n1_delta, n2_yaw, n2_delta = graph_utils.estimate_path_yaws(path, local_length=20)

    dx_sum = 0.0
    dy_sum = 0.0
    for point in e['path']:
        color = img_color.getpixel(point)
        color_dx, color_dy = graph_utils.normalize_dx_dy(color[0] - 128, 128 - color[1])
        dx_sum += color_dx
        dy_sum += color_dy
    color_angle = np.arctan2(dy_sum, dx_sum)

    # n1_color = img_color.getpixel(n1)
    # color_dx, color_dy = graph_utils.normalize_dx_dy(n1_color[0] - 128, 128 - n1_color[1])
    # color_angle = np.arctan2(color_dy, color_dx)
    
    angle_diff = np.fabs(np.rad2deg(graph_utils.normalize_angle_rad(color_angle - n1_yaw)))

    if angle_diff < 90.0:
        direction = 1 # inlet
        n1_dx, n1_dy = graph_utils.normalize_dx_dy(n1_delta[0], n1_delta[1])
        n2_dx, n2_dy = graph_utils.normalize_dx_dy(n2_delta[0], n2_delta[1])
    else: 
        direction = 0 # outlet
        n1_dx, n1_dy = graph_utils.normalize_dx_dy(-n1_delta[0], -n1_delta[1])
        n2_dx, n2_dy = graph_utils.normalize_dx_dy(-n2_delta[0], -n2_delta[1])
        n1_yaw = graph_utils.normalize_angle_rad(n1_yaw + np.pi)
        n2_yaw = graph_utils.normalize_angle_rad(n2_yaw + np.pi)

    start = (n1[0], n1[1], n1_yaw, n1_dx, n1_dy, direction)
    end = (n2[0], n2[1], n2_yaw, n2_dx, n2_dy, direction)
    return start, end


def find_key_nodes(graph: nx.Graph, img_color: Image) -> tuple:
    nodes_1_degree = [(node[0], node[1]) for (node, degree) in graph.degree if degree == 1]
    
    terminal_nodes = []
    branching_nodes = []
    for n1 in nodes_1_degree:
        _, n2, k = list(graph.edges(n1, keys=True))[0]
        terminal, branch = determine_node_direction(graph, img_color, n1, n2)

        terminal_nodes.append(terminal)
        if graph.degree(n2) > 1:
            branching_nodes.append(branch)

    return np.array(terminal_nodes), np.array(branching_nodes)


def curve_is_valid(curve: np.ndarray, route: list, dist_tol: float=1.0, min_rate: float=0.5) -> bool:
    inlier_count = 0
    for node in route:
        dists = np.hypot(curve[:, 0] - node[0], curve[:, 1] - node[1])
        if np.min(dists) <= dist_tol:
            inlier_count += 1
    
    if inlier_count/len(route) >= min_rate:
        return True
    else:
        return False
    # return True


def route_is_valid(route: list, graph: nx.Graph) -> bool:
    for i in range(len(route) - 2):
        nl = route[i]
        n = route[i+1]
        nr = route[i+2]

        yaws = []
        nodes = []
        edges = list(graph.edges(n, keys=True))
        for n0, n1, k in edges:
            e = graph[n0][n1][k]
            e_path = graph_utils.correct_path_direction(e['path'], n0, n1) # paths pointing away from n0
            if e_path:
                n0_yaw, n0_delta, _, _ = graph_utils.estimate_path_yaws(e_path, 10)
                yaws.append(n0_yaw)
                nodes.append(n1)

        nl_id = nodes.index(nl)
        nr_id = nodes.index(nr)
        votes, connect_matrix = voting_by_yaw_angle(yaws)
        if connect_matrix[nl_id, nr_id] == False:
            return False
        elif graph_utils.calc_path_yaw_diff(yaws[nl_id], yaws[nr_id]) >= np.pi/4:
            return False

    return True


def find_paths_among_terminals(graph: nx.Graph, inlets: np.ndarray, outlets: np.ndarray, thresh: int=4) -> tuple:
    inlets_T = inlets.T.astype(int)
    outlets_T = outlets.T.astype(int)
    inlets = list(zip(inlets_T[0], inlets_T[1]))
    outlets = list(zip(outlets_T[0], outlets_T[1]))

    routes = []
    waypoints_all = []
    for n1 in inlets:
        for n2 in outlets:
            if nx.has_path(graph, source=n1, target=n2):
                route = nx.shortest_path(graph, n1, n2, weight='d', method='dijkstra')
                if route_is_valid(route, graph):
                    waypoints_all.append(graph_utils.trace_route(graph, route))
                    routes.append(route)

    logger.debug(f'Found {len(routes)} paths')
    return routes, waypoints_all


def voting_by_yaw_angle(yaws) -> tuple:
    votes = np.zeros(len(yaws), dtype=int)
    connect_matrix = np.zeros((len(yaws), len(yaws)), dtype=bool)

    for i, yaw1 in enumerate(yaws):
        diffs = []
        for j, yaw2 in enumerate(yaws):
            if i == j:
                diffs.append(2*np.pi)
            else:
                diff = graph_utils.calc_path_yaw_diff(yaw1, yaw2) # both angles are pointing away from n0
                diffs.append(diff) 
        
        # Pair the best matches
        min_id = np.argmin(diffs) # find the minimum angle difference
        votes[min_id] = votes[min_id] + 1
        connect_matrix[i, min_id] = True
        connect_matrix[min_id, i] = True
    
    # logger.debug(f'yaws: {yaws}')
    # logger.debug(f'votes: {votes}')
    # logger.debug(f'connect_matrix: \n{connect_matrix}')

    return votes, connect_matrix


def simplify_graph(graph: nx.Graph, img_gray: Image=None, plot: bool=False) -> nx.Graph:
    graph_changed = True
    iter = 0
    
    while graph_changed:
        graph_changed = False

        if plot:
            radius = 2
            fig, axes = plt.subplots(1, 2, figsize=(10, 20), sharex=True, sharey=True)
            axes = axes.ravel()
            axes[0].imshow(img_gray)
            axes[1].imshow(img_gray)
            axes[0].title.set_text('Before Reduced')
            axes[1].title.set_text('After Reduced')

        for n0, degree in graph.degree:
            if 'type' in graph.nodes[n0]:
                node_type = graph.nodes[n0]['type']
            else:
                node_type = ''

            if degree < 2 or node_type == 'branch':
                continue

            if plot:
                axes[0].add_patch(Circle(n0, radius, color='r'))
                axes[1].add_patch(Circle(n0, radius, color='r'))
            # Find all edges connected at n0, and their yaw angles (pointing away from n0)
            yaws = []
            paths = []
            nodes = []
            edges = list(graph.edges(n0, keys=True))
            for n, n1, k in edges:
                e1 = graph[n0][n1][k]
                e1_path = graph_utils.correct_path_direction(e1['path'], n0, n1) # paths pointing away from n0
                if len(e1_path) > 0:
                    n0_yaw, n0_delta, _, _ = graph_utils.estimate_path_yaws(e1_path, 100)
                    yaws.append(n0_yaw)
                    paths.append(e1_path)
                    nodes.append(n1)
                    if plot:
                        axes[0].add_patch(Circle(n1, radius, color='y'))
                        axes[0].plot([n0[0], n1[0]], [n0[1], n1[1]], c='y', label='edge')
                        axes[0].quiver(n0[0], n0[1], n0_delta[0], n0_delta[1], color='r', angles='xy', scale_units='xy', scale=0.1)

            # Among all edges at n0, vote for the best matches
            votes, connect_matrix = voting_by_yaw_angle(yaws)

            # Reconnect graph based on the vote result
            branch_ids = [i for (i, vote) in enumerate(votes) if vote > 1]
            passer_ids = [i for i in range(len(nodes)) if i not in branch_ids]
            
            for i in branch_ids:
                n1 = nodes[i]
                path = graph_utils.correct_path_direction(paths[i], n0, n1)
                n0_new_np = path[1]
                n0_new = (n0_new_np[0], n0_new_np[1])
                new_path = path[1:]
                graph.add_node(n0_new, type='branch')
                graph.add_edge(n0_new, n1, path=new_path, d=len(new_path)-1)
                if plot:
                    axes[1].add_patch(Circle(n0_new, radius, color='pink'))
                    axes[1].plot([n0_new[0], n1[0]], [n0_new[1], n1[1]], c='pink', label='root')

                js = [j for (j, val) in enumerate(connect_matrix[i]) if val == True]
                for j in js:
                    if j in passer_ids:
                        passer_ids.remove(j)
                    n2 = nodes[j]
                    new_path = [n0_new_np] + graph_utils.correct_path_direction(paths[j], n0, n2)
                    # new_path = join_paths([n0_new_np], correct_path_direction(paths[j], n0, n2))
                    graph.add_edge(n0_new, n2, path=new_path, d=len(new_path)-1)
                    if plot:
                        axes[1].add_patch(Circle(n2, radius, color='b'))
                        axes[1].plot([n0_new[0], n2[0]], [n0_new[1], n2[1]], c='b', label='branch')

            for i in passer_ids:
                n1 = nodes[i]
                js = [(i+j) for (j, val) in enumerate(connect_matrix[i, i:]) if val == True]
                for j in js:
                    n2 = nodes[j]
                    path1 = graph_utils.correct_path_direction(paths[i], n1, n0)
                    path2 = graph_utils.correct_path_direction(paths[j], n0, n2)
                    new_path = path1 + path2[1:]
                    # new_path = join_paths(path1, path2)
                    graph.add_edge(n1, n2, path=new_path, d=len(new_path)-1)
                    if plot:
                        axes[1].add_patch(Circle(n1, radius, color='g'))
                        axes[1].add_patch(Circle(n2, radius, color='g'))
                        axes[1].plot([n1[0], n2[0]], [n1[1], n2[1]], c='g', label='passer')

            graph.remove_node(n0)
            iter = iter + 1
            graph_changed = True
            break
        
        if plot:
            plt.show()

    return graph


def break_down_graph(graph: nx.Graph, plot: bool=False) -> nx.Graph:
    graph_changed = True
    iter = 0
    while graph_changed:
        graph_changed = False
        
        if plot:
            radius = 2
            fig, axes = plt.subplots(1, 2, figsize=(10, 20), sharex=True, sharey=True)
            axes = axes.ravel()
            axes[0].imshow(skel.T, cmap='gray')
            axes[1].imshow(skel.T, cmap='gray')
            axes[0].title.set_text('Before Breakdown')
            axes[1].title.set_text('After Breakdown')

        for n0, degree in graph.degree:
            if 'type' in graph.nodes[n0]:
                node_type = graph.nodes[n0]['type']
            else:
                node_type = ''

            if degree < 2 or node_type == 'entry' or node_type == 'exit':
                continue
            
            if plot:
                axes[0].add_patch(Circle(n0, radius, color='r'))
                axes[1].add_patch(Circle(n0, radius, color='r'))

            # Find all edges connected at n0, and their yaw angles (pointing away from n0)
            yaws = []
            paths = []
            nodes = []
            edges = list(graph.edges(n0, keys=True))
            for n, n1, k in edges:
                e1 = graph[n0][n1][k]
                e1_path = graph_utils.correct_path_direction(e1['path'], n0, n1) # paths pointing away from n0
                # n0_yaw = e1_path[0][2]
                n0_yaw, n0_delta, _, _ = graph_utils.estimate_path_yaws(e1_path, 10)
                yaws.append(n0_yaw)
                paths.append(e1_path)
                nodes.append(n1)
                if plot:
                    axes[0].add_patch(Circle(n1, radius, color='y'))
                    axes[0].plot([n0[0], n1[0]], [n0[1], n1[1]], c='y', label='edge')
                    axes[0].quiver(n0[0], n0[1], np.cos(n0_yaw), np.sin(n0_yaw), color='r', angles='xy', scale_units='xy', scale=0.1)

            # Among all edges at n0, vote for the best matches
            votes, connect_matrix = voting_by_yaw_angle(yaws)

            for i in range(connect_matrix.shape[0]):
                n1 = nodes[i]
                js = [(i+j) for (j, val) in enumerate(connect_matrix[i, i:]) if val == True]
                for j in js:
                    n2 = nodes[j]
                    path1 = graph_utils.correct_path_direction(paths[i], n1, n0)
                    path2 = graph_utils.correct_path_direction(paths[j], n0, n2)
                    # new_path = join_paths(path1, path2)
                    new_path = path1 + path2[1:]
                    graph.add_edge(n1, n2, path=new_path, d=new_path[-1][-1])
                    if plot:
                        axes[1].add_patch(Circle(n1, radius, color='g'))
                        axes[1].add_patch(Circle(n2, radius, color='g'))
                        axes[1].plot([n1[0], n2[0]], [n1[1], n2[1]], c='g', label='path')
                
            graph.remove_node(n0)
            iter = iter + 1
            graph_changed = True
            break
        
        if plot:
            plt.show()

    return graph


def find_intersections(graph: nx.Graph, img_color: Image, terminal_nodes: np.ndarray, length_thresh: int=25, offset: int=10) -> nx.DiGraph:
    directed_graph = nx.DiGraph()

    # Step 1: Find all entry and exit lanes (edges) in the graph, and cut them into a new directed graph
    removed_nodes = []
    edges = []
    nodes = []
    directed_edges = []
    directed_nodes = []
    for node in terminal_nodes:
        direction = node[-1] # 1 for inlet, 0 for outlet
        original_node = (node[0], node[1])
        if direction == 1:
            n1, n2, k = list(graph.edges(original_node, keys=True))[0]
            target_degree = graph.degree(n2)
        else:
            n2, n1, k = list(graph.edges(original_node, keys=True))[0]
            target_degree = graph.degree(n1)
        e = graph[n1][n2][k]
        path = graph_utils.correct_path_direction(e['path'], n1, n2)
        path_np = np.array(path)
        path_np = graph_utils.downsample_path(path_np, ratio=16)
        curve = cubic_spline.fit_cubic_spline(path_np[:, 0], path_np[:, 1])
        length_curve = curve[-1][-1]

        if target_degree <= 1: # if it connects to another terminal, move both to directed graph
            removed_nodes.append(n1)
            removed_nodes.append(n2)
            directed_edges.append((n1, n2, {'path': curve, 'd': length_curve}))
            directed_nodes.append((n1, {'yaw': curve[0][2], 'type': 'map_entry'}))
            directed_nodes.append((n2, {'yaw': curve[-1][2], 'type': 'map_exit'}))
        else: # if it connects to a normal node, intersect the edge near the end and move the front segment to directed graph
            removed_nodes.append(original_node)

            if len(curve) <= offset + 1:
                offset = len(curve) - 2

            if direction == 1:
                intersect_id = -(offset + 1)
                new_terminal = (round(curve[intersect_id][0], 1), round(curve[intersect_id][1], 1))
                curve_keep = curve[intersect_id:]
                curve_move = curve[:intersect_id + 1]
                length_keep = curve_keep[-1][-1] - curve_keep[0][-1]
                length_move = curve_move[-1][-1] - curve_move[0][-1]
                
                curve_keep_T = np.array(curve_keep).T
                path_keep = list(zip(curve_keep_T[0], curve_keep_T[1]))
                edges.append((new_terminal, n2, {'path': path_keep, 'd': length_keep}))
                directed_edges.append((n1, new_terminal, {'path': curve_move, 'd': length_move}))
                nodes.append((new_terminal, {'yaw': curve_move[-1][2], 'type': 'entry'}))
                directed_nodes.append((new_terminal, {'yaw': curve_move[-1][2], 'type': 'entry'}))
                directed_nodes.append((n1, {'yaw': curve_move[0][2], 'type': 'map_entry'}))
            else:
                intersect_id = offset
                new_terminal = (round(curve[intersect_id][0], 1), round(curve[intersect_id][1], 1))
                curve_keep = curve[:intersect_id + 1]
                curve_move = curve[intersect_id:]
                length_keep = curve_keep[-1][-1] - curve_keep[0][-1]
                length_move = curve_move[-1][-1] - curve_move[0][-1]
                
                curve_keep_T = np.array(curve_keep).T
                path_keep = list(zip(curve_keep_T[0], curve_keep_T[1]))
                edges.append((n1, new_terminal, {'path': path_keep, 'd': length_keep}))
                directed_edges.append((new_terminal, n2, {'path': curve_move, 'd': length_move}))
                nodes.append((new_terminal, {'yaw': curve_move[0][2], 'type': 'exit'}))
                directed_nodes.append((new_terminal, {'yaw': curve_move[0][2], 'type': 'exit'}))
                directed_nodes.append((n2, {'yaw': curve_move[-1][2], 'type': 'map_exit'}))

    graph.remove_nodes_from(removed_nodes)
    graph.add_edges_from(edges)
    graph.add_nodes_from(nodes)
    directed_graph.add_edges_from(directed_edges) # comment this line for only saving intersection graph
    directed_graph.add_nodes_from(directed_nodes)

    # Step 2: Find all long lanes (edges) in the graph, and cut them into a new directed graph
    removed_edges = []
    edges = []
    nodes = []
    directed_edges = []
    directed_nodes = []

    for n1, n2, k in list(graph.edges(keys=True)):
        e = graph[n1][n2][k]
        if e['d'] < length_thresh:
            continue
        
        start, end = determine_node_direction(graph, img_color, n1, n2)
        direction = start[-1]
        if direction == 1: # if the direction is correct
            path = graph_utils.correct_path_direction(e['path'], n1, n2)
            n1 = (start[0], start[1])
            n2 = (end[0], end[1])
        else: # if the direction is opposite
            path = graph_utils.correct_path_direction(e['path'], n2, n1)
            n1 = (end[0], end[1])
            n2 = (start[0], start[1])

        # Fit a curve 
        path_np = np.array(path)
        path_np = graph_utils.downsample_path(path_np, ratio=20)
        curve = cubic_spline.fit_cubic_spline(path_np[:, 0], path_np[:, 1])
        length_curve = curve[-1][-1]
        
        # Find the index to cut based on offset
        if len(curve) <= 2*offset + 1:
            logger.debug("Found a long edge but didn't cut")
            continue

        # Remove the original edge
        removed_edges.append((n1, n2, 0))
            
        new_n1_id = offset
        new_n2_id = -(offset + 1)

        new_n1 = (round(curve[new_n1_id][0], 1), round(curve[new_n1_id][1], 1))
        new_n2 = (round(curve[new_n2_id][0], 1), round(curve[new_n2_id][1], 1))

        curve1_keep = curve[:new_n1_id + 1]
        curve2_keep = curve[new_n2_id:]
        length_keep1 = curve1_keep[-1][-1] - curve1_keep[0][-1]
        length_keep2 = curve2_keep[-1][-1] - curve2_keep[0][-1]

        curve_move = curve[new_n1_id:new_n2_id + 1]
        length_move = curve_move[-1][-1] - curve_move[0][-1]
        
        curve1_keep_T = np.array(curve1_keep).T
        path1_keep = list(zip(curve1_keep_T[0], curve1_keep_T[1]))
        curve2_keep_T = np.array(curve2_keep).T
        path2_keep = list(zip(curve2_keep_T[0], curve2_keep_T[1]))
        edges.append((n1, new_n1, {'path': path1_keep, 'd': length_keep1}))
        edges.append((new_n2, n2, {'path': path2_keep, 'd': length_keep2}))
        directed_edges.append((new_n1, new_n2, {'path': curve_move, 'd': length_move}))
        nodes.append((new_n1, {'yaw': curve1_keep[-1][2], 'type': 'exit'}))
        nodes.append((new_n2, {'yaw': curve2_keep[0][2], 'type': 'entry'}))
        directed_nodes.append((new_n1, {'yaw': curve1_keep[-1][2], 'type': 'exit'}))
        directed_nodes.append((new_n2, {'yaw': curve2_keep[0][2], 'type': 'entry'}))

    graph.remove_edges_from(removed_edges)
    graph.add_edges_from(edges)
    graph.add_nodes_from(nodes)
    directed_graph.add_edges_from(directed_edges)
    directed_graph.add_nodes_from(directed_nodes)

    return graph, directed_graph


def connect_intersections(graph: nx.Graph, directed_graph: nx.DiGraph, simplified_graph: nx.Graph=None) -> nx.Graph:
    entries = [n for n in list(directed_graph.nodes()) if directed_graph.nodes[n]['type'] == 'entry']
    exits   = [n for n in list(directed_graph.nodes()) if directed_graph.nodes[n]['type'] == 'exit']

    # Add known conections at intersection from the simplified graph
    if simplified_graph is not None:
        simple_edges = []
        for n1, n2, k in list(simplified_graph.edges(keys=True)):
            
            try:
                n1_yaw = simplified_graph.nodes[n1]['yaw']
                n2_yaw = simplified_graph.nodes[n2]['yaw']
                n1_type = simplified_graph.nodes[n1]['type']
                n2_type = simplified_graph.nodes[n2]['type']
            except KeyError:
                try:
                    if directed_graph.has_node(n1) and directed_graph.has_node(n2):
                        n1_yaw = directed_graph.nodes[n1]['yaw']
                        n2_yaw = directed_graph.nodes[n2]['yaw']
                        n1_type = directed_graph.nodes[n1]['type']
                        n2_type = directed_graph.nodes[n2]['type']
                    else:
                        continue
                except KeyError:
                    continue
            
            if n1_type == 'entry' and n2_type == 'exit':
                curve = bezier_curve.fit_bezier_curve((n1[0], n1[1], n1_yaw), (n2[0], n2[1], n2_yaw))
                simple_edges.append((n1, n2, {'path': curve, 'd': curve[-1][-1]}))
            elif n2_type == 'entry' and n1_type == 'exit':
                curve = bezier_curve.fit_bezier_curve((n2[0], n2[1], n2_yaw), (n1[0], n1[1], n1_yaw))
                simple_edges.append((n2, n1, {'path': curve, 'd': curve[-1][-1]}))

        directed_graph.add_edges_from(simple_edges)

    # Fit unknown conections at intersection based on original graph
    edges = []
    for n1 in entries:
        n1_yaw = directed_graph.nodes[n1]['yaw']
        for n2 in exits:
            n2_yaw = directed_graph.nodes[n2]['yaw']

            # Verify if the entry and exit are connected in the original graph
            try: 
                route = nx.shortest_path(graph, n1, n2, weight='d', method='dijkstra')
            except nx.NetworkXNoPath:
                continue

            # Verify if the entry and exit are already connected in the directed graph
            if directed_graph.has_edge(n1, n2):
                continue

            # Verify if the connection is valid
            route_valid = True
            for node in route[1:-1]:
                if node in exits or node in entries:
                    route_valid = False
                    break

            if route_valid:
                waypoints = graph_utils.trace_route(graph, route)
                curve = bezier_curve.fit_bezier_curve((n1[0], n1[1], n1_yaw), (n2[0], n2[1], n2_yaw))
                # Test if the entry and exit will form a simple curve
                pos_angle = graph_utils.normalize_angle_rad(np.arctan2(n2[1] - n1[1], n2[0] - n1[0]) - n1_yaw)
                yaw_diff = graph_utils.normalize_angle_rad(n2_yaw - n1_yaw)
                if pos_angle < 0:
                    angle = -graph_utils.normalize_angle_rad(yaw_diff - pos_angle)
                else:
                    angle = graph_utils.normalize_angle_rad(yaw_diff - pos_angle)

                # If this is a direct connection
                if len(route) - 2 <= 1:
                    edges.append((n1, n2, {'path': curve, 'd': curve[-1][-1]}))
                # If this is a straight line
                # elif np.fabs(pos_angle) <= np.deg2rad(10) and np.fabs(yaw_diff) <= np.deg2rad(10):
                elif np.fabs(pos_angle) <= np.deg2rad(10) and np.fabs(angle) <= np.deg2rad(10):
                    edges.append((n1, n2, {'path': curve, 'd': curve[-1][-1]}))

                # If this is a short turn
                # elif distance(n1, n2) <= 50:
                #     if curve_is_valid(curve, waypoints, dist_tol=4.0, min_rate=0.5):
                #         edges.append((n1, n2, {'path': curve, 'd': curve[-1][-1]}))
                
                # Turn angle is too large
                elif np.fabs(yaw_diff) > np.deg2rad(135):
                    continue
                # This is a long turn
                elif np.deg2rad(-5) <= angle and angle <= np.deg2rad(95):
                    ratio = np.fabs(pos_angle / angle)
                    if 1/ratio < 2 and ratio < 2:
                        if curve_is_valid(curve, waypoints, dist_tol=3.0, min_rate=0.5):
                            edges.append((n1, n2, {'path': curve, 'd': curve[-1][-1]}))

    directed_graph.add_edges_from(edges)
    return directed_graph


def path_is_smooth(path: np.ndarray, yaw_d_thresh: float=500.0, yaw_dd_thresh: float=500.0) -> bool:
    _, idx = np.unique(path[:, 2], return_index=True, axis=0)
    path = path[np.sort(idx)]
    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    ds = np.hypot(dx, dy)
    yaw = np.rad2deg(np.arctan2(dy, dx))
    yaw_d = np.diff(yaw) / ds[:-1]
    yaw_dd = np.diff(yaw_d) / ds[:-2]

    if np.max(np.fabs(yaw_d)) > yaw_d_thresh:
        logger.debug(f'max yaw_d: {np.max(yaw_d):.1f}, threshold: {yaw_d_thresh} degree/m')
        return False
    # elif np.max(np.fabs(yaw_dd)) > yaw_dd_thresh:
    #     logger.debug(f'max yaw_dd: {np.max(yaw_dd):.1f}, threshold: {yaw_dd_thresh} degree/m^2')
    #     return False
    else:
        return True


def extract_polylines_from_img(img_color: Image, img_gray: Image=None, map_range: float=80, plot: bool=True, save_path: str=None) -> tuple:
    '''
    Return a list of 2D `np.ndarray`.
    Each array is of the shape [N, 5], containing attributes: [x, y, yaw, k, s]
    '''
    if img_gray is None:
        img_gray = image_utils.get_gray_image(img_color, plot=False)
        # img_gray.show()

    skel, graph = image_to_graph(img_gray)
    
    if plot:
        fig_pixels = 1080
        dpi = 100
        fig, axes = plt.subplots(3, 3, figsize=(fig_pixels/dpi, fig_pixels/dpi), sharex=True, sharey=True)
        axes = axes.ravel()
        for axe in axes:
            axe.set_aspect('equal')
            axe.margins(0)
            axe.invert_yaxis()
            axe.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
            axe.grid(False)
            # axe.axis("off")

    # 0. Visualize the nodes, colored by degrees of connectivity
    nodes = np.array([node for (node, degree) in graph.degree()], dtype=float)
    degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
    original_degrees = degrees
    logger.info(f'Original graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {original_degrees}')
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        logger.warning(f'Failed to extract graph from image')
        return None, None

    # 1. Find 1-degree terminals
    terminal_nodes, branching_nodes = find_key_nodes(graph, img_color)

    if terminal_nodes.shape[0] < 2 or len(terminal_nodes.shape) < 2:
        logger.warning(f'Failed to extract terminal nodes from image')
        return None, None

    if plot:
        try:
            axes[0].scatter(nodes[:, 0], nodes[:, 1], c=degrees, s=4)
            axes[0].title.set_text('Raw Nodes')
        
            polylines = graph_utils.graph_utils.graph_to_polylines(graph)
            for polyline in polylines:
                xs = polyline[:, 0]
                ys = polyline[:, 1]
                axes[1].plot(xs, ys, c=graph_utils.random_color())
            axes[1].title.set_text('All Edges')

            terminal_colors = ['lime' if node[-1] == 1 else 'red' for node in terminal_nodes]
            branching_colors = ['springgreen' if node[-1] == 1 else 'crimson' for node in branching_nodes]
            axes[2].quiver(terminal_nodes[:, 0], terminal_nodes[:, 1], 
                        terminal_nodes[:, 3]*10, terminal_nodes[:, 4]*10, 
                        color=terminal_colors, angles='xy', scale_units='xy', units='xy', scale=1)
            axes[2].quiver(branching_nodes[:, 0], branching_nodes[:, 1], 
                        branching_nodes[:, 3]*10, branching_nodes[:, 4]*10, 
                        color=branching_colors, angles='xy', scale_units='xy', units='xy', scale=1)
            axes[2].title.set_text('Key Nodes')
        except:
            logger.warning(f'The original graph contains too few nodes')

    #################################### Find Intersection Nodes ####################################
    graph, directed_graph = find_intersections(graph, img_color, terminal_nodes, offset=5)

    if plot:
        # Plot all graph edges
        degrees = np.array([degree for (node, degree) in directed_graph.degree()], dtype=float)
        logger.info(f'Directed graph: {directed_graph.number_of_nodes()} nodes, {directed_graph.number_of_edges()} edges, degrees: {degrees}')
        polylines = graph_utils.graph_to_polylines(directed_graph)
        for polyline in polylines:
            xs = polyline[:, 0]
            ys = polyline[:, 1]
            axes[3].plot(xs, ys, c=graph_utils.random_color())
        axes[3].title.set_text('Directed Graph')

        degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
        logger.info(f'Undirected graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {degrees}')
        polylines = graph_utils.graph_to_polylines(graph)
        for polyline in polylines:
            xs = polyline[:, 0]
            ys = polyline[:, 1]
            axes[4].plot(xs, ys, c=graph_utils.random_color())
        axes[4].title.set_text('Undirected Graph')


    #################################### Graph Simplification ####################################
    # 6. Simplify the graph
    simplified_graph = copy.deepcopy(graph)
    simplified_graph = simplify_graph(simplified_graph, img_gray=img_gray, plot=False)

    if plot:
        try:
            nodes = np.array([node for (node, degree) in graph.degree()], dtype=float)
            degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
            logger.info(f'Reduced graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {degrees}')
            axes[5].scatter(nodes[:, 0], nodes[:, 1], c=degrees, s=4)
            axes[5].title.set_text('Reduced Graph')
            polylines = graph_utils.graph_to_polylines(graph)
            for polyline in polylines:
                xs = polyline[:, 0]
                ys = polyline[:, 1]
                axes[5].plot(xs, ys, c=graph_utils.random_color())
        except:
            logger.warning(f'Reduced graph contains too few nodes: {graph.number_of_nodes()}')

    #################################### Graph Breakdown ####################################
    simplified_graph = break_down_graph(simplified_graph)
    
    if plot:
        # 8. Plot all graph edges
        degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
        logger.info(f'Broken down graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {degrees}')
        polylines = graph_utils.graph_to_polylines(graph)
        for polyline in polylines:
            xs = polyline[:, 0]
            ys = polyline[:, 1]
            axes[6].plot(xs, ys, c=graph_utils.random_color())
        axes[6].title.set_text('Brokendown Graph')

    #################################### Fit Intersection Lanes ####################################
    # directed_graph = fit_intersection_paths(graph, directed_graph)
    
    # if plot:
    #     # 7. Plot all graph edges
    #     degrees = np.array([degree for (node, degree) in directed_graph.degree()], dtype=float)
    #     logger.info(f'Directed Graph: {directed_graph.number_of_nodes()} nodes, {directed_graph.number_of_edges()} edges, degrees: {degrees}')
    #     polylines = graph_utils.graph_to_polylines(directed_graph)
    #     for polyline in polylines:
    #         xs = polyline[:, 0]
    #         ys = polyline[:, 1]
    #         axes[7].plot(xs, ys, c=graph_utils.random_color())
    #     axes[7].title.set_text('Fitted Intersections')

     #################################### Connect Intersections ####################################
    directed_graph = connect_intersections(graph, directed_graph, simplified_graph=simplified_graph)
    if plot:
        try:
            nodes = np.array([node for (node, degree) in directed_graph.degree()], dtype=float)
            degrees = np.array([degree for (node, degree) in directed_graph.degree()], dtype=float)
            logger.info(f'Connected graph: {directed_graph.number_of_nodes()} nodes, {directed_graph.number_of_edges()} edges, degrees: {degrees}')
            axes[-1].scatter(nodes[:, 0], nodes[:, 1], c=degrees, s=4)
            axes[-1].title.set_text('Connected Graph')
            polylines = graph_utils.graph_to_polylines(directed_graph)
            for polyline in polylines:
                xs = polyline[:, 0]
                ys = polyline[:, 1]
                axes[-1].plot(xs, ys, c=graph_utils.random_color())
        except:
            logger.warning(f'Connected graph contains too few nodes: {directed_graph.number_of_nodes()}')

    #################################### Convert to World Frame ####################################
    polylines = graph_utils.graph_to_polylines(directed_graph)
    polylines_world = graph_utils.polylines_to_world_frame(polylines, skel.shape, map_range=map_range)
    output = graph_utils.polylines_to_output(polylines_world)

    if plot:
        if save_path is not None:
            fig.savefig(save_path, transparent=True, dpi=dpi*5, format="png")
        else:
            plt.show()

    return output, directed_graph
