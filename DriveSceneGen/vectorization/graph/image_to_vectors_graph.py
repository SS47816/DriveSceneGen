import logging
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from DriveSceneGen.vectorization.utils import image_utils, graph_utils
from DriveSceneGen.vectorization.curve import cubic_polynomial, cubic_spline, straight_line
from DriveSceneGen.vectorization.graph import extract_network


def image_to_graph(img: Image) -> tuple:
    rgb = (255, 255, 255)
    px = extract_network.find_color(img, rgb).T
    return extract_network.extract_network(px, min_distance=4)


def find_terminal_nodes(graph: nx.Graph) -> list:
    nodes_terminal = [(node[0], node[1]) for (node, degree) in graph.degree if degree == 1]
    return nodes_terminal


def find_branching_nodes(graph: nx.Graph, nodes_terminal: list) -> list:
    nodes_branching = []
    for n1 in nodes_terminal:
        n1, n1_neighbour, k = list(graph.edges(n1, keys=True))[0]
        nodes_branching.append(n1_neighbour)
    
    return nodes_branching


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
                n0_yaw, n0_delta = estimate_path_front_yaw(e_path, 10)
                yaws.append(n0_yaw)
                nodes.append(n1)

        nl_id = nodes.index(nl)
        nr_id = nodes.index(nr)
        votes, connect_matrix = voting_by_yaw_angle(yaws)
        if connect_matrix[nl_id, nr_id] == False:
            return False
        elif calc_path_yaw_diff(yaws[nl_id], yaws[nr_id]) >= np.pi/4:
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

    logging.debug(f'Found {len(routes)} paths')
    return routes, waypoints_all


def normalize_angle_rad(angle) -> float:
    # normalize angle to (-pi, pi]
    while angle > np.pi:
        angle = angle - 2*np.pi
    while angle <= -np.pi:
        angle = angle + 2*np.pi
    return angle


def calc_path_yaw_diff(yaw1: float, yaw2: float) -> float:
    return np.fabs(normalize_angle_rad(yaw1 - yaw2 - np.pi))


def estimate_path_front_yaw(path: list, local_length: int=10) -> tuple:
    path_np = np.array(path)

    if path_np.shape[0] > local_length:
        front_delta = path_np[local_length-1] - path_np[0]
        rear_delta = path_np[-local_length] - path_np[-1]
    else:
        front_delta = path_np[-1] - path_np[0]
        rear_delta = path_np[0] - path_np[-1]

    front_delta = graph_utils.normalize_dx_dy(front_delta[0], front_delta[1])
    rear_delta = graph_utils.normalize_dx_dy(rear_delta[0], rear_delta[1])
    front_yaw = np.arctan2(front_delta[1], front_delta[0])    
    rear_yaw = np.arctan2(rear_delta[1], rear_delta[0])

    # return front_yaw, rear_yaw
    return front_yaw, front_delta


def voting_by_yaw_angle(yaws) -> tuple:
    votes = np.zeros(len(yaws), dtype=int)
    connect_matrix = np.zeros((len(yaws), len(yaws)), dtype=bool)

    for i, yaw1 in enumerate(yaws):
        diffs = []
        for j, yaw2 in enumerate(yaws):
            if i == j:
                diffs.append(2*np.pi)
            else:
                diff = calc_path_yaw_diff(yaw1, yaw2) # both angles are pointing away from n0
                diffs.append(diff) 
        
        # Pair the best matches
        min_id = np.argmin(diffs) # find the minimum angle difference
        votes[min_id] = votes[min_id] + 1
        connect_matrix[i, min_id] = True
        connect_matrix[min_id, i] = True
    
    # logging.debug(f'yaws: {yaws}')
    # logging.debug(f'votes: {votes}')
    # logging.debug(f'connect_matrix: \n{connect_matrix}')

    return votes, connect_matrix


def reduce_graph(graph: nx.Graph, plot: bool=False) -> nx.Graph:
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
                if e1_path:
                    n0_yaw, n0_delta = estimate_path_front_yaw(e1_path, 10)
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
                n0_new = paths[i][1]
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
                    new_path = [n0_new] + graph_utils.correct_path_direction(paths[j], n0, n2)
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


def smoothen_graph_edges(graph: nx.Graph, length_thresh: int=20, step: int=1) -> nx.Graph:
    edges = []
    for n1, n2, k in list(graph.edges(keys=True)):
        e = graph[n1][n2][k]
        path = np.array(graph_utils.correct_path_direction(e['path'], n1, n2)) # paths start from n1
        if path.shape[0] <= length_thresh:
            curve = straight_line.fit_straight_line(path[:, 0], path[:, 1], step=step)
        else:
            curve = cubic_polynomial.fit_cubic_polynomial(path[:, 0], path[:, 1], step=step)

        edges.append((n1, n2, {'path': curve, 'd': curve[-1][-1]}))

    new_graph = nx.MultiGraph()
    new_graph.add_edges_from(edges)

    nodes = []
    for n, degree in graph.degree:
        if degree < 2:
            node_type = 'terminal'
        else:
            node_type = 'branch'
        nodes.append((n, {'type': node_type}))

    new_graph.add_nodes_from(nodes)

    return new_graph


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

            if degree < 2 or node_type == 'terminal':
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
                n0_yaw = e1_path[0][2]
                # n0_yaw, n0_delta = estimate_path_front_yaw(e1_path, 10)
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
                    # new_path = path1 + [(*(pt[:-1]), path1[-2][-1]+pt[-1]) for pt in path2]
                    new_path = graph_utils.join_paths(path1, path2)
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
        logging.info(f'max yaw_d: {np.max(yaw_d):.1f}, threshold: {yaw_d_thresh} degree/m')
        return False
    # elif np.max(np.fabs(yaw_dd)) > yaw_dd_thresh:
    #     logging.info(f'max yaw_dd: {np.max(yaw_dd):.1f}, threshold: {yaw_dd_thresh} degree/m^2')
    #     return False
    else:
        return True


def verify_final_graph(graph: nx.Graph, inlets: np.ndarray, outlets: np.ndarray, thresh: float=90.0) -> nx.Graph:
    inlets_T = inlets.T.astype(int)
    outlets_T = outlets.T.astype(int)
    inlets_list = list(zip(inlets_T[0], inlets_T[1]))
    outlets_list = list(zip(outlets_T[0], outlets_T[1]))
    
    new_graph = nx.MultiDiGraph()

    for n1, n2, k in list(graph.edges(keys=True)):
        e = graph[n1][n2][k]
        if n1 in inlets_list:
            if n2 in outlets_list:
                path = e['path']
                new_path = graph_utils.correct_path_direction(path, n1, n2)
                new_graph.add_edge(n1, n2, path=new_path, d=new_path[-1][-1])
            else:
                logging.info(f'Invalid path from {n1} to {n2}, both inlets')
                # graph.remove_edge(n1, n2, key=k)
                continue

        elif n1 in outlets_list:
            if n2 in inlets_list:
                path = e['path']
                new_path = graph_utils.correct_path_direction(path, n2, n1)
                new_graph.add_edge(n2, n1, path=new_path, d=new_path[-1][-1])
            else:
                logging.info(f'Invalid path from {n1} to {n2}, both outlets')
                # graph.remove_edge(n1, n2, key=k)
                continue

        else:
            continue
            # path = np.array(graph_utils.correct_path_direction(e['path'], n1, n2)) # paths start from n1
            # if not path_is_smooth(path):
            #     graph.remove_edge(n1, n2, key=k)
            #     continue

    return new_graph


def extract_polylines_from_img(img_color: Image, img_gray: Image=None, map_range: float=80, plot: bool=True, save_path: str=None) -> np.ndarray:
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
        axes[0].imshow(skel.T, cmap='gray')
        axes[1].imshow(skel.T, cmap='gray')
        axes[2].imshow(skel.T, cmap='gray')
        axes[3].imshow(img_color)
        axes[4].imshow(img_gray)
        axes[5].imshow(img_gray)
        axes[6].imshow(img_gray)
        axes[7].imshow(img_gray)
        axes[8].imshow(img_gray)

        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')
        axes[2].set_aspect('equal')
        axes[3].set_aspect('equal')
        axes[4].set_aspect('equal')
        axes[5].set_aspect('equal')

    # 0. Visualize the nodes, colored by degrees of connectivity
    nodes = np.array([node for (node, degree) in graph.degree()], dtype=float)
    degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
    original_degrees = degrees
    logging.info(f'Orignal graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {original_degrees}')
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        logging.warning(f'Failed to extract graph from image')
        return []

    # 1. Find 1-degree terminals
    nodes_1_degree = find_terminal_nodes(graph)
    nodes_1_degree_np = np.array(nodes_1_degree)
    
    # 2. Fix small gaps in the orignal graph
    graph = graph_utils.connect_small_gaps(graph, nodes_1_degree, thresh=8)
    nodes_terminal = find_terminal_nodes(graph)
    nodes_branching = find_branching_nodes(graph, nodes_terminal)
    nodes_terminal_np = np.array(nodes_terminal)
    nodes_branching_np = np.array(nodes_branching)
    
    if plot:
        axes[0].scatter(nodes[:, 0], nodes[:, 1], c=degrees, s=1)
        axes[0].scatter(nodes_1_degree_np[:, 0], nodes_1_degree_np[:, 1], c='red', s=3)
        axes[0].title.set_text('Raw Nodes')
        axes[1].scatter(nodes_terminal_np[:, 0], nodes_terminal_np[:, 1], c='red', s=3)
        axes[1].scatter(nodes_branching_np[:, 0], nodes_branching_np[:, 1], c='blue', s=3)
        axes[1].title.set_text('Key Nodes')
        polylines = graph_utils.graph_to_polylines(graph)
        colors = graph_utils.generate_random_colors(len(polylines))
        for i, polyline in enumerate(polylines):
            xs = polyline[:, 0]
            ys = polyline[:, 1]
            axes[2].plot(xs, ys, c=colors[i])
        axes[2].title.set_text('All Edges (Gaps fixed)')

    # 3. Find 1-degree terminals after fixing the gaps
    nodes_terminal = find_terminal_nodes(graph)
    inlets, outlets = graph_utils.find_node_directions(graph, nodes_terminal, img_color)
    logging.info(f'Fixed graph: found {inlets.shape[0]} inlets, {outlets.shape[0]} outlets')
    
    if plot:
        axes[3].quiver(inlets[:, 0], inlets[:, 1], inlets[:, 2], inlets[:, 3], color='r', angles='xy', scale_units='xy', scale=0.1)
        axes[3].quiver(outlets[:, 0], outlets[:, 1], outlets[:, 2], outlets[:, 3], color='g', angles='xy', scale_units='xy', scale=0.1)
        axes[3].title.set_text('Terminal Inlets and Outlets')

        # 4. Plot paths between terminals
        paths, path_waypoints = find_paths_among_terminals(graph, inlets, outlets)
        for path in paths:
            path = np.array(path, dtype=float)
            xs = path[:, 0]
            ys = path[:, 1]
            axes[4].plot(xs, ys)
            axes[4].title.set_text('Paths between Inlets and Outlets')

    #################################### Graph Reduction ####################################

    # 6. Simplify the graph
    graph = reduce_graph(graph)
    
    if plot:
        nodes = np.array([node for (node, degree) in graph.degree()], dtype=float)
        degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
        axes[5].scatter(nodes[:, 0], nodes[:, 1], c=degrees, s=1)
        axes[5].title.set_text('Reduced Graph nodes')
        logging.info(f'Reduced graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {degrees}')

        # 6. Find 1-degree terminals
        nodes_terminal = find_terminal_nodes(graph)
        inlets, outlets = graph_utils.find_node_directions(graph, nodes_terminal, img_color)
        logging.info(f'found {inlets.shape[0]} inlets, {outlets.shape[0]} outlets')
        axes[5].quiver(inlets[:, 0], inlets[:, 1], inlets[:, 2], inlets[:, 3], color='r', angles='xy', scale_units='xy', scale=0.1)
        axes[5].quiver(outlets[:, 0], outlets[:, 1], outlets[:, 2], outlets[:, 3], color='g', angles='xy', scale_units='xy', scale=0.1)
        axes[5].title.set_text('Reduced Graph Terminals')

    #################################### Polyline Fitting ####################################
    graph = smoothen_graph_edges(graph, length_thresh=20, step=1)
    
    if plot:
        # 7. Plot all graph edges
        degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
        logging.info(f'Smoothened graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {degrees}')
        polylines = graph_utils.graph_to_polylines(graph)
        colors = graph_utils.generate_random_colors(len(polylines))
        for i, polyline in enumerate(polylines):
            xs = polyline[:, 0]
            ys = polyline[:, 1]
            axes[6].plot(xs, ys, c=colors[i])
        axes[6].title.set_text('Reduced & Smoothened Graph')

    #################################### Graph Breakdown ####################################
    graph = break_down_graph(graph)
    
    if plot:
        # 8. Plot all graph edges
        degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
        logging.info(f'Broken down graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {degrees}')
        polylines = graph_utils.graph_to_polylines(graph)
        colors = graph_utils.generate_random_colors(len(polylines))
        for i, polyline in enumerate(polylines):
            xs = polyline[:, 0]
            ys = polyline[:, 1]
            axes[7].plot(xs, ys, c=colors[i])
        axes[7].title.set_text('Brokendown Graph')

    #################################### Graph Verification ####################################
    graph = verify_final_graph(graph, inlets, outlets)

    if plot:
        # 8. Plot all graph edges
        degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
        logging.info(f'Verified graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degrees: {degrees}')
        polylines = graph_utils.graph_to_polylines(graph)
        colors = graph_utils.generate_random_colors(len(polylines))
        for i, polyline in enumerate(polylines):
            xs = polyline[:, 0]
            ys = polyline[:, 1]
            axes[8].plot(xs, ys, c=colors[i])
        axes[8].title.set_text('Verified Graph')

    #################################### Convert to World Frame ####################################
    polylines = graph_utils.graph_to_polylines(graph)
    polylines_world = graph_utils.polylines_to_world_frame(polylines, skel.shape, map_range=map_range)
    output = graph_utils.polylines_to_output(polylines_world)

    if plot:
        if save_path is not None:
            fig.savefig(save_path, transparent=True, dpi=dpi*5, format="png")
        else:
            plt.show()

    return output, graph
