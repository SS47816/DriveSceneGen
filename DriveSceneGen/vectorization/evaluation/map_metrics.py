import os
import math

import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm
from scipy.stats import norm, wasserstein_distance


def frechet_distance_univariate(mu_x: float, sigma_x: float, mu_y: float, sigma_y: float) -> float:
    a = abs(mu_x - mu_y)
    b = math.sqrt(sigma_x ** 2 + sigma_y ** 2)
    c = math.sqrt(2 * sigma_x * sigma_y) * math.exp(-0.5 * ((mu_x - mu_y) / math.sqrt(sigma_x ** 2 + sigma_y ** 2)) ** 2)
    
    return a + b - c

def frechet_distance(mu_x: np.ndarray, sigma_x: np.ndarray, mu_y: np.ndarray, sigma_y: np.ndarray) -> float:
    a = (mu_x - mu_y).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = np.linalg.eigvals(np.matmul(sigma_x, sigma_y)).sqrt().real.sum()
    return a + b - 2 * c

def gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    pairwise_sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
    K = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    
    return K

def mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0, dist_function: str = None) -> float:
    K_XX = gaussian_kernel(X, X, sigma)
    K_YY = gaussian_kernel(Y, Y, sigma)
    
    mean_x = np.mean(K_XX, axis=0)
    mean_y = np.mean(K_YY, axis=0)
    
    if dist_function == 'wasserstein':
        mmd_value = wasserstein_distance(mean_x, mean_y)
    elif dist_function == 'tvd':
        mmd_value = 0.5*np.linalg.norm(mean_x - mean_y, ord=1)
    else:
        K_XY = gaussian_kernel(X, Y, sigma)
        K_YX = gaussian_kernel(Y, X, sigma)
        mmd_value = np.mean(K_XX) + np.mean(K_YY) - np.mean(K_XY) - np.mean(K_YX)
    
    return mmd_value


def transform_to_world_frame(graph: nx.Graph, map_range: float = 80.0, map_res: int = 256) -> nx.Graph:
    scale = map_range/map_res # m/pixel
    center = (map_res/2 * scale, map_res/2 * scale)
    
    edges = list(graph.edges)
    new_edges = []
    new_nodes = []
    for n1, n2 in edges:
        d = graph[n1][n2]['d']
        new_dist = d * scale
        new_n1 = (n1[0] * scale - center[0], center[1] - n1[1] * scale)
        new_n2 = (n2[0] * scale - center[0], center[1] - n2[1] * scale)
        new_n1_yaw = -graph.nodes[n1]['yaw']
        new_n2_yaw = -graph.nodes[n2]['yaw']
        new_edges.append((new_n2, n2, {'dist': new_dist}))
        new_nodes.append((new_n1, {'yaw': new_n1_yaw}))
        new_nodes.append((new_n2, {'yaw': new_n2_yaw}))
        
    new_graph = nx.Graph()
    new_graph.add_edges_from(new_edges)
    new_graph.add_nodes_from(new_nodes)
    
    return new_graph


def compute_stats(graph: nx.Graph, map_range: float = 80.0, map_res: int = 256) -> np.ndarray:
    if None not in (map_range, map_res):
        graph = transform_to_world_frame(graph, map_range=map_range, map_res=map_res)
        area = map_range*map_range
    else:
        nodes_np = np.array([node for (node, degree) in graph.degree()])
        x = np.max(nodes_np[:, 0]) - np.min(nodes_np[:, 0])
        y = np.max(nodes_np[:, 1]) - np.min(nodes_np[:, 1])
        area = x*y
        
    nodes = [node for (node, degree) in graph.degree()]
    degrees = [degree for (node, degree) in graph.degree()]
    edges = list(graph.edges())
    
    n_nodes = len(nodes)
    n_edges = len(edges)

    if n_nodes < 2:
        distances = [0.0]
    else:
        distances = []
        for i, n1 in enumerate(nodes[:-1]):
            for j, n2 in enumerate(nodes[i+1:]):
                try: 
                    distance, path = nx.single_source_dijkstra(graph, source=n1, target=n2, cutoff=None, weight='dist')
                except nx.NetworkXNoPath:
                    continue
                distances.append(distance)
    
    # Urban planning
    connectivity = np.mean(degrees)
    density = n_nodes #/area
    reach = n_edges #/area         
    convenience = np.mean(distances)
    
    # Geometry fidelity
    lengths = [dist for dist in nx.get_edge_attributes(graph, 'dist').values()]
    orientations = [yaw for yaw in nx.get_node_attributes(graph, 'yaw').values()]
    length = np.mean(lengths)
    orientation = np.mean(orientations)
    
    # Topology fidelity
    degree = np.mean(degrees)
    spectrum = np.sum(nx.laplacian_spectrum(graph, weight='dist'))
    # spectrum = np.sum(nx.laplacian_spectrum(graph))
    
    urban_plan = np.array([connectivity, density, reach, convenience])
    geo = np.array([length, orientation])
    topo = np.array([degree, spectrum])
    
    return urban_plan, geo, topo


def compute_map_stats(files, save_path: str, map_range: float = None, map_res: int = None) -> None:
    stats_names = [
        'Connectivity',
        'Density',
        'Reach',
        'Convenience',
        'Length',
        'Orientation',
    ]
    
    urban_plans = []
    geos = []
    topos = []
    
    for file in tqdm(files):
        graph = pickle.load(open(file, 'rb'))
        urban_plan, geo, topo = compute_stats(graph, map_range=map_range, map_res=map_res)
        urban_plans.append(urban_plan)
        geos.append(geo)
        topos.append(topo)
    
    # urban_plans_np = np.vstack(urban_plans)
    # geos_np = np.vstack(geos)
    urban_plan_geo_np = np.hstack((np.vstack(urban_plans), np.vstack(geos)))
    topos_np = np.vstack(topos)
    
    # Compute Statistics
    stats = []
    for i, data in enumerate(urban_plan_geo_np.T):
        mu, std = norm.fit(data)
        stats.append((mu, std))
        print(f'{stats_names[i]}: mu = {mu}, std = {std}')
    stats_np = np.array(stats)
    
    # Topology Degree
    degrees = topos_np[:, 0].ravel()
    print(f'Topo Degree: {degrees}')
    # Topology Spectrum
    spectrum = topos_np[:, 1].ravel()
    print(f'Topo Spectrum: {spectrum}')
    
    np.save(os.path.join(save_path, 'stats.npy'), stats_np)
    np.save(os.path.join(save_path, 'degrees.npy'), degrees)
    np.save(os.path.join(save_path, 'spectrum.npy'), spectrum)
    
    return


def compute_map_metrics(gt_stats, gt_degrees, gt_spectrum, gen_stats, gen_degrees, gen_spectrum) -> None:
    print(f'gt_stats: {gt_stats}')
    print(f'gt_degrees: {gt_degrees}')
    print(f'gt_spectrum: {gt_spectrum}')
    print(f'gen_stats: {gen_stats}')
    print(f'gen_degrees: {gen_degrees}')
    print(f'gen_spectrum: {gen_spectrum}')
    
    fds = []
    for gt, gen in zip(gt_stats, gen_stats):
        fd = frechet_distance_univariate(gt[0], gt[1], gen[0], gen[1]) # np.abs(gt[0] - gen[0])
        fds.append(fd)
    fds_np = np.array(fds)
    print(f'fds_np: {fds_np}')
    
    # print(gt_degrees.shape, gen_degrees.shape)
    mmd_degrees = mmd(gt_degrees.reshape(-1, 1), gen_degrees.reshape(-1, 1), dist_function='wasserstein')
    print(f'mmd_degrees: {mmd_degrees}')
    
    # print(gt_spectrum.shape, gen_spectrum.shape)
    mmd_spectrum = mmd(gt_spectrum.reshape(-1, 1), gen_spectrum.reshape(-1, 1), dist_function='wasserstein')
    print(f'mmd_spectrum: {mmd_spectrum}')
    
    return

def plot_agent_histogram(all_agents: np.ndarray, save_path: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.ravel()
    for i, data in enumerate(all_agents.T):
        ax = axes[i]
        ax.hist(data, bins=21, alpha=0.5, label=f'{i+1}')
        ax.set_title(f'max:{np.max(data)}, min:{np.min(data)}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    
    return

def compute_agent_stats(agent_files: list, metrics_dir: str) -> np.ndarray:
    all_agents = []
    for file in tqdm(agent_files):
        agents = np.load(file) # [N, 9], [x, y, z, l, w, h, heading, v_x, v_y]
        n = agents.shape[0]
        if n == 0:
            continue
        # all_agents.append(np.append(np.mean(agents, axis=0), n))
        # all_agents.append(agents)
        all_agents.append(np.mean(agents, axis=0))
            
    all_agents_np = np.vstack(all_agents)
    np.save(os.path.join(metrics_dir, 'agents.npy'), all_agents_np)
    plot_agent_histogram(all_agents_np, os.path.join(metrics_dir, 'agents.png'))
    
    # return np.mean(all_agents_np, axis=0)
    return all_agents_np # [M, 9], [x, y, z, l, w, h, heading, v_x, v_y]


def compute_track_stats(track_files: list, metrics_dir: str) -> np.ndarray:
    all_tracks = []
    for file in tqdm(track_files):
        track_dict = pickle.load(open(file, 'rb'))
        # object_ids = track_dict['object_id']
        # object_types = track_dict['object_type'] # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        trajs = track_dict['trajs'] # (num_objects, 91, 10) # [x, y, z, l, w, h, heading, v_x, v_y, valid, type]
        # track_indices = track_dict['track_index']
        # print(trajs.shape)
        
        if trajs.shape[0] == 0:
            continue
        
        sdc_id = 0
        curr_trajs = trajs[:, 10, :]
        curr_valid = trajs[:, 10, -2]
        curr_type = trajs[:, 10, -1]
        
        # Keep only valid vehicle
        curr_trajs = curr_trajs[np.logical_and(curr_valid, curr_type)]
        # Transform to local coordinates centered around sdc
        curr_trajs[:, :3] = curr_trajs[:, :3] - curr_trajs[sdc_id, :3]
        # print(f"totally {n} agents, filtered to {curr_trajs.shape[0]} valid vehicles")

        all_tracks.append(np.mean(curr_trajs[:, :9], axis=0))
        
    all_tracks_np = np.vstack(all_tracks) # [M, 9], [x, y, z, l, w, h, heading, v_x, v_y]
    np.save(os.path.join(metrics_dir, 'agents.npy'), all_tracks_np)
    plot_agent_histogram(all_tracks_np, os.path.join(metrics_dir, 'agents.png'))
    
    # means = np.mean(all_tracks_np, axis=0)
    # maxs = np.max(all_tracks_np, axis=0)
    # mins = np.min(all_tracks_np, axis=0)
    # print(f'means: {means}')
    # print(f'maxs: {maxs}')
    # print(f'mins: {mins}')

    return all_tracks_np # [M, 9], [x, y, z, l, w, h, heading, v_x, v_y]
