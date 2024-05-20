import glob
import logging
import multiprocessing
import os
import argparse
import yaml

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from DriveSceneGen.utils.io import get_logger
from DriveSceneGen.utils.render import render_vectorized_scenario_on_axes
from DriveSceneGen.vectorization.direct.extract_vehicles import extract_agents
from DriveSceneGen.vectorization.graph import image_to_polylines, image_to_vectors_graph

logger = get_logger('vectorization', logging.WARNING)


def vectorize(img_color: Image, method: str = "GRAPH_FIT", map_range: float = 80.0, plot: bool = True, pic_save_path: str = None) -> tuple:
    """
    Returns
    ---
    lanes: `list` [lane1, lane2, ...]
        lane: `list` [point1, point2, ...] (follow the sequence of the traffic flow)
            point: `list` [x, y, z, dx, dy, dz]
    agents: `list` [agent1, agent2, ...]
        agent: `list` [center_x, center_y, center_z, length, width, height, angle, velocity_x, velocity_y]
    """
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_color)

    try:
        # Extract centerlines
        if method == "GRAPH":
            lanes, graph = image_to_vectors_graph.extract_polylines_from_img(img_color, map_range=map_range, plot=plot, save_path=pic_save_path)
        
        elif method == "GRAPH_FIT":
            lanes, graph = image_to_polylines.extract_polylines_from_img(img_color, map_range=map_range, plot=plot, save_path=pic_save_path)
        
        elif method == "SEARCH":
            # TODO: Implement this method
            pass

        elif method == "DETR":
            # TODO: Implement this method
            pass

        else:
            print("Unknown method, Vectorization failed")
            return None, None, None, None
    
    except ValueError:
        logger.warning(f'Could not extract polylines from img')
        return None, None, None
    
    # Extract agents' properties
    agents = extract_agents(img_tensor, lanes)

    fig, axes = plt.subplots(1, 3)
    dpi = 100
    size_inches = 800 / dpi
    fig.set_size_inches([3*size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_tight_layout(True)
    fig.set_facecolor("azure")  # 'azure', '#FFF5E0', 'lightcyan', 'xkcd:grey'
    axes = axes.ravel()
    axes[0].imshow(img_color)
    axes[0].set_aspect("equal")
    axes[0].margins(0)
    axes[0].grid(False)
    axes[0].axis("off")
    axes[1] = render_vectorized_scenario_on_axes(
        axes[1], lanes, [], map_range=map_range
    )
    axes[2] = render_vectorized_scenario_on_axes(
        axes[2], [], agents, map_range=map_range
    )
    
    return lanes, graph, agents, fig


def multiprocessing_func(data_files: list, cfg: dict, vectorized_dir: str, picture_dir: str, graph_dir: str, agent_dir: str, n_proc: int, proc_id: int):
    for index, file in enumerate(tqdm(data_files)):
        with open(file, "rb") as f:
            img_id = index*n_proc + proc_id
            img_color = Image.open(f).convert("RGB")

            # Vectorization
            vec_save_path = f'{vectorized_dir}/{img_id}.pkl'
            pic_save_path = f'{picture_dir}/{img_id}_process.png'
            graph_save_path = f'{graph_dir}/{img_id}_graph.pickle'
            agent_save_path = f'{agent_dir}/{img_id}_agents.npy'
            
            try:
                lanes, graph, agents, fig = vectorize(img_color, method=cfg['method'], map_range=cfg['map_range'], plot=cfg['plot'], pic_save_path=pic_save_path)
                
                if fig is not None:
                    fig.savefig(f'{picture_dir}/{img_id}.png', transparent=True, format="png")
                plt.close()
                
                if graph is not None:
                    try:
                        with open(graph_save_path, "wb") as f:
                            pickle.dump(graph, f)
                    except ValueError:
                        logger.error(f'Failed to save graph!')
                        continue
                
                if agents is not None and lanes is not None:
                    np.save(agent_save_path, np.array(agents))
                
            except OSError:
                logger.warning(f'File no. {img_id}: failed to be vectorized due to insufficient memory!')
                plt.close()
                break
            except Exception as e:
                logger.warning(f'File no. {img_id} failed to be vectorized due to {e}')
                plt.close()
                continue
            
            ## save the scenario
            output_dict = {}
            output_dict["scenario_id"] = index
            output_dict["sdc_track_index"] = 0
            output_dict["object_type"] = np.ones((len(agents)))
            output_dict["all_agent"] = agents
            output_dict["lane"] = lanes

            torch.save(output_dict, vec_save_path)
            
    return


def chunks(input, n):
    """Yields successive n-sized chunks of input"""
    for i in range(0, len(input), n):
        yield input[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vectorization')
    parser.add_argument('--load_path',default="./data/preprocessed", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="./data/rasterized/",type=str, help='path to save processed data')
    parser.add_argument('--cfg_file', default="./DriveSceneGen/config/vectorization.yaml",type=str, help='path to cfg file')
    
    sys_args = parser.parse_args()
    with open(sys_args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
    n_proc = cfg['n_proccess']
    map_range = cfg['vectoriztion']['map_range']
    
    # Generated Dataset Paths
    input_dir = f'./data/generated_{map_range}m_5k'
    generated_imgs_dir = os.path.join(input_dir, 'diffusion')
    outputs_dir = input_dir
    
    vectorized_output_dir = os.path.join(outputs_dir, "vectorized")
    vectorized_picture_dir = os.path.join(outputs_dir, "vectorized_pics")
    graph_dir = os.path.join(outputs_dir, "graph")
    agent_dir = os.path.join(outputs_dir, "agent")
    os.makedirs(vectorized_output_dir, exist_ok=True)
    os.makedirs(vectorized_picture_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)

    ## get the list of all the files
    all_files = glob.glob(generated_imgs_dir + "/*")

    ## split the input files into n_proc chunks
    chunked_files = list(chunks(all_files, int(len(all_files) / n_proc) + 1))

    # Initialize the parallel processes list
    processes = []
    for proc_id in np.arange(n_proc):
        """Execute the target function on the n_proc target processors using the splitted input"""
        p = multiprocessing.Process(
            target=multiprocessing_func,
            args=(chunked_files[proc_id], cfg['vectoriztion'], vectorized_output_dir, vectorized_picture_dir, graph_dir, agent_dir, n_proc, proc_id),
        )
        processes.append(p)
        p.start()
    for process in processes:
        process.join()

    print(f"Process finished!!!, results saved to: {vectorized_output_dir}")
