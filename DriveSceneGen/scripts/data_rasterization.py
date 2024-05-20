import glob
import multiprocessing
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import yaml

from torchvision import transforms
from DriveSceneGen.utils.datasets.rasterization import rasterize_static_map

def multiprocessing_func(data_files, cfg, tensor_output_dir, n_proc, proc_id):
    # Settings
    cfg = cfg['rasterization']
    map_range = cfg['map_range']/2  # total size should x2
    img_res = (cfg['img_res'], cfg['img_res']) # will be resized to 256x256
    resize = cfg['resize']
    scatter_size = cfg['scatter_size']
    with_agent = cfg['with_agent']
    scatter_as_line = cfg['scatter_as_line']
    
    for i, file in enumerate(tqdm(data_files)):
        with open(file, 'rb') as f:
            scenario_info = pickle.load(f)
            if not isinstance(scenario_info, dict):
                print(file)
                print(type(scenario_info))
                continue    

        # fig_tensor= [256,256,3]
        fig_tensor=rasterize_static_map(scenario_info, 
                                        img_res=img_res, 
                                        map_range=map_range, 
                                        with_agent=with_agent,
                                        scatter_as_line=scatter_as_line,
                                        resize=resize,
                                        scatter_size=scatter_size)
        if fig_tensor is None:
            print("in this region, no figure to display") 
            continue

        to_img = transforms.ToPILImage()
        fig_tensor = fig_tensor.permute(2,0,1)
        fig_img = to_img(fig_tensor)
        fig_img.save(f"{tensor_output_dir}/"+f"{proc_id}"+"_"+ f"{i}.png")

        f.close()
        
def chunks(input, n):
    """Yields successive n-sized chunks of input"""
    for i in range(0, len(input), n):
        yield input[i : i + n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing 2')
    parser.add_argument('--load_path',default="./data/preprocessed", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="./data/rasterized/",type=str, help='path to save processed data')
    parser.add_argument('--cfg_file', default="./DriveSceneGen/config/data_rasterization.yaml",type=str, help='path to cfg file')
    
    sys_args = parser.parse_args()
    with open(sys_args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
    n_proc = cfg['n_proccess']
    map_range = cfg['rasterization']['map_range']
    raw_data_dir = sys_args.load_path
    output_dir = sys_args.save_path
    tensor_output_dir= os.path.join(output_dir, f"GT_70k_s{map_range}_dxdy_agents_img")
    
    if not os.path.exists(tensor_output_dir):
        os.makedirs(tensor_output_dir, exist_ok=True)
    
    all_files = glob.glob(raw_data_dir + "/*")
    chunked_files = list(
        chunks(all_files, int(len(all_files) / n_proc) + 1)
    )  # Splits input among n_proc chunks

    processes = []  # Initialize the parallel processes list
    for i in np.arange(n_proc):
        """Execute the target function on the n_proc target processors using the splitted input"""
        p = multiprocessing.Process(
            target=multiprocessing_func,
            args=(chunked_files[i], cfg, tensor_output_dir, n_proc, i),
        )
        processes.append(p)
        p.start()
    for process in processes:
        process.join()

    print("Process finished!!!")

        
        
        
