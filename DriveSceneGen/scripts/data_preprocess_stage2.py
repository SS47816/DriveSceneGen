import glob
import multiprocessing
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

from torchvision import transforms
from DriveSceneGen.utils.datasets.rasterization import rasterize_static_map

def multiprocessing_func(data_files, tensor_output_dir, n_proc, proc_id):
    # Settings
    map_range = 40.0  # total size should x2
    img_res = (512,512) # will be resized to 256x256
    resize = False
    scatter_size = 0.15
    with_vehicle_rectangle=True
    scatter_as_line=True
    
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
                                        with_vehicle_rectangle=with_vehicle_rectangle,
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
    n_proc = 8  # Numer of available processors
    
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--load_path',default="./data/processed", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="./data/processed/stage2/",type=str, help='path to save processed data')
    
    sys_args = parser.parse_args()

    sources_dir = sys_args.load_path
    raw_data_dir = os.path.join(sources_dir, "stage1")

    output_dir = sys_args.save_path
    tensor_output_dir= os.path.join(output_dir, "GT_diff_s80_70k_dxdy_agents_img")
    
    if not os.path.exists(tensor_output_dir):
        os.mkdir(tensor_output_dir)
    
    all_files = glob.glob(raw_data_dir + "/*")
    chunked_files = list(
        chunks(all_files, int(len(all_files) / n_proc) + 1)
    )  # Splits input among n_proc chunks

    processes = []  # Initialize the parallel processes list
    for i in np.arange(n_proc):
        """Execute the target function on the n_proc target processors using the splitted input"""
        p = multiprocessing.Process(
            target=multiprocessing_func,
            args=(chunked_files[i], tensor_output_dir, n_proc, i),
        )
        processes.append(p)
        p.start()
    for process in processes:
        process.join()

    print("Process finished!!!")

        
        
        
