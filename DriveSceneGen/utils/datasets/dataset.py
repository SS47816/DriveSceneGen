# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
import os
import glob
import torch.utils.data as torch_data
from torchvision import transforms
from PIL import Image

class Image_Dataset(torch_data.Dataset):
    
    def __init__(self, config):
        self.data_list = glob.glob(config.dataset_name)
        self.config = config

        self.normalize = transforms.Compose([
            transforms.Resize((config.patterns_size_height, config.patterns_size_width),antialias=False),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self):
        return len(self.data_list)
    
    def remove_sample(self, index):
        del self.data_list[index]

    def __getitem__(self, index):
        file = self.data_list[index]
        file_extension = os.path.splitext(file)[1].lower()
        
        with open(file, 'rb') as f:
            if file_extension == '.pkl':
                data_dict = torch.load(f)
                if isinstance(data_dict, dict) == False:
                    return self.__getitem__(index + 1)
                sample = data_dict['fig_tensor'][:,:,:].permute(2,0,1)

            else:
                totensor = transforms.ToTensor()
                sample = totensor(Image.open(f))
            
            sample = self.normalize(sample)
        f.close()
    
        return sample
