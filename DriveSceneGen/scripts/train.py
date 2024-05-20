import torch
from diffusers import UNet2DModel,DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import notebook_launcher
import matplotlib.pyplot as plt
from DriveSceneGen.utils.datasets.dataset import Image_Dataset
from DriveSceneGen.pipeline.training_pipeline import TrainingPipeline
from dataclasses import dataclass
from torchvision import transforms

####---initial config---####
@dataclass
class TrainingConfig:
    patterns_size_height = 256 # max 400 remember change at dataset
    patterns_size_width = 256
    train_batch_size = 14
    eval_batch_size = 1
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1 # save model epoch
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = './DriveSceneGen/model_dxdy_agents_256_s80'  # the generated model name
    dataset_name = "./data/rasterized/GT_70k_s80_dxdy_agents_img/*"
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 14555

config = TrainingConfig()

####---load dataset---####

dataset = Image_Dataset(config)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

##---load model---####

model = UNet2DModel(
    sample_size=(config.patterns_size_height,config.patterns_size_width),  # the target pattern resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 256, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",  
        "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "UpBlock2D",  # a ResNet upsampling block with spatial self-attention  
        "UpBlock2D",
        "UpBlock2D"  
    ),
)

# model = UNet2DModel.from_pretrained(config.output_dir,subfolder="unet")
print("model parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

####---load optimizer, scheduler and pipeline---####

pipeline = TrainingPipeline(config)
noise_scheduler = DDPMScheduler()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def add_noise_verification(data,index,noisy=True,intensities=100):
    
    inverse_normalize = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    ])
    
    toimage = inverse_normalize(data[index])
     
    if noisy == True:
        
        #add noise to original pattern        
        noise = torch.randn(toimage.shape)

        #diffusion timesteps, determine how much noise to add
        timesteps = torch.LongTensor([intensities])
        
        #add noise to original pattern
        noisy_pattern = noise_scheduler.add_noise(toimage, noise, timesteps)
        
        #tensor->numpy
        numpy_noisy_pattern = noisy_pattern.numpy()
    
        plt.imshow(numpy_noisy_pattern[:,:,:])
    else:
        
        # additional_channel=torch.ones(toimage.shape[0],toimage.shape[1],1)*0.5
        # toimage = torch.cat((toimage,additional_channel),dim=2)
        # plt.imshow(toimage[:,:,2].numpy(),cmap='gray')
        
        plt.imshow(toimage[:,:,:])
    
    plt.show()

    

####---start training---####

if __name__ == "__main__":
         
    print("dataset:",len(dataset))
    
    ## show examples, with forward diffusion
    # for example_index in range(0,len(dataset)):
    #     print(dataset[example_index].shape)
    #     add_noise_verification(dataset, example_index, noisy=False, intensities=100)
    
    ## train
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(pipeline.train_loop, args, num_processes=1)

