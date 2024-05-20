from diffusers import DDPMPipeline
import os

# initialize the model
diffusion_steps = 750
modelpath = "./DriveSceneGen/model_dxdy_agents_256_s80"
ddpm = DDPMPipeline.from_pretrained(modelpath,variant="fp16").to('cuda')

output_dir = "./data/generated_80m_5k/diffusion"
os.makedirs(output_dir, exist_ok=True)

for num in range(20):
    # generate dx dy
    polylines = ddpm(
        batch_size = 5,
        # generator=torch.manual_seed(1),
        num_inference_steps=diffusion_steps,
        #   output_type="pil",
        #   return_dict=False
        ).images

    for i, image in enumerate(polylines):
        # save image
        image.save(f"{output_dir}/loop_{num:03d}_batch_{i:03d}.png")