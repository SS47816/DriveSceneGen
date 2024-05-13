from diffusers import DDPMPipeline
import os

# initialize the model
diffusion_steps = 750
modelpath = "traffic_diffuser_model_dxdy_agents_256_s80"
ddpm = DDPMPipeline.from_pretrained(modelpath,variant="fp16").to('cuda')

test_dir = os.path.join(modelpath, "results")
os.makedirs(test_dir, exist_ok=True)

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
      image.save(f"{test_dir}/"+f"loop_{num:03d}"+f"_batch_{i:03d}.png")