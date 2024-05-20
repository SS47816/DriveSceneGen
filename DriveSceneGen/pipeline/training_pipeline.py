import torch
from torchvision import transforms
from diffusers import DDPMPipeline
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import numpy as np


class TrainingPipeline():
    def __init__(self,config):
        self.config = config
        
    ####----evaluate---####
    def evaluate(self,config, epoch, pipeline):
        
        self.inverse_normalize = transforms.Compose([
            # transforms.Lambda(lambda t: (t + 1) / 2),
            # transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        polylines_patterns = pipeline(
            num_inference_steps=750,
            batch_size = config.eval_batch_size, 
            generator=torch.manual_seed(config.seed),
            output_type="np.array",
            return_dict=False
        )

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        file_count = len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
        polylines_patterns = torch.tensor(polylines_patterns)
        polylines_patterns = polylines_patterns[0, 0, :, :, :]

        # save image
        toimage = self.inverse_normalize(polylines_patterns)
        toimage.save(f"{test_dir}/"+f"{file_count:03d}"+".png")
        
    ####---define train loop---####
    def train_loop(self,config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps, 
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs")
        )
        if accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")
        
        # Prepare everything
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        
        global_step = 0
        
        # show the progress bar only on the main process
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                # Sample noise to add to the patterns(batch)
                noise = torch.randn(batch.shape).to(batch.device)
                bs = batch.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=batch.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_pattern = noise_scheduler.add_noise(batch, noise, timesteps).to(torch.float)
                
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_pattern, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    self.evaluate(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline.save_pretrained(config.output_dir) 
