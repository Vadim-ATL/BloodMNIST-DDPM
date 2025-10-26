import torch
import torch.nn as nn
from torchvision.utils import save_image

from models.unet import UNet
from models.ddpm import DDPM
from lit_ddpm import LitDDPM

from omegaconf import OmegaConf
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sample_images():
    print("Loading model and scheduler...")
    
    config = OmegaConf.load("config.yaml")

    img_size = config.data.img_size
    timesteps = config.model.timesteps
    in_channels = config.model.in_channels
    
    checkpoint_path = config.inference.checkpoint_path
    sample_batch_size = config.inference.sample_batch_size
    save_name = config.inference.save_name
    
    print(f"Loading Lightning checkpoint from: {checkpoint_path}")
    model_config_dict = OmegaConf.to_container(config.model, resolve=True)
    
    lit_model = LitDDPM.load_from_checkpoint(checkpoint_path, map_location=DEVICE, **model_config_dict)
    lit_model = lit_model.to(DEVICE)
    lit_model.eval()

    model = lit_model.model

    print("Loading scheduler...")
    ddpm = DDPM(T=timesteps, device=DEVICE)

    print(f"Generating {sample_batch_size} images...")
    with torch.no_grad():
        shape = (sample_batch_size, in_channels, img_size, img_size)
        generated_images = ddpm.sample(unet=model, shape=shape)

    generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)

    save_dir = Path(config.inference.output)
    save_path = save_dir / save_name
    
    save_dir.mkdir(parents=True, exist_ok=True)

    save_image(
        generated_images, 
        save_path, 
        nrow=8 
    )

    print(f"Sampling complete. Saved to '{save_path}'")

if __name__ == "__main__":
    sample_images()