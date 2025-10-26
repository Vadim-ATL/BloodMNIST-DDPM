import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid
from pathlib import Path

from models.unet import UNet
from models.ddpm import DDPM


class LitDDPM(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_dim=256,
        timesteps=1000,
        lr=0.001,
        img_size=32,
        samples_dir='samples'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.timesteps = timesteps
        self.lr = lr
        self.img_size = img_size
        self.samples_dir = Path(samples_dir)

        self.model = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            time_dim=self.time_dim
        )

        self.ddpm = DDPM(T=self.timesteps, device=self.device)

    
    def training_step(self, batch, batch_idx):

        images, _ = batch
        batch_size = images.shape[0]

        t = torch.randint(
            low=0, 
            high=self.timesteps, 
            size=(batch_size,), 
            device=self.device
        )
        self.ddpm.to(self.device)
        noise = self.ddpm.sample_noise(images)
        noisy = self.ddpm.forward_diffusion(x0=images,t=t,epsilon=noise)

        pred_noise = self.model(noisy,t)

        loss = F.mse_loss(pred_noise, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
  
        images, _ = batch
        batch_size = images.shape[0]
        t = torch.randint(
            low=0, 
            high=self.timesteps, 
            size=(batch_size,), 
            device=self.device
        )
        self.ddpm.to(self.device)
        noise = self.ddpm.sample_noise(images)
        noisy = self.ddpm.forward_diffusion(x0=images, t=t, epsilon=noise)

        pred_noise = self.model(noisy, t)

        loss = F.mse_loss(pred_noise, noise)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.lr
        )
        return optimizer

    def on_validation_epoch_end(self):
        print(f"\nGenerating samples for epoch {self.current_epoch}...")
        n_samples = 16
        
        noise_shape = (
            n_samples,
            self.in_channels,
            self.img_size,
            self.img_size
        )
        self.ddpm.to(self.device)
        

        generated_images = self.ddpm.sample(self.model, noise_shape)        
        generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)

        filename = f"epoch_{self.current_epoch}_GENERATED_samples.png"
        save_path = self.samples_dir / filename

        save_image(
            generated_images, 
            save_path,
            nrow=4
        )
        
        if self.logger:
            grid = make_grid(generated_images, nrow=4)
            self.logger.experiment.add_image(
                "generated_samples", 
                grid, 
                self.current_epoch
            )


