import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data import MedMNISTDataLoader
from omegaconf import OmegaConf
from pathlib import Path

from lit_ddpm import LitDDPM

def main():

    # 1. Config loading
    config = OmegaConf.load("config.yaml")

    # 2. Defining the outputs paths of the results
    output_dir = Path(config.logging.output_dir)

    logs_dir = output_dir / "logs"
    weights_dir = output_dir / "weights"
    samples_dir = output_dir / "samples"

    # 3. Creating dataset with dataloaders

    dm = MedMNISTDataLoader(
        data_flag=config.data.data_flag,
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        img_size=config.data.img_size
    )

    model_config_dict = OmegaConf.to_container(config.model, resolve=True)
    
    # 4. creating model with UNet + Scheduler enhanced with Pytorch lightning 
    model = LitDDPM(
        **model_config_dict,
        lr=config.training.lr,
        img_size=config.data.img_size,
        samples_dir=samples_dir
    )

    # 5. Trivia: Logging and checkpoint callback

    logger = TensorBoardLogger(
        logs_dir,
        name=config.logging.experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_dir,
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        max_epochs=config.training.epochs,
        log_every_n_steps=config.training.log_every_n_steps,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    print("Starting training")
    trainer.fit(model, datamodule=dm)

    print("Training finished. Saving the model")
    
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    print("Model saved.")

if __name__ == "__main__":
    main()