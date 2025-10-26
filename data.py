import pytorch_lightning as pl
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms

class MedMNISTDataLoader(pl.LightningDataModule):
    def __init__(self, data_flag: str = 'bloodmnist', data_dir: str = "./data", batch_size: int = 128, img_size: int = 32):
        super().__init__()
        self.data_flag = data_flag
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        info = INFO[self.data_flag]
        self.DataClass = getattr(medmnist, info['python_class'])

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def prepare_data(self):
        self.DataClass(split='train', root=self.data_dir, download=True)
        self.DataClass(split='val', root=self.data_dir, download=True)

    def setup(self, stage:str = None):
        if stage == "fit" or stage == None:
            self.train_dataset  = self.DataClass(
                split = 'train',
                root=self.data_dir,
                transform=self.transform,
                download=True
            )
            self.val_dataset = self.DataClass(
                    split = 'val',
                    root=self.data_dir,
                    transform=self.transform,
                    download=True
                )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=4
        )
