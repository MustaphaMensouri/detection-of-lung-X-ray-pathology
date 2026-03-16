import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

class LungDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 32, 
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: list = [0.8, 0.1, 0.1],
        image_size: list = [224, 224]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        
        self.train_transform = transforms.Compose([
            transforms.Resize(tuple(image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder(root=self.data_dir)
        
        total_len = len(full_dataset)
        train_len = int(self.hparams.train_val_split[0] * total_len)
        val_len = int(self.hparams.train_val_split[1] * total_len)
        test_len = total_len - train_len - val_len
        
        self.train_data, self.val_data, self.test_data = random_split(
            full_dataset, [train_len, val_len, test_len]
        )
        
        self.train_data.dataset.transform = self.train_transform
        self.val_data.dataset.transform = self.val_transform
        self.test_data.dataset.transform = self.val_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )