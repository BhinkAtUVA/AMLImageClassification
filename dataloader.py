import pandas as pd
import torch as tc
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import os

# idx, image, label, path
ItemType = tuple[int, tc.Tensor, int, str]
    
class BirdsDataset(Dataset):
    def __init__(self, csv_path: Path|str, images_root: Path, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.image_paths = self.df["image_path"].values
            self.labels = self.df["label"].values - 1
        else:
            self.image_paths = self.df["image_path"].values
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel = self.image_paths[idx]
        filename = os.path.basename(rel)
        img_path = self.images_root / filename

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.is_train:
            label = int(self.labels[idx])
            return img, label
        else:
            return img, int(self.ids[idx])