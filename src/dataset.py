import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import datasets



class CusDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, download=False, if_save=False):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.download = download
        self.if_save = if_save
        self.image_files = os.listdir(self.root_dir)
    

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.if_save:
            null_label = self.image_files[idx]
        else:
            null_label = 0
                
        return image, null_label
    


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, download=False, if_save=False):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.download = download
        self.if_save = if_save
        self.image_files = os.listdir(self.root_dir)


    def __len__(self):
         return len(self.image_files)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.if_save:
            null_label = self.image_files[idx]
        else:
            null_label = 0
            
        return image, null_label

