import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

class CTDataset(Dataset):
    def __init__(self, image_dir):
        self.hdf5_file_path = image_dir
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        self.image = self.hdf5_file['images'].astype(np.float32) # images or patches

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        image = torch.from_numpy(image).unsqueeze(0)
        image = (image / 255) * 2 - 1
        return image

class TifImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.tiff') or fname.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  

        image = np.array(image).astype(np.float32)
        image = image / 255.0  * 2 - 1

        image = torch.from_numpy(image).unsqueeze(0) 

        if self.transform:
            image = self.transform(image)

        return image
