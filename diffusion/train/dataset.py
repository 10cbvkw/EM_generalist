import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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
        image = image / 255.0  
        image = (image * 2) - 1

        image = torch.from_numpy(image).unsqueeze(0) 

        if self.transform:
            image = self.transform(image)

        return image
