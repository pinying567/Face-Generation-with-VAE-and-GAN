import torch.utils.data as data
from PIL import Image
import os
import torch
import random

class faceDataset(data.Dataset):
    
    def __init__(self, img_dir, transform):

        self.img_dir = img_dir
        self.transform = transform
        self.data = []    
        self.fname = []
        
        for x in os.listdir(img_dir):
            img_path = os.path.join(img_dir, x)
            img = Image.open(img_path).convert('RGB')
            self.fname.append(x)
            self.data.append(img)
        
    def __getitem__(self, index):
        img = self.transform(self.data[index])
        fname = self.fname[index]
        return img, fname

    def __len__(self):
        return len(self.data)
