import torch.utils.data as data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets
from PIL import Image
import os
import skimage.io as io
from PIL import Image


class PetCTDataset(Dataset):
    """My dataset for regression task."""

    def __init__(self, pet_csv, ct_csv, pet_root_dir, ct_root_dir, transform=None):
        self.transform = transform
        
        self.pet_csv_pth = pet_csv
        self.pet_csv_data = pd.read_csv(self.pet_csv_pth)
        
        self.ct_csv_pth = ct_csv
        self.ct_csv_data = pd.read_csv(self.ct_csv_pth)[['Patient Number', 'Age\n(years)']]
        
        self.pet_imgs_pth = pet_root_dir
        self.pet_imgs = [os.path.join(self.pet_imgs_pth, img) for img in os.listdir(self.pet_imgs_pth)]
        self.pet_number = len(self.pet_imgs)
        
        self.ct_imgs_pth = ct_root_dir
        self.ct_imgs = [os.path.join(self.ct_imgs_pth, img) for img in os.listdir(self.ct_imgs_pth)]
        self.ct_number = len(self.ct_imgs)

    def create_pure(self, idx):
        filename = self.pure_file_names[idx]
        path = self.root_dir + filename
        img = self.loader(path)
        patient_id = int(filename.split('-')[0])
        age = int((self.ct_csv_data).loc[self.ct_csv_data['Patient Number'] == patient_id, 'Age\n(years)'])
        if self.transform is not None:
            img = self.transform(img)
        return np.array(img), age
    


    def __len__(self):
        return self.pet_number + self.ct_number
    
    def __getitem__(self, idx):
        if idx < self.pet_number:
            img_pth = self.pet_imgs[idx]
            img_name = img_pth.split('/')[-1]
            # Get image
            # image = Image.fromarray(io.imread(img_pth), mode='RGB')
            image = Image.open(img_pth).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Get label
            patient_id = '-'.join(img_name.split('-')[:3])
            label = int((self.pet_csv_data).loc[self.pet_csv_data['Patient #'] == patient_id, 'Age'])
            # Return
            return image, label
        else:
            idx -= self.pet_number
            img_pth = self.ct_imgs[idx]
            img_name = img_pth.split('/')[-1]
            # Get image
            # image = Image.fromarray(io.imread(img_pth), mode='RGB')
            image = Image.open(img_pth).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Get label
            patient_id = int(img_name.split('-')[0])
            label = int((self.ct_csv_data).loc[self.ct_csv_data['Patient Number'] == patient_id, 'Age\n(years)'])
            # Return
            return image, label
            
        

        
        
        
        
        
        
