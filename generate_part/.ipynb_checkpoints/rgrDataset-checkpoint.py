import torch.utils.data as data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets
from PIL import Image
import os
import os.path

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

    
class RgrDataset(Dataset):
    """My dataset for regression task."""

    def __init__(self, csv_file, root_dir, mode, transform=None, root_dir_results=None, loader=default_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            root_dir: datasets/pure/train/
            root_dir_results: datasets/together/train/mr2ct-pretrained/test_latest/images/
            img contains TCGA_HT_A61B should be overlooked(cauz. Nan in csv)
            
            ---
            
            
            pure_mri: /yy-volume/gan_example/datasets/mr_ct/trainA
            pure_mri_val: /yy-volume/gan_example/datasets/mr_ct/testA
            fake_ct: /yy-volume/train_rgr/datasets/together/train/mr2ct-pretrained/test_latest/images
            fake_ct_val: /yy-volume/train_rgr/datasets/pure_mri/val/mr2ct_pretrained/test_latest/images
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.root_dir_results = root_dir_results
        self.mode = mode
        self.loader = loader
        self.transform = transform
        
        self.pure_file_names = [file_name for file_name in  os.listdir(self.root_dir) if len(file_name) > 6]
        if root_dir_results is not None:
            self.toge_file_names = [file_name for file_name in os.listdir(self.root_dir_results) if 'TCGA_HT_A61B' not in file_name \
                                    and 'fake' in file_name \
                                    and len(file_name) > 12]
        self.ct_csv_data = pd.read_csv('ct.csv')[['Patient Number', 'Age\n(years)']]
        self.mri_csv_data = pd.read_csv('mri.csv')[['Patient', 'age_at_initial_pathologic']]
        
        self.expe_pure_mr = [file_name for file_name in os.listdir(self.root_dir) if 'TCGA_HT_A61B' not in file_name \
                                    and len(file_name) > 12]
        self.expe_fake_ct_dir =  [file_name for file_name in os.listdir(self.root_dir) if 'TCGA_HT_A61B' not in file_name \
                                    and 'fake' in file_name \
                                    and len(file_name) > 12]
#         self.pure_mri = '/yy-volume/gan_example/datasets/mr_ct/trainA/'
#         self.pure_mri_val = '/yy-volume/gan_example/datasets/mr_ct/testA/'
#         self.fake_ct = '/yy-volume/train_rgr/datasets/together/train/mr2ct-pretrained/test_latest/images/'
#         self.fake_ct_val = '/yy-volume/train_rgr/datasets/pure_mri/val/mr2ct_pretrained/test_latest/images/'
        
        
    def create_pure(self, idx):
        filename = self.pure_file_names[idx]
        path = self.root_dir + filename
        img = self.loader(path)
        patient_id = int(filename.split('-')[0])
        age = int((self.ct_csv_data).loc[self.ct_csv_data['Patient Number'] == patient_id, 'Age\n(years)'])
        if self.transform is not None:
            img = self.transform(img)
        return np.array(img), age

    def create_results(self, idx):
        idx -= len(self.pure_file_names)
        filename = self.toge_file_names[idx]
        path = self.root_dir_results + filename
        img = self.loader(path)
        patient_id = filename[:12]
        age = int((self.mri_csv_data).loc[self.mri_csv_data['Patient'] == patient_id, 'age_at_initial_pathologic'])
        if self.transform is not None:
            img = self.transform(img)
        return np.array(img), age
    
#     Experiment
    
    def expe(self, idx):
        filename = None
#         print('>> LEN:', len(self.expe_fake_ct_dir))
        if 'trans' in self.root_dir:
            filename = self.expe_fake_ct_dir[idx]
        else:
            filename = self.expe_pure_mr[idx]
        
        path = self.root_dir + filename
        img = self.loader(path)
        patient_id = filename[:12]
        age = int((self.mri_csv_data).loc[self.mri_csv_data['Patient'] == patient_id, 'age_at_initial_pathologic'])
        if self.transform is not None:
            img = self.transform(img)
        return np.array(img), age
    
#     def create_fake_ct(self, idx):
#         filename = os.listdir(self.fake_ct)[idx]
#         path = self.root_dir + filename
#         img = self.loader(path)
#         patient_id = filename[:12]
#         age = int((self.mri_csv_data).loc[self.mri_csv_data['Patient'] == patient_id, 'age_at_initial_pathologic'])
#         if self.transform is not None:
#             img = self.transform(img)
#         return np.array(img), age
    
#     def create_pure_mri_val(self, idx):
#         filename = os.listdir(self.pure_mri_val)[idx]
#         path = self.pure_mri + filename
#         img = self.loader(path)
#         patient_id = filename[:12]
#         age = int((self.mri_csv_data).loc[self.mri_csv_data['Patient'] == patient_id, 'age_at_initial_pathologic'])
#         if self.transform is not None:
#             img = self.transform(img)
#         return np.array(img), age
    
#     def create_pure_mri_val(self, idx):
#         filename = os.listdir(self.fake_ct_val)[idx]
#         path = self.pure_mri + filename
#         img = self.loader(path)
#         patient_id = filename[:12]
#         age = int((self.mri_csv_data).loc[self.mri_csv_data['Patient'] == patient_id, 'age_at_initial_pathologic'])
#         if self.transform is not None:
#             img = self.transform(img)
#         return np.array(img), age
    
#     Bult-in
    
    def __len__(self):
        if self.mode == 'pure':
            return len(self.pure_file_names)
        elif self.mode == 'toge':
            return len(self.toge_file_names) + len(self.pure_file_names)
        elif self.mode == 'expe':
            if 'trans' in self.root_dir:
                return len(self.expe_fake_ct_dir)
            else:
                return len(self.expe_pure_mr)
        else:
            print('===================!')
            print('WRONG MODE!')
            print('===================!')

    def __getitem__(self, idx):
#         print('Current idx:', idx)
        if self.mode == 'pure':
            return self.create_pure(idx)
        elif self.mode == 'toge':
            if idx < len(self.pure_file_names):
                return self.create_pure(idx)
            else:
                return self.create_results(idx)
        elif self.mode == 'expe':
            return self.expe(idx)
        else:
            print('===================')
            print('WRONG MODE!')
            print('===================')
            
        

        
        
        
        
        
        
