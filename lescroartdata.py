import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize
from glob import glob
import os
import numpy as np

class LescroartDataset(Dataset):
    def __init__(self, dataset_dir='../Lescroart.etal.2018/'):
        
        dirlist = glob(os.path.join(dataset_dir, 'stimuli_trn_*'))
        
        file_list = []
        for dir in dirlist:
            file_list.extend(glob(os.path.join(dir, 'fr*.png')))
        
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img1 = read_image(file_name)
        img1 = img1[0:3, :, :] # remove alpha channel
        
        ret = {'img1': img1, 'gt_segment': torch.randint(4, (1, 512, 512), dtype=torch.int64)}
        
        return ret
    
class BonnerDataset(Dataset):
    def __init__(self, dataset_dir='../stimuli/'):
        
        self.file_list = glob(os.path.join(dataset_dir, 'pathways*.jpg'))
        self.resize_op = Resize(size=(512, 512))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img1 = self.resize_op(read_image(file_name))
        
        ret = {'img1': img1, 
               'gt_segment': torch.randint(4, (1, 512, 512), dtype=torch.int64)}
        
        return ret
        
def fetch_dataloader_lesc(args):
    
    dataset = LescroartDataset()
    
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=False)     
    return dataloader

if __name__ == "__main__":
    pass