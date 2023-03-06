from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from glob import glob
import os
import numpy as np

class LescroartDataset(Dataset):
    def __init__(self, dataset_dir='../Lescroart.etal.2018/stimuli_trn_run0'):
        
        self.file_list = glob(os.path.join(dataset_dir, 'fr000000[0-8].png'))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img1 = read_image(file_name)
        img1 = img1[0:3, :, :]
        
        ret = {'img1': img1}
        
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