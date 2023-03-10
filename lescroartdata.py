from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from glob import glob
import os
import numpy as np

class LescroartDataset(Dataset):
    def __init__(self, dataset_dir='../Lescroart.etal.2018/stimuli_trn_run0',
                       batch_size=1):
        
        #self.file_list = glob(os.path.join(dataset_dir, 'fr001[1-6]000.png'))
        file_list = [
            'fr0000692.png',
            'fr0001171.png',
            'fr0000840.png',
            'fr0000757.png',
            'fr0000637.png',
            'fr0001270.png',
            'fr0001383.png',
            'fr0001408.png',
            'fr0001737.png',
            'fr0002496.png'
        ]
        self.file_list = [os.path.join(dataset_dir, f) for f in file_list]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img1 = read_image(file_name)
        img1 = img1[0:3, :, :] # remove alpha channel
        
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