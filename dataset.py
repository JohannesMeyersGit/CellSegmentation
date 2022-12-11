import os
import glob
from skimage import io
from torch.utils.data import Dataset
import numpy as np


class CellDataset(Dataset):

    def __init__(self,image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = glob.glob(image_dir+'\*.tif')
        self.masks = glob.glob(mask_dir+'\*.png')
        self.no_of_ims = len(self.images)
    
    def __getset__(self, idx):
        if idx < self.no_of_ims:
            image = np.array(io.imread(self.images[idx]))
            mask = np.array(io.imread(self.masks[idx]))
            # binarize mask 0 - background 1-foreground
            mask[mask==255.0] = 1.0        
        else:
            print('set idx out of bound!')
            image = -1
            mask = -1 
        
        return image, mask

