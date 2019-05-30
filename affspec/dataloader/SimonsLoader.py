import os, random, time

import numpy as np
import pandas as pd

import torch
import torch.utils.data

from skimage import io

import pdb

from torchvision import transforms

from .custom_transforms import FaceCrop
from .CombinedLoader import valid_transformer

#%%

valid_transformer = transforms.Compose([
        FaceCrop( scale = 1.3 ),
        transforms.Resize( size = [224, 224] ),
        transforms.ToTensor()
])




#%%    
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_name, transform = None, base_path = "" ):
        self.csv_name = csv_name
        df =  pd.read_csv( csv_name )
        df = df.reset_index()
        self.df = df
        
        if transform is None:
            transform = valid_transformer 
        
        self.transform = transform 
        
        self.base_path = base_path
    
    
    def __len__(self):  
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        try:
            imgpath = self.df.img_path[idx]
            
            if len(self.base_path) >= 0:
                imgpath = os.path.join(self.base_path, imgpath)

            t, r, b, l = self.df.top[idx], self.df.right[idx], self.df.bottom[idx], self.df.left[idx]


            img = io.imread(imgpath)
            
            input_data = {"image_path":imgpath, 'image':img, "top":t, "right":r, "bottom":b, "left": l}

            if self.transform:
                data = self.transform( input_data )
            else:
                raise("unknown transform function!")
            
            return imgpath, data

        except Exception as e:
            print( e )
#            pdb.set_trace()
            print( t, r, b, l, imgpath ) 
            return imgpath, torch.Tensor( np.zeros([3,224,224]) )
                    

#%%
