import os, random, time

import numpy as np
import pandas as pd

import torch
import torch.utils.data

from skimage import io

import pdb

import matplotlib.pyplot as plt


#%%    
class EmotioNetDataset(torch.utils.data.Dataset):
    def __init__(self, csv_name, transform = None, base_path = "../../Emotionet/" ):
        self.csv_name = csv_name
        df =  pd.read_csv( csv_name )
        df = df.reset_index()
        self.df = df
        
        self.transform = transform
        self.base_path = base_path

        # self.labels = []
        self.AUs = [ "1", "2", "4", "5", "6", "9", "12", "17", "20", "25", "26", "43" ] # the actual AUs
        self.labels = [ str( int(_) - 1 ) for _ in self.AUs  ] # we need minus one because our index starts at zero.

        self.get_weight_for_each_sample_with_all_aus()
        
        
        print( "EmotioNet DF size:", self.df.shape )
        

    def get_weight_for_each_sample_with_all_aus( self ):        
        
        self.label_activate_ratio = []
        for lab in self.labels:
            self.label_activate_ratio.append( np.sum( self.df[lab] ) / self.df.shape[0] )
            
            
        weights = np.ones( self.__len__() )
        for i in range(12):
            ratio = self.label_activate_ratio[ i ]
            column_name = self.labels[ i ]
            arr = self.df[ column_name ].apply( lambda _ : 1 - ratio if _ else  ratio )
        
            arr = np.array( arr.tolist() )
        
            weights = weights * arr
        
        
        self.weights =  np.sqrt( np.log(weights) - np.min( np.log(weights) ) + 1 )

#        pdb.set_trace()
        return weights.tolist()
    
    
    def __len__(self):  
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        try:
            imgpath = self.df.imgname[idx]
#            imgpath = imgpath.replace("/homes/gws/beibin/Emotionet/",  self.base_path )

            
            t, r, b, l = self.df.top[idx], self.df.right[idx], self.df.bottom[idx], self.df.left[idx]

            img = io.imread(os.path.join(imgpath))
            
            input_data = {"image_path":imgpath, 'image':img, "top":t, "right":r, "bottom":b, "left": l}

            if self.transform:
                data = self.transform( input_data )
            else:
                raise("unknown transform function!")

            label = self.df.loc[ idx, self.labels ].tolist() # possibility of expression
            label = np.array( label )
            if np.sum( label == 999 ) > 0: print( "ERROR: there should not be 999!" )
            
            label = np.array( label > 0 ).astype( np.float )
            
            weights = []
            for i in range(len(label)):
                w = 1 / self.label_activate_ratio[i] if label[i] else 1 / (1 - self.label_activate_ratio[i])
                weights.append( w )
            weights = np.array( weights ).astype( np.float )                

            label = {  "expression": -100,
                     "action units": label.reshape(-1), 
                     "valence": -100.0,
                     "arousal": -100.0,
                     } # wrap in a dictionary

            weights = { "action units": weights.reshape(-1), 
                     "expression": 0.0,
                     "valence": 0.0,
                     "arousal": 0.0,
                     } # wrap in a dictionary
            
            
#            print( "EmotioNet weights", weights )
            return data, label, weights
#            return data, label, 0

        except Exception as e:
            print( e )
            print( t, r, b, l, imgpath ) 
            # print( idx )
            # pdb.set_trace()
            # raise( 'dam' )
            # Return Another Random Choice            
            return self.__getitem__( np.random.randint(0, self.__len__()) )
                    


if __name__ == "__main__":    
    fname =  "../../Emotionet/all_AU_coding_test.csv" 
    
    from torchvision import transforms, datasets
        
    import matplotlib.pyplot as plt
    
    from .custom_transforms import FaceCrop

    transformer = transforms.Compose([
            FaceCrop( scale = 1.3 ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop( size = 224, scale = (0.9, 1.1) ),
            transforms.RandomAffine(5, shear = 20),
            transforms.ToTensor()
    ])
    
    train_dataset = EmotioNetDataset( fname, transform = transformer )

#
    trainloader = torch.utils.data.DataLoader( train_dataset, batch_size = 1, shuffle = True  )
    
    #%%
    df = train_dataset.df
    for label in train_dataset.labels:
        r = df[ label ].sum() / df.shape[0]
        print( label, r )

    #%%
    import tqdm
    for inputs, labels, weights in tqdm.tqdm(trainloader):
#        print( inputs, labels )
#        continue
#        pdb.set_trace()
        
#        x = inputs.numpy()
#        x = x.reshape(3, 224, 224 )
#        x = np.swapaxes( x, 0, 2 )
#        x = np.swapaxes( x, 0, 1 )
#        plt.imshow( x )
#        break
        pass
