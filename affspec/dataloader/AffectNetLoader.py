import os, random, time

import numpy as np
import pandas as pd

import torch
import torch.utils.data

from skimage import io

import pdb

import matplotlib.pyplot as plt



#%%
BASE_PATH = "/projects/grail/affectnet/Manually_Annotated_Images/"

#%%
class AffectNetDataset(torch.utils.data.Dataset):
    def __init__(self, csv_name, transform ):
        self.csv_name = csv_name
        df =  pd.read_csv( csv_name )
        df = df[ df.expression < 8 ]
        df = df[ df.valence >= -1.0 ]
        df = df[ df.arousal >= -1.0 ]
        df = df.reset_index()
        
        self.transform = transform
        self.df = df
        self.get_weights()
        
        print( "AffecNet DF size:", self.df.shape )

    def __len__(self):
        return self.df.shape[0]


    def get_weights(self):        
        counts = np.array( self.df.expression.value_counts( sort = False ).tolist() ) # sort = False is important
        weight = 1 / counts        
        self.weights = weight / np.sum( weight ) * len( counts )        
        print( "The weights for AffectNet is:" , self.weights )      
        
        return self.weights

    def __getitem__(self, idx):
        
        try:
            imgpath = self.df.subDirectory_filePath[idx]
            imgpath = os.path.join( BASE_PATH, imgpath )
            
            t, r, b, l = self.df.face_y[idx], self.df.face_x[idx] + self.df.face_width[idx], self.df.face_x[idx] + self.df.face_height[idx], self.df.face_x[idx]

            img = io.imread(os.path.join(imgpath))
            
            input_data = {"image_path":imgpath, 'image':img, "top":t, "right":r, "bottom":b, "left": l}

            if self.transform:
                data = self.transform( input_data )
            else:
                raise("unknown transform function!")


            labels = { "expression": self.df.expression[idx],
                      "valence": float( self.df.valence[idx] ), 
                      "arousal": float( self.df.arousal[idx] ), 
                      "action units": np.zeros(12).reshape(-1), 
                      }



            weights =  { 
                          "expression": self.weights[ labels["expression"] ],
                          "valence": float( self.weights[ labels["expression"] ] ),
                          "arousal": float( self.weights[ labels["expression"] ] ),
                          #"arousal": float( 1.0 ),
                          "action units": np.zeros(12).reshape(-1)
                      }

            return data, labels, weights
#            return data, labels, 0

        except Exception as e:
            print( e )

            #pdb.set_trace()
            # Return Another Random Choice            
            # raise("error")
            return self.__getitem__( np.random.randint(0, self.__len__()) )
                    
        # return img, valence_arousal, weight 




if __name__ == "__main__":
#    train_csv_name = "/projects/grail/affectnet/validation.csv"
    train_csv_name = "/projects/grail/affectnet/training.csv"

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
    
    train_dataset = AffectNetDataset( train_csv_name, transform = transformer )
    
    
    trainloader = torch.utils.data.DataLoader( train_dataset, batch_size = 1, shuffle = False )
    
    #%%
    import tqdm
    count = 0 
    for inputs, labels, weights in tqdm.tqdm(trainloader):        
        x = inputs.numpy()
        pdb.set_trace()
        x = x.reshape(3, 224, 224 )
        x = np.swapaxes( x, 0, 2 )
        x = np.swapaxes( x, 0, 1 )
        plt.imshow( x )
        plt.savefig( "rst_%d.jpg" % count )
        
        count += 1
        if count > 10: break

