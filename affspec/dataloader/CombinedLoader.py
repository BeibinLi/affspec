
import torch
from torchvision import transforms

from .AffectNetLoader import AffectNetDataset
from .EmotioNetLoader import EmotioNetDataset

from .custom_transforms import FaceCrop

import pdb


#%%
train_transformer = transforms.Compose([
        FaceCrop( scale = 1.3 ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop( size = 224, scale = (0.9, 1.1) ),
        transforms.RandomAffine(5, shear = 20),
        transforms.ToTensor()
])
    
valid_transformer = transforms.Compose([
        FaceCrop( scale = 1.3 ),
        transforms.Resize( size = (224, 224) ),
        transforms.ToTensor()
])


#%%
def get_loader( transforms, affect_csv, emotio_csv, batch_size = 64, shuffle = True, workers = 4 ):    
    if emotio_csv is None:
        datasets = torch.utils.data.ConcatDataset( 
            [ AffectNetDataset( affect_csv, transform = transforms ) ]
        )    
    elif affect_csv is None:
        datasets = torch.utils.data.ConcatDataset( 
            [ EmotioNetDataset( emotio_csv, transform = transforms ) ]
        )
    else:
        datasets = torch.utils.data.ConcatDataset( 
            [ AffectNetDataset( affect_csv, transform = transforms ) ,
             EmotioNetDataset( emotio_csv, transform = transforms ) ]
        )    
        
    loader = torch.utils.data.DataLoader( datasets, batch_size = batch_size, shuffle = shuffle, num_workers = workers )

    return loader

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import tqdm
    count = 0 
    trainloader = get_loader( train_transformer, "../data_splits/affectnet_train.csv", "../../Emotionet/all_AU_coding_train_v2.csv"  )

    for inputs, labels, weights in tqdm.tqdm(trainloader):        
        #pdb.set_trace()
#        x = inputs.numpy()
#        x = x.reshape(3, 224, 224 )
#        x = np.swapaxes( x, 0, 2 )
#        x = np.swapaxes( x, 0, 1 )
#        print( labels )
#        plt.imshow( x )
#        plt.savefig( "rst_%d.jpg" % count )
        
        count += 1
#        if count > 10: break
    
    
    
