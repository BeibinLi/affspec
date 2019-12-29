import numpy as np
import random
from PIL import Image

import skimage

#%%
class FaceCrop(object):
    """
    Crop the face region from an image

    Args:
        scale (float): The scale is numerical ratio for cropping the face.
            e.g.
            If it is greater than 1, then the cropped face will be bigger than the detected face (crop more in backgrounds)
            If it is smaller than 1, then the cropped face will be smaller than the detected face (hair, chin, etc are cropped away)

    Returns:
        image (Image.array): the PIL image.        
    """
        
    def __init__(self, scale = 1.3):
        self.scale = scale
        
    def __call__(self, sample):
        img = sample['image'] 
        
        top, right, bottom, left = sample["top"], sample["right"], sample["bottom"], sample["left"]
        
        if top is None or right is None or bottom is None or left is None or np.isnan(top) or np.isnan(right) or np.isnan(bottom) or np.isnan(left) or top + left + right + bottom == 0:
            # cannot detect the face, use the whole image
            top = 0
            left = 0
            right = img.shape[1] - 1
            bottom = img.shape[0] - 1
        else:                
            width =  right - left
            height =  bottom - top
            top -= height * (self.scale - 1)
            bottom += height * (self.scale - 1)
            left -= width * (self.scale - 1)
            right += width * (self.scale - 1)
            
            top = max(0, top)
            bottom = min( img.shape[0]-1, bottom)
            left = max(0, left)
            right = min( img.shape[1]-1, right)

#            top, left, right, bottom = int(top), int(left), int(right), int(bottom)
            
#        print( img.shape, top, left, right, bottom )

        try:
            img = skimage.color.gray2rgb( img ) # cast gray image to rgb
            img = img[ int(top):int(bottom), int(left):int(right) ]
        except Exception as e:
            print(e)
            print( "Error Occured in %s" % sample["image_path"] )
            print( img.shape, top, left, right, bottom )
        
#        print( img.shape )
        return Image.fromarray( img )
    
    
    
#%%
if __name__ == "__main__":
    
    from skimage import io, transform
    import matplotlib.pyplot as plt
    from torchvision import transforms, datasets

    img = io.imread("test.jpg")
    x = {"image": img,
         "top":144,
         "bottom":644,
         "left":38,
         "right":538 }
    
    transformer = transforms.Compose([
            FaceCrop( ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop( size = 224, scale = (0.9, 1.1) ),
            transforms.RandomAffine(5, shear = 20),
#            transforms.ToTensor()
    ])
    
    #%%
    y = transformer( x )
    
#    plt.imshow(x)
    plt.imshow(y)