import copy, time, pdb, random, os, glob

import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models import *

import config

import matplotlib
#matplotlib.use( "Agg" ) # plot in the backend without interaction
import matplotlib.pyplot as plt
import face_recognition as fr
import cv2


#%%
img_size = [ 224, 224, 3 ]
OUT_DIR = "image/"

#%%

def detect_face( img ):
    """
    img is get from cv2.imread()

    Args:
        img (numpy.array): the input image

    Returns:
        t, r, b, l (int): top, right, bottom, left
        
    """    
    all_locations = fr.face_locations( img , model = 'hog' ) # get locations of all faces

    max_area = 0
    face_location = None

    for x in all_locations: # x is the location for one face
        face_size = ( x[2] - x[0] ) * ( x[3] - x[1] )

        if abs( face_size ) > max_area:
            face_location = copy.copy( x )

    # pdb.set_trace()
    t, r, b, l = face_location
    

    return t, r, b, l




from dataloader.custom_transforms import FaceCrop
from torchvision import transforms

transformer = transforms.Compose([FaceCrop(), transforms.Resize(size=[224, 224]), transforms.ToTensor()])

def img2torch( img ):
    """
    Detect and crop the face, and then convert the largest face to PyTorch tensor

    Args:
        img (numpy.array):  the input image as the numpy array with [h, w, 3] size.

    Returns:
        img (torch.Tensor): the input image as PyTorch object with [3, 224, 224] size.
        
    """    
    try:
        top, right, bottom, left = detect_face( img )
        
        img_dict = {"image": img, "top":top, "bottom":bottom, "left":left, "right":right}
        # pdb.set_trace()
        tensor = transformer(img_dict)
        tensor = tensor.unsqueeze(0) # make 3-dim to 4-dim
        return tensor
        
        img = img[ top:bottom, left:right, : ]
        
        
        # plt.imshow(img)
        
    except Exception as e:
        print( "unable to get face" )
        
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)
 
    try:
        img = cv2.resize( img, (img_size[0], img_size[1]) )
    except Exception as e:
        print( "Error resizing from", img.shape, "to", img_size )
        raise( e )

    img = img.swapaxes( 0, 2 ) # swap height axis and color_depth axis
    # img = img.swapaxes( 1, 2 ) # swap height and width axis
    # Now, the img has shape [3, 224, 224] i.e. [ channel, height, width ]

    img = img.astype( np.float64 )
    
    img = torch.Tensor( img )
    img = img.unsqueeze(0) # add one more dimension before the data

    return img


#%%
class Process:
    """
    Set up as session (aka process) to serve the CNN model.
    
    Args:
        weight_loc (str):  the default location for the model weights
        
    """    
    def __init__( self, weight_loc, cuda_id = 0):
        self.cuda_id = cuda_id
        
        if cuda_id == "all":
            pass
        if cuda_id >= 0:
            loc = 'cuda:%d'% cuda_id
            cuda_id = -1
            torch.cuda.set_device( cuda_id )
        else:
            loc = 'cpu' 
            
        if not torch.cuda.is_available(): loc = "cpu"
            
        self.model = torch.load( weight_loc, map_location=loc )    
        # if the saved model is a data parallel model, de-data parallel 
        if hasattr( self.model, "module" ): self.model = self.model.module 


        self.model = self.model.eval()
        
        # For Action Units
        self.sigmoid = torch.nn.Sigmoid() 
        
        # Cuda
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model = self.model.cuda()
        
        # Copy Paste from my Dataloader
        self.AUs = [ "1", "2", "4", "5", "6", "9", "12", "17", "20", "25", "26", "43" ] # the actual AUs
        self.AUs = [ int(_) - 1 for _ in self.AUs ] # AU from string to index (0 indexing)
        self.AUs = sorted( self.AUs ) # sanity check
    
    
    def au_array_2_description( self, arr ):
        
        names = ["AU1 Inner Brow Raiser", 
                 "AU2 Outer Brow Raiser ", 
                 "AU4 Brow Lowerer", 
                 "AU5 Upper Lid Raiser",
                 "AU6 Cheek Raiser", 
                 "AU9 Nose Wrinkler", 
                 "AU12 Lip Corner Puller", 
                 "AU17 Chin Raiser	", 
                 "AU20 Lip stretcher", 
                 "AU25 Lips Part", 
                 "AU26 Jaw Drop", 
                 "AU43 Eyes Closed" ] # the actual AUs

        names = ["Inner Brow Raiser", 
                 "Outer Brow Raiser ", 
                 "Brow Lowerer", 
                 "Upper Lid Raiser",
                 "Cheek Raiser", 
                 "Nose Wrinkler", 
                 "Lip Corner Puller", 
                 "Chin Raiser", 
                 "Lip stretcher", 
                 "Lips Part", 
                 "Jaw Drop", 
                 "Eyes Closed" ] # the actual AUs

        rst = ""
        action_count = 0
        for i in range(len(names)):
            if arr[i] == 0: continue # if the val is zero, skip
#            rst += "%s: %d " % (names[i], arr[i])
            rst += "%s, " % names[i]
            action_count += 1
            #if action_count % 3 == 0 and action_count < 8: rst += "\n"
            
        return rst.strip()
  
    def run_one_img( self, img, imgname = None ):
        tic = time.time()
        inputs = img.float()
        
        if self.use_cuda: inputs = inputs.cuda()
        
#        pdb.set_trace()
        
        # Get the output
        outputs = self.model( inputs )
        output_exp = outputs[:,0:8]
        output_val = outputs[:,8] 
        output_aro = outputs[:,9] 
        output_au = self.sigmoid( outputs[:,10:22] )
        
        
        # AU: action units
        au_prediction = ( self.sigmoid( output_au ) > 0.5).data.cpu().numpy().reshape(-1).tolist()

        # Exp: expression
        val, idx = torch.max( output_exp, 1 )
        idx = idx.data.cpu().tolist()[0]
        expression = config.expressions[ idx ]
        
        confidence = output_exp[0,idx] / output_exp.sum()

        # VA: valence and arousal
        valence = output_val.item()
        arousal = output_aro.item()
        
        
        au_description = self.au_array_2_description( au_prediction )        
        msg = "%s (%.2f) %.2f %.2f\n%s" % (expression.upper(), confidence, valence, arousal, str(au_description) )
        img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)
        
        plt.cla(); plt.clf()
        plt.imshow( img )
        plt.title( msg )
        plt.axis( 'off' )
        
        if imgname is not None:
            imgname = os.path.basename( imgname )
            outname = imgname[ :imgname.rfind( "." ) ] + "_rst.jpg"
            plt.savefig( outname )
        
        dur = time.time() - tic
        print( "It takes %.3f seconds to process this image" % dur )
            
        
    def run_one_batch( self, inputs ):
        inputs = inputs.type(torch.FloatTensor)
        if self.use_cuda: inputs = inputs.cuda()
       
        outputs = self.model( inputs.float() )
        output_exp = outputs[:,0:8] 
        output_val = outputs[:,8] 
        output_aro = outputs[:,9] 
        output_au = self.sigmoid( outputs[:,10:22] )

        return output_exp, output_au, output_val, output_aro        
        
