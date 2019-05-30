from dataloader import SimonsLoader

import pipeline


import os, time, glob, pdb, re
import tqdm

import numpy as np
import pandas as pd

import torch

import sys
import cv2

#%%



#%%
def process_csv(csv_name, skip_if_exists=True):
    """ 
    Process a CSV file

    Args:
    csv_name (str): file path for the csv file
    skip_if_exists (bool): if set True, then it will skip processing the already processed images


    Returns:
    None (None): nothing

    """
    #%%
    batch_size = 32
    
    out_name = csv_name.replace( ".csv", "_expression2.csv" )
    
    
#    pdb.set_trace()
    pid = re.findall( ".*esults.*\d+\W+(\w+).*face_location.csv", csv )[0]
    
    out_name = os.path.join( outdir , pid + ".csv" )

    if skip_if_exists and os.path.exists(out_name):
        print(out_name, "already processed. Skip!")
        return

    print( "begin process", pid , "-"*30)
    #%%
    f = open( out_name, "w" )
    f.write( "loc," + ",".join( expressions[0:8] + AUs) + ",valence,arousal\n"  )
    #rst = session.run_one_img( img  )
    
    
    # IMPORTANT ! the following three lines should be replaced to read image from video
    # Then crop the face, and cast the face image to pytorch tensor
    dataset = SimonsLoader.VideoDataset( csv_name, base_path = "D:/Simons_iPad_Seattle" ) 
    testloader = torch.utils.data.DataLoader( dataset, batch_size = batch_size, shuffle = False )
    
    
    for imgpaths, inputs in tqdm.tqdm( testloader ):
        try:
            exp_outputs, au_probability, valence, arousal  = session.run_one_batch( inputs ) # inputs is [b, 224*224*3]
        except Exception as e:
            print(e)
            print( "Error!!!!: Programming BUG" )
            print( "#" * 60 )
            print( imgpaths )
#            continue
#            pdb.set_trace()
        
        curr_batch_size = inputs.size()[0]
        for i in range(curr_batch_size):
            data = [ imgpaths[i] ]
            # IMPORTANT: output the following four lines
            data += exp_outputs[i,:].cpu().data.tolist()
            data += au_probability[i,:].cpu().data.tolist()
            data += [ valence[i].item() ]
            data += [ arousal[i].item() ]
            
            f.write(  ",".join( [str(_) for _ in data]) + "\n" )       
            
    
    f.close()


def process_dir( dirname ):
    files = glob.glob(  os.path.join(dirname, "*") )
    files = sorted(files)
    
    from skimage import io
    
    for imgname in files:        
        img = io.imread( imgname )
        img2 = pipeline.img2torch(img)
        expression, au_prediction, valence, arousal =  session.run_one_batch(img2)
        exp, idx = torch.max( expression, 1 )
        
        print(imgname)
        print(expression)
        # pdb.set_trace()
        print(expressions[idx])
        
#%%
    
if __name__ == "__main__":
    process_dir( "/Users/deepalianeja/Desktop/combined_learning/user1/" )
    
#    print( "begin" )
#    folders =  glob.glob( "D:/Simons_iPad_Seattle/results- ADULT/20*" )
#
##     folders = glob.glob( "D:/Simons_iPad_Seattle/Results ASD/20*" ) \
##        + glob.glob( "D:/Simons_iPad_Seattle/Results TD/20*" )  \
##        + glob.glob( "D:/Simons_iPad_Seattle/Results ASD-sib/20*" )  \
##        + glob.glob( "D:/Simons_iPad_Seattle/Results DD/20*" )  \
##        + glob.glob( "D:/Simons_iPad_Seattle/Results Interns/20*" ) \
##        + glob.glob( "D:/Simons_iPad_Seattle/results- ADULT/20*" )
#    
#    for fold in folders:
#        csv = glob.glob( os.path.join( fold, "*face_location.csv") ) [0]
#        try:
#            process_csv( csv )
#            print(csv, "done") 
#        except Exception as e:
#            print(e)
#    print( "All set" )
    
