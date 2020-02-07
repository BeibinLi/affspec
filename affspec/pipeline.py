import os
import sys

sys.path.append(os.path.realpath(__file__))


from .dataloader.custom_transforms import FaceCrop
from torchvision import transforms

import torch


import numpy as np

#from .models import eespnet, mobilenet, resnet

from .config import expressions

import matplotlib.pyplot as plt
import face_recognition as fr
import cv2

import copy
import time
import pdb
import random
import glob
import warnings


#%%
# Default Weights Locations
ESP_WEIGHT = "affspec/weights/model_esp.pth"
MOB_WEIGHT = "affspec/weights/model_mob.pth"
RES_WEIGHT = "affspec/weights/model_res.pth"


img_size = [224, 224, 3]
OUT_DIR = "image/"
transformer = transforms.Compose([FaceCrop(), 
                                  transforms.Resize(size=[224, 224]), 
                                  transforms.ToTensor()])

#%%

def detect_face(img):
    """
    Detect a face from RGB image. If there are multiple faces in this image, then
    return the largest face detected.

    Args:
        img (numpy.array): the input image

    Returns:
        t, r, b, l (int): top, right, bottom, left of the face
    """    
    all_locations = fr.face_locations(img , model = 'hog') # get locations of all faces

    max_area = 0
    face_location = None, None, None, None

    for x in all_locations: # x is the location for one face
        face_size = (x[2] - x[0]) * (x[3] - x[1])

        if abs(face_size) > max_area:
            face_location = copy.copy(x)

    t, r, b, l = face_location
    return t, r, b, l



class NoFaceError(Exception):
    pass


def img2torch(img, detect_face_before_process=True):
    """
    Detect and crop the face, and then convert the largest face to PyTorch tensor
    
    If the user asks to detect the face, but there is no face in the image,
    it will raise an error.

    Args:
        img (numpy.array): the input image as the numpy array with [h, w, 3] size.
        detect_face_before_process (bool): Run face detection algorithm to find the face or not
        
    Returns:
        img (torch.Tensor): the input image as PyTorch object with [3, 224, 224] size.        
    """  
    if detect_face_before_process:
        top, right, bottom, left = detect_face(img)
        
        if top is None:
            # No face detected in the image, but the user want to detect face
            raise(NoFaceError("No face in the image"))
    else:
        top, right, bottom, left = None, None, None, None
        
    img_dict = {"image": img, "top":top, "bottom":bottom, "left":left, "right":right}
    tensor = transformer(img_dict)
    tensor = tensor.unsqueeze(0) # make 3-dim to 4-dim
    return tensor        



#%%
class Process:
    """
    Set up as session (aka process) to serve the CNN model.
    
    Args:
        backbone (str): either "esp" (ESPNet) or "mob" (MobileNet) or 
            "res" (ResNet), which defines the backbone of CNN.
        cuda_id (str or int): defines which CUDA to use. If equals "all", then
            it will use all CUDAs.
    """    
    def __init__(self, backbone="esp", cuda_id= 0):
        self.cuda_id = cuda_id
        
        if cuda_id == "all":
            pass
        if cuda_id >= 0:
            loc = 'cuda:%d'% cuda_id
            cuda_id = -1
            torch.cuda.set_device( cuda_id )
        else:
            loc = 'cpu' 
            
        if not torch.cuda.is_available(): 
            loc = "cpu"

        backbone = backbone[:3]
        if backbone == "esp":
            from .models.eespnet import EESPNet
            self.model = EESPNet(classes=22, s=2)
            weights = torch.load(ESP_WEIGHT, map_location=loc)
        elif backbone == "mob":
            from .models.mobilenet import MobileNets
            self.model = MobileNets(classes=22, s=2)
            weights = torch.load(MOB_WEIGHT, map_location=loc)
        elif backbone == "res":
            from .models.resnet import ResNet
            self.model = ResNet(classes=22, s=2)
            weights = torch.load(RES_WEIGHT, map_location=loc)
        self.model.load_state_dict(weights)

        self.model = self.model.eval()
        
        # For Action Units
        self.sigmoid = torch.nn.Sigmoid() 
        self.softmax = torch.nn.Softmax(dim=0)
        
        # Cuda
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model = self.model.cuda()
        
        self.AUs = ["1", "2", "4", "5", "6", "9", "12", "17", 
                    "20", "25", "26", "43"] # the actual AUs
        # AU from string to index (0 indexing)
        self.AUs = [int(_) - 1 for _ in self.AUs] 
        self.AUs = sorted( self.AUs ) # sanity check
        
        
        self.rst_placeholder = {
               "imgname": None, 
               "expression": None,
               "expression confidence": None,
               "valence": None,
               "arousal": None,
               "action units": None
               }
    
    

  
    def run_one_img(self, img=None, imgname=None, detect_face_before_process=True):
        """
        Process one image. 
        
        Args:
            img (np.array): size with [h, w, 3]. 
            imgname (str): image name. If this is specified, then it will ignore 
                the other arguments
            detect_face_before_process (bool): Run face detection algorithm to find the face or not
                
        Output:
            rst (dict): a dictionary for results.
        """


        if imgname is not None:
            try:
                img = cv2.imread(imgname)
            except:
                print("Unaboe to get the image:", imgname)
                return self.rst_placeholder
        
        try:
            inputs = img2torch(img, detect_face_before_process)
        except NoFaceError:
            # Not face detected
            rst = copy.deepcopy(self.rst_placeholder)
            rst["imgname"] = imgname if imgname is not None else None
            return self.rst_placeholder
        
        inputs = inputs.float()
        if self.use_cuda:
            inputs = inputs.cuda()
                
        # Get the output
        outputs = self.model(inputs)
        rst = self.output_2_dict(outputs)
        
        if imgname is not None:
            rst["imgname"] = imgname
        
        return rst
        
        
    def output_2_dict(self, outputs):
        """
        Args:
            outputs (torch.tensor): the output with 22 elements, including 
                    information with expression, AU, valence, and arousal
            
        Output:
            rst (dict): a dictionary for results.
        """
        outputs = outputs.reshape(-1)
        assert(outputs.shape[0] == 22)
        output_expression = outputs[0:8]
        output_val = outputs[8] 
        output_aro = outputs[9] 
        output_au = self.sigmoid(outputs[10:22])
        
        
        # AU: action units
        au_prediction = (output_au > 0.5).data.cpu().numpy().reshape(-1).tolist()

        # Exp: expression
        val, idx = torch.max(output_expression, 0)
        idx = idx.data.cpu().tolist()
        expression = expressions[idx]
        confidence = self.softmax(output_expression)[idx].item()
        
        # VA: valence and arousal
        valence = output_val.item()
        arousal = output_aro.item()
        
        rst = {"expression": expression,
               "expression confidence": confidence,
               "valence": valence,
               "arousal": arousal,
               "action units": au_prediction}
        
        return rst    
        
    def run_imgs(self, imagenames, batch_size=3, detect_face_before_process=True):
        """
        Process a list of images
        
        Args:
            imagenames (list): a list of images
            batch_size (int): number of images to process per iteration. If 
                the GPU is powerful, it can be 10 or more. If only CPU is used,
                then the batch size should be smaller than 3.
            detect_face_before_process (bool): Run face detection algorithm to find the face or not
            
        Output:
            rst (list): a list of outputs, where each element is a dictionary
                that stores the results for one image.
        """
        assert(type(imagenames) is list)
        
        rsts = [self.rst_placeholder] * len(imagenames)
        
        for i in range(0, len(imagenames), batch_size):
            n_in_batch = len(imagenames[i:i + batch_size])
            inputs = []
            inputs_indices = []
            
            for j in range(n_in_batch):
                imgname = imagenames[i + j]
                try:
                    img = cv2.imread(imgname)
                    img = img2torch(img, detect_face_before_process)
                except Exception as e:
                    if e is NoFaceError:
                        warnings.warn("Unable to find the face in: %s" % imgname, RuntimeWarning)
                    else:
                        warnings.warn("Unable to read image: %s" % imgname, RuntimeWarning)
                    continue
                
                inputs.append(img)
                inputs_indices.append(i + j)
                    

            inputs = torch.cat(inputs, dim=0)

            inputs = inputs.type(torch.FloatTensor)
            if self.use_cuda: 
                inputs = inputs.cuda()
           
            outputs = self.model(inputs.float())

            for j in range(inputs.shape[0]):
                rst = self.output_2_dict(outputs[j, :])
                actual_idx = inputs_indices[j]
                rsts[actual_idx] = rst
                
                
        for i in range(len(imagenames)):
            rsts[i]["imgname"] = imagenames[i]
            
            
        return rsts        
        
