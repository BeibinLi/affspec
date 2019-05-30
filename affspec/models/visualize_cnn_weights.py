import os
import time
import pdb

import torch.optim as optim
import torch.nn as nn
import torch.functional as F
import torch

import numpy as np
import matplotlib.pyplot as plt

from EESPNet_test import EESPNet
# %%



out_dir = "out/"

    
#%% Visualzie the first convolutional layer's weights
def visualize_weights_first_layer( model ):
    for out_channel in range(64):
    #    for in_channel in range(3):
        patch = model.conv1.weight[out_channel, :].data.cpu().numpy()
        patch = np.swapaxes(patch,0,2)
        
        patch = ( patch - np.min(patch) ) / (np.max(patch) - np.min(patch) )
        plt.imshow( patch )
        
        plt.savefig( os.path.join( out_dir, "conv1_weights_%d.jpg" % (out_channel) ) )
        
    
#%%
class Visualization:
    def __init__( self, model, use_cuda = True):
        self.model = model        
        self.use_cuda = use_cuda
        
        if use_cuda:
            model = model.cuda()
        
        self.attach_hook()
        
        self.sigmoid = nn.Sigmoid()
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD( model.parameters(), lr=0.00001, momentum=0.9)

        
    def attach_hook( self ):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_outputs = grad_out # 0 means we want the result for the first img

        # Hook the selected layer
        self.all_layer_outputs = []
#        self.model.conv1.register_forward_hook( lambda m, i, o: self.conv_outputs = o[0] )
#        self.model.layer1.register_forward_hook( lambda m, i, o: self.layer1_outputs = o[0] )
#        self.model.beibin_intro.register_forward_hook( lambda m, i, o: self.all_layer_outputs.append( o[0] ) )
        self.model.level1.conv.register_forward_hook( lambda m, i, o: self.all_layer_outputs.append( o[0] ) )
        self.model.level4[-1].conv_1x1_exp.conv.register_forward_hook( lambda m, i, o: self.all_layer_outputs.append( o[0] ) )
#        self.model.layer1.register_forward_hook( lambda m, i, o: self.all_layer_outputs.append( o[0] ) )
#        self.model.layer2.register_forward_hook( lambda m, i, o: self.all_layer_outputs.append( o[0] ) )
#        self.model.layer3.register_forward_hook( lambda m, i, o: self.all_layer_outputs.append( o[0] ) )
#        self.model.layer4.register_forward_hook( lambda m, i, o: self.all_layer_outputs.append( o[0] ) )
#        
#        
        self.all_backward_outputs = []
#        self.model.beibin_intro.register_backward_hook( lambda m, i, o: self.all_backward_outputs.append( o[0] ) )
        self.model.level1.conv.register_backward_hook( lambda m, i, o: self.all_backward_outputs.append( o[0] ) )
        self.model.level4[-1].conv_1x1_exp.conv.register_backward_hook( lambda m, i, o: self.all_backward_outputs.append( o[0] ) )
#        self.model.layer2.register_backward_hook( lambda m, i, o: self.all_backward_outputs.append( o[0] ) )
#        self.model.layer3.register_backward_hook( lambda m, i, o: self.all_backward_outputs.append( o[0] ) )
#        self.model.layer4.register_backward_hook( lambda m, i, o: self.all_backward_outputs.append( o[0] ) )
##        self.model.layer4.register_backward_hook( lambda m, i, o: print(i, o) )
#
        self.all_backward_inputs = []
#        self.model.beibin_intro.register_backward_hook( lambda m, i, o: self.all_backward_inputs.append( i[0] ) )
        self.model.level1.conv.register_backward_hook( lambda m, i, o: self.all_backward_inputs.append( i[0] ) )
        self.model.level4[-1].conv_1x1_exp.conv.register_backward_hook( lambda m, i, o: self.all_backward_inputs.append( i[0] ) )
#        self.model.layer2.register_backward_hook( lambda m, i, o: self.all_backward_inputs.append( i[0] ) )
#        self.model.layer3.register_backward_hook( lambda m, i, o: self.all_backward_inputs.append( i[0] ) )
#        self.model.layer4.register_backward_hook( lambda m, i, o: self.all_backward_inputs.append( i[0] ) )
        
        
    def viz_one_layer( self, whole_filter, ofname ):
        
        # For backward pass, we need to extract one more dimension
        if ofname.find( "back" )>=0: whole_filter = whole_filter[0]
        
#        pdb.set_trace()
        
#        filters, idx = F.softmax(whole_filter, 0).max(0)
        filters, idx =  whole_filter.max(0)
        
        filters = filters.data.cpu().numpy()
#        pdb.set_trace()
#        print( filters )
        return filters
            
#        plt.imshow( filters )
#        plt.savefig( os.path.join( dirname, "%s.jpg" % (ofname ) ) )
        
    
    def visualise_layer_with_hooks(self, inputs, labels ):
        self.all_layer_outputs = []
        self.all_backward_outputs = []
        self.all_backward_inputs = []

        # Process image and return variable
        img = inputs.cpu().numpy()
        img = img[0] # get the frsit in batch
        img = img.swapaxes( 0, 2 )
        img = img.swapaxes( 0, 1 )
        img = img.astype(np.uint8)

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        
        
        if self.use_cuda:                 
            inputs = inputs.cuda()
            labels = labels.cuda()
        
#        inputs, labels, weights = Variable(inputs), Variable(labels), Variable(weights)
        
        outputs = self.model( inputs )
        print("output size", outputs.shape)
        
        # todo: remove. This is used for debugging
        labels = outputs.clone()
        center_idx = labels.shape[3] // 2
        labels[0,:,center_idx,center_idx] = 0.5
        print(labels[0,0,center_idx,center_idx], outputs[0,0,center_idx,center_idx])
        
        loss = self.criterion( outputs, labels )
        
        loss.backward()
        self.optimizer.step()
        
        fig=plt.figure(figsize=(8, 8))
        
        
        rows = 4
        cols = 5
        sub_plt_container = []
        
        sub_plt_container.append( fig.add_subplot(rows, cols, 1) )
        plt.imshow( img )
        
#        sub_plt_container.append( fig.add_subplot(rows, cols, 3) )
#        plt.title( "Pred :" + ",".join([ '%.2f' % _ for _ in pred.tolist()] ) + "\nLabel:" + ",".join([ "%.2f" % _ for _ in lab.tolist()] ) )

        for i, layer in enumerate( self.all_layer_outputs ):
            x = self.viz_one_layer( layer, "layer_%d" % i )            
            sub_plt_container.append(  fig.add_subplot(rows, cols, cols + i + 1 )  )
            plt.imshow(x)
            
        for i, layer in enumerate( self.all_backward_inputs ):
            if layer is None: continue
            x = self.viz_one_layer( layer, "back_layer_in_%d" % i )            
            sub_plt_container.append(  fig.add_subplot(rows, cols, 2 * cols +i+ 1)  )
            plt.imshow(x)
            
        for i, layer in enumerate( self.all_backward_outputs ):
            if layer is None: continue
            x = self.viz_one_layer( layer, "back_layer_out_%d" % i )        
            sub_plt_container.append( fig.add_subplot(rows, cols, 3 * cols + i + 1) )
            plt.imshow(x)
#        pdb.set_trace()
            
        # Perform action for all subplots in the figure
#        for ax in sub_plt_container:
#            ax.set_xticks([])
#            ax.set_yticks([])
                
        plt.savefig( os.path.join( out_dir, "%f.png" % ( time.time() ) ),  dpi=300 )
        
        plt.close()
        return outputs
    
    
    
#%%
if __name__ == "__main__":
    
    
    for i in range(10):
        img_width = 300
        inputs = torch.randn(1, 3, img_width, img_width).cuda()
    #    inputs = torch.ones((1, 3, img_width, img_width)).cuda()
        model = EESPNet(classes=1000, s=2)
        model = model.cuda()
    #    out = model(inputs)
    #    print('Output size')
    
    
#        print(model.level1.conv.weight.grad)
    #    print(out.size())
        
    
        #%%
        viz = Visualization( model, True )
        
    
        viz.visualise_layer_with_hooks( inputs , inputs.clone()  ) # use inputs.clone to supress error.
    
