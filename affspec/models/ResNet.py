from torch.nn import init
import torch.nn.functional as F
from .cnn_utils import *
import math
import torch

__author__ = "Sachin Mehta"
__version__ = "1.0.1"
__maintainer__ = "Sachin Mehta"

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResidualBottleNeckBlock(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, down_method='esp'): #down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        self.conv1 = conv1x1(nIn, nOut//4)
        self.bn1 = nn.BatchNorm2d(nOut//4)
        self.relu1 = nn.PReLU(nOut//4)
        self.conv2 = conv3x3(nOut//4, nOut//4, stride)
        self.bn2 = nn.BatchNorm2d(nOut//4)
        self.relu2 = nn.PReLU(nOut//4)
        self.conv3 = conv1x1(nOut//4, nOut)
        self.bn3 = nn.BatchNorm2d(nOut)
        self.relu3 = nn.PReLU(nOut)

        self.downAvg = True if down_method == 'avg' else False

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 2 and self.downAvg:
            return out

        out += residual

        return self.relu3(out)


class DownSampler(nn.Module):
    '''
    Down-sampling fucntion that has three parallel branches: (1) avg pooling,
    (2) EESP block with stride of 2 and (3) efficient long-range connection with the input.
    The output feature maps of branches from (1) and (2) are concatenated and then additively fused with (3) to produce
    the final output.
    '''

    def __init__(self, nin, nout, reinf=True):
        '''
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        '''
        super().__init__()
        nout_new = nout - nin
        self.eesp = ResidualBottleNeckBlock(nin, nout_new, stride=2, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(config_inp_reinf, config_inp_reinf, 3, 1),
                CB(config_inp_reinf, nout, 1, 1)
            )
        self.act =  nn.PReLU(nout)

    def forward(self, input, input2=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)

        if input2 is not None:
            #assuming the input is a square image
            # Shortcut connection with the input image
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.act(output)

class ResNet(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self, classes=1000, s=1):
        '''
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        reps = [0, 3, 7, 3]  # how many times EESP blocks should be repeated at each spatial level.
        channels = 3

        r_lim = [13, 11, 9, 7, 5]  # receptive field at each spatial level
        K = [4]*len(r_lim) # No. of parallel branches at different levels

        base = 32 #base configuration
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i== 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(512)
        elif s <= 3.0:
            config.append(512)
        else:
            ValueError('Configuration not supported')


        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True # True for the shortcut connection with input

        assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(channels, config[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(config[0], config[1], reinf=self.input_reinforcement)  # out = 56

        self.level3_0 = DownSampler(config[1], config[2], reinf=self.input_reinforcement) # out = 28
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(ResidualBottleNeckBlock(config[2], config[2], stride=1))

        self.level4_0 = DownSampler(config[2], config[3], reinf=self.input_reinforcement) #out = 14
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(ResidualBottleNeckBlock(config[3], config[3], stride=1))

        self.level5_0 = DownSampler(config[3], config[4]) #7
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(ResidualBottleNeckBlock(config[4], config[4], stride=1))

        # expand the feature maps using depth-wise convolution followed by group point-wise convolution
        self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(CBR(config[4], config[5], 1, 1, groups=1))

        self.classifier = nn.Linear(config[5], classes)
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_l1 = self.level1(input)  # 112
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        out_l3_0 = self.level3_0(out_l2, input)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        #out_l3 = F.dropout(out_l3, p=p, training=self.training)

        out_l4_0 = self.level4_0(out_l3, input)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        #out_l4 = F.dropout(out_l4, p=p, training=self.training)

        out_l5_0 = self.level5_0(out_l4)  # down-sample
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)
        #out_l5 = F.dropout(out_l5, p=p, training=self.training)

        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)

        return self.classifier(output_1x1)


if __name__ == '__main__':
    input = torch.Tensor(1, 3, 224, 224).cuda()
    model = EESPNet(classes=1000, s=1.0)
    out = model(input)
    print('Output size')
    print(out.size())
