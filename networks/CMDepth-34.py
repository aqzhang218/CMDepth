

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from layers import *


class DepthEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))

        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])
        self.ACM0 = ACM(num_ch_enc[0], num_ch_enc[0])
        self.ACM1 = ACM(num_ch_enc[1], num_ch_enc[1])
        self.ACM2 = ACM(num_ch_enc[2], num_ch_enc[2])
        self.ACM3 = ACM(num_ch_enc[3], num_ch_enc[3])

        # decoder
        self.convs = OrderedDict()



   
        self.Mambablocks = VSSBlock(num_ch_enc[-1], num_ch_enc[-1])
        self.cov4 = ConvBlock(512, 256)
        self.cov3 = ConvBlock(256, 128)
        self.cov2 = ConvBlock(128, 64)
        self.cov1 = ConvBlock(64, 64)
        self.TFAM4 = TFAM(256)
        self.TFAM3 = SFF(128)
        self.TFAM2 = SFF(64)
        self.TFAM1 = SFF(64)




        self.convs= Conv3x3(64, self.num_output_channels)

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        input_features[0] = self.ACM0(input_features[0])
        input_features[1] = self.ACM1(input_features[1])
        input_features[2] = self.ACM2(input_features[2])
        input_features[3] = self.ACM3(input_features[3])



        x = input_features[-1]
        x = x.permute(0, 2, 3, 1)
        x = self.Mambablocks(x)
        x = x.permute(0, 3, 1, 2)



        x = self.cov4(x)
        x = upsample(x)
        x = self.TFAM4(input_features[3],x)

        x = self.cov3(x)
        x = upsample(x)
        x = self.TFAM3(input_features[2], x)

        x = self.cov2(x)
        x = upsample(x)
        x = self.TFAM2(input_features[1], x)

        x = self.cov1(x)
        x = upsample(x)
        x = self.TFAM1(input_features[0], x)

        x = upsample(x)

        self.outputs[("disp", 0)] = self.sigmoid(self.convs(x))




        return self.outputs
