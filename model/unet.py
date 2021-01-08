import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from model.model_helper import *

class Unet(nn.Module):
    def __init__(self, num_ch, num_class, bilinear = True):
        '''
        parameters 
            num_ch (int)
                number of ch
            num_class (int)
                number of class (output 으로 결정하기를 원하는 class 갯수, segmentation)
            bilinear (boolean)
                Decoder block에서 up_conv룰 어떻게 할 지 결정함
        '''

        super(Unet,self).__init__()

        self.num_ch = num_ch
        self.num_class = num_class
        self.bilinear = bilinear

        self.init_conv= DoubleConv(num_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128 ,64, bilinear)

        self.out = OutConv(64, num_class)
        
    def forward(self, x):
        x1 = self.init_conv(x)

        encoder_x1 = self.down1(x1)
        encoder_x2 = self.down2(encoder_x1)
        encoder_x3 = self.down3(encoder_x2)

        decoder_x1 = self.up1(encoder_x3, encoder_x2)
        decoder_x2 = self.up2(decoder_x1, encoder_x1)
        decoder_x3 = self.up3(decoder_x2, x1)

        logits = self.out(decoder_x3)

        return logits


