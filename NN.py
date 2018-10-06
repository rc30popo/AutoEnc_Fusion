# -*- Coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable

chainer.config.train = False

# Class NN

class AutoEnc6(Chain):
    def __init__(self):
        super(AutoEnc6, self).__init__(
            conv1 = L.Convolution2D(in_channels=1, out_channels=32, ksize=5, stride=2, pad=2),
            conv2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=2, pad=2),
            conv3 = L.Convolution2D(in_channels=64, out_channels=128, ksize=5, stride=2, pad=2),
            conv4 = L.Convolution2D(in_channels=128, out_channels=256, ksize=5, stride=2, pad=2),            
            bn1 = L.BatchNormalization(32),
            bn2 = L.BatchNormalization(64),
            bn3 = L.BatchNormalization(128),
            bn4 = L.BatchNormalization(256),
            l0 = L.Linear(None, 16),
            l1 = L.Linear(None,256 * 4 * 4),
            deconv1 = L.Deconvolution2D(in_channels=256,out_channels=128,ksize=5,stride=2,pad=1),
            deconv2 = L.Deconvolution2D(in_channels=128,out_channels=64,ksize=5,stride=2,pad=2),
            deconv3 = L.Deconvolution2D(in_channels=64,out_channels=32 ,ksize=5,stride=2,pad=2),
            deconv4 = L.Deconvolution2D(in_channels=32 ,out_channels=1  ,ksize=5,stride=2,pad=2),
            bn5 = L.BatchNormalization(256 * 4 * 4),
            bn6 = L.BatchNormalization(128),
            bn7 = L.BatchNormalization(64),
            bn8 = L.BatchNormalization(32)        
        )
    def enc(self, x,ratio=0.5):
        h = F.reshape(x,(len(x),1,64,64))
        h = F.dropout(F.leaky_relu(self.bn1(self.conv1(h))),ratio=ratio)
        h = F.dropout(F.leaky_relu(self.bn2(self.conv2(h))),ratio=ratio)
        h = F.dropout(F.leaky_relu(self.bn3(self.conv3(h))),ratio=ratio)
        h = F.dropout(F.leaky_relu(self.bn4(self.conv4(h))),ratio=ratio)
        h = self.l0(h)
        return h
    def dec(self,x,ratio=0.5):
        h = F.dropout(F.leaky_relu(self.bn5(self.l1(x))),ratio=ratio)
        h = F.reshape(h,(len(h),256,4,4))
        h = F.dropout(F.leaky_relu(self.bn6(self.deconv1(h))),ratio=ratio)
        h = F.dropout(F.leaky_relu(self.bn7(self.deconv2(h))),ratio=ratio)
        h = F.dropout(F.leaky_relu(self.bn8(self.deconv3(h))),ratio=ratio)
#        h = F.tanh(self.deconv4(h))
        h = F.sigmoid(self.deconv4(h))
        h = h[:,:,:-1,:-1]
        h = F.reshape(h,(len(h),4096))
        return h
    def forward(self,x,ratio=0.5):
        h = self.enc(x,ratio=ratio)
        h = self.dec(h,ratio=ratio)
        return h

            
            
            

