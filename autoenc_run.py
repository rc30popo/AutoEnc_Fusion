# -*- Coding: utf-8 -*-

# OS
import os
# OpenCV
import cv2
# Numpy
import numpy as np

# Chainer
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable
from chainer import serializers
# random
#import random
from numpy.random import *
# Matplotlib
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

# My Neural Network
from NN import AutoEnc6

# Image size
img_x = 64
img_y = 64

# test data
input_dirs = ['./base3/']
output_dir = './test_out/'
# Output FilenameBase
output_base = 'img_'

#drop_ratio = 0.1


def make_outfname(out_dir,out_base,index):
    outfname = '{dir}{fname}{num:05}.jpg'.format(dir=out_dir,fname=out_base,num=index)
    return outfname

def out_genimages(out_dir,out_base,gen_output):
    global img_x,img_y
    outnum = len(gen_output)
#    print('img_x = %d' % img_x)
#    print('img_y = %d' % img_y)
#    print('out img num = %d' % outnum)
    for i in range(outnum):
        t_result = []
        t_result.append(gen_output[i].data)
        t_result = np.array(t_result, dtype=np.float32)
        t_result = t_result.flatten()
        t_result *= 255

        t_result = t_result.reshape(img_x,img_y)
        out_filename = make_outfname(out_dir,out_base,i)
        print('output file = %s' % out_filename)
        cv2.imwrite(out_filename,t_result)



x_train_data = []

for base_dir in input_dirs:
    train_files = os.listdir(base_dir)

    for image in train_files:
        if image.endswith('.jpg') or image.endswith('.png'):
            img = cv2.imread(base_dir + image)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(img_x,img_y),interpolation=cv2.INTER_AREA)
            img_gs = img.flatten()
#        print('img _gs len = %d' % len(img_gs))
            x_train_data.append(img_gs)
#        t_train_data.append(img_gs)

total_datacount = len(x_train_data)

x_train = np.array(x_train_data,dtype=np.float32)
x_train /= 255

print('x_train len = %d' % len(x_train))

x_train_data = None


# Learned data
enc_model_data_file = './enc_model6.npz'



enc = AutoEnc6()
serializers.load_npz(enc_model_data_file, enc)
chainer.config.train = False

x = Variable(x_train)
y = enc.forward(x)

out_genimages(output_dir,output_base,y)

