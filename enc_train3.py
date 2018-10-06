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

from NN import AutoEnc6

# My data

input_dirs = ['./base/']

# Learned data
model_data_file = './enc_model6.npz'
optim_data_file = './enc_optim6.npz'


img_x = 64
img_y = 64

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

total_datacount = len(x_train_data)

x_train = np.array(x_train_data,dtype=np.float32)
x_train /= 255

print('x_train len = %d' % len(x_train))
#print('t_train len = %d' % len(t_train))

x_train_data = None


# Initialize Neural Network
model = AutoEnc6()

chainer.config.train = True

optimizer = optimizers.Adam()

optimizer.setup(model)

cont_flag = False
if cont_flag == True:
    serializers.load_npz(model_data_file, model)
    serializers.load_npz(optim_data_file, optimizer)

# epoch
n_epoch = 1000

# batch size
batch_size = 100

# Intermediate save
s_epoch = 50

print(' Number of test data = %d' % total_datacount)


for epoch in range(n_epoch):
    if epoch != 0 and s_epoch != 0 and (epoch % s_epoch == 0):
        print('Save intermediate models')
        serializers.save_npz(model_data_file, model)
        serializers.save_npz(optim_data_file, optimizer)
        
    sum_loss = 0
    perm = np.random.permutation(total_datacount)

    for i in range(0,total_datacount,batch_size):
        if i + batch_size <= total_datacount:
            local_batch_size = batch_size
        else:
            local_batch_size = total_datacount - i
#        print('i = %d' % i)
#        print('local_batch_size = %d' % local_batch_size)
        x = Variable(x_train[perm[i:i+local_batch_size]])
#        t = Variable(t_train[perm[i:i+local_batch_size]])
#        x = Variable(x_train)
#        t = Variable(t_train)
        y = model.forward(x,ratio=0.2)
        model.cleargrads()
        loss = F.mean_squared_error(y,x)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data*local_batch_size

    print("epoch: {0}, mean loss: {1}".format(epoch,sum_loss/total_datacount))


# Save Result
serializers.save_npz(model_data_file, model)
serializers.save_npz(optim_data_file, optimizer)
