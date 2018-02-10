# -*- coding: utf-8 -*-
from mxnet.gluon import nn
from mxnet import nd

class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                              strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                  strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)
    
class ResNet(nn.Block):
    def __init__(self, num_classes, n ,verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        def stack(n):
            out = nn.Sequential()
            out.add(nn.Conv2D(16,kernel_size=3, padding=1),
                    nn.BatchNorm(),
                    nn.Activation(activation='relu')
            )
            for i in range(n):
                if i == 0:
                    out.add(Residual(16,same_shape=False))
                else: out.add(Residual(16))
            for i in range(n):
                if i == 0:
                    out.add(Residual(32,same_shape=False))
                else:out.add(Residual(32))
            for i in range(n):
                if i == 0:
                    out.add(Residual(64,same_shape=False))
                else:out.add(Residual(64))
                return out                
        with self.name_scope():
            # block 1
#            b1 = nn.Sequential()
#            b1.add(nn.Conv2D(16, kernel_size=3,padding=1),
#                   nn.BatchNorm(),
#                   nn.Activation(activation='relu')
#            )
#            
#            # block 2
#            b2 = nn.Sequential()
#            b2.add(
#                Residual(16,same_shape=False),
#                Residual(16),
#                Residual(16)
#            )
#            # block 3
#            b3 = nn.Sequential()
#            b3.add(
#                Residual(32,same_shape=False),
#                Residual(32),
#                Residual(32)
#            )
#            # block 4
#            b4 = nn.Sequential()
#            b4.add(
#                Residual(64,same_shape=False),
#                Residual(64),
#                Residual(64)
#            )
#            # block 5
            b5 = nn.Sequential()
            b5.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(stack(n) , b5)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
    

import utils
from mxnet import gluon
import mxnet


train_data, test_data = utils.load_data_cifar10(batch_size=128)

ctx = utils.try_gpu()
epochs = 4
n=[3,5,7,9]

import matplotlib.pyplot as plt 
import numpy as np

def train_res(n_queue):
    his_testacc=[]
    for n in n_queue:
        net = ResNet(10,n) 
        net.initialize(ctx=ctx, init=mxnet.initializer.MSRAPrelu())
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': .1,'wd':.0001,'momentum':.9})
    
        his_testacc.append(utils.train(train_data, test_data, net, loss,
                    trainer, ctx, num_epochs=epochs))
    return his_testacc


def train_le(n_queue):
    his_testacc=[]
    def LeNet(n):
        out = nn.Sequential()
        with out.name_scope():
            out.add(nn.Conv2D(16,kernel_size=3, padding=1),
                    nn.BatchNorm(),
                    nn.Activation(activation='relu')
            )
            for i in range(2*n):
                if i == 0:
                    out.add(nn.Conv2D(channels=16, kernel_size=3,padding=1,strides=2))
                    out.add(nn.BatchNorm(axis=1))
                    out.add(nn.Activation(activation="relu"))
                else:
                    out.add(nn.Conv2D(channels=16, kernel_size=3,padding=1))
                    out.add(nn.Activation(activation="relu"))
            for i in range(2*n):
                if i == 0:
                    out.add(nn.Conv2D(channels=32, kernel_size=3,padding=1,strides=2))
                    out.add(nn.BatchNorm(axis=1))
                    out.add(nn.Activation(activation="relu"))
                else:
                    out.add(nn.Conv2D(channels=32, kernel_size=3,padding=1))
                    out.add(nn.Activation(activation="relu"))
            for i in range(2*n):
                if i == 0:
                    out.add(nn.Conv2D(channels=64, kernel_size=3,padding=1,strides=2))
                    out.add(nn.BatchNorm(axis=1))
                    out.add(nn.Activation(activation="relu"))
                else:
                    out.add(nn.Conv2D(channels=64, kernel_size=3,padding=1))
                    out.add(nn.Activation(activation="relu"))
            out.add(nn.AvgPool2D(pool_size=3))
            out.add(nn.Dense(10))
        return out
    for n in n_queue:
        net = LeNet(n) 
        net.initialize(ctx=ctx, init=mxnet.initializer.MSRAPrelu())
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': .1,'wd':.0001,'momentum':.9})
        his_testacc.append(utils.train(train_data, test_data, net, loss,
                    trainer, ctx, num_epochs=epochs))
    return his_testacc



his_testacc = train_res(n)
x_axis = np.linspace(0,epochs,len(his_testacc[0]))
plt.figure(figsize=(20,20))
plt.plot(x_axis,his_testacc[0],label='ResNet20')
plt.plot(x_axis,his_testacc[1],label='ResNet32')
plt.plot(x_axis,his_testacc[2],label='ResNet44')
plt.plot(x_axis,his_testacc[3],label='ResNet56')
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.legend()
plt.show()

his_testacc_le = train_le(n)
x_axis = np.linspace(0,epochs,len(his_testacc_le[0]))
plt.figure(figsize=(20,20))
plt.plot(x_axis,his_testacc_le[0],label='plain20')
plt.plot(x_axis,his_testacc_le[1],label='plain32')
plt.plot(x_axis,his_testacc_le[2],label='plain44')
plt.plot(x_axis,his_testacc_le[3],label='plain56')
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.legend()
plt.show()


