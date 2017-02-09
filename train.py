#!/usr/bin/env python
# coding=utf-8
from Gmm import *
import numpy as np
import math
import scipy.io as sio
import datetime

trainfile = "/home/ydliu/文档/algorithm/GMM/TrainSamples.mat"
trainlabelfile = "/home/ydliu/文档/algorithm/GMM/TrainLabels.mat"
dataSet = loadDataSet(trainfile)
trainlabel = loadDataSet(trainlabelfile)

N,M = np.shape(dataSet)

K = 10
centroids, clusterAssment,computerindex = kmeans(dataSet,K,N,M)
gaussindex,variance = getGaussIndex(dataSet,centroids,clusterAssment,computerindex,N,M,K)
label = whichLabel(clusterAssment,trainlabel,K)
sio.savemat("/home/ydliu/文档/algorithm/GMM/data.mat",{'centroids':centroids,'gaussindex':gaussindex,'variance':variance,'label':label})

print "Train complete!!"

