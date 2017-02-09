#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math
import scipy.io as sio
from Gmm import *

#zhuanhuan geshi
filename = "/home/ydliu/文档/algorithm/GMM/TestSamples.mat"
dataset = loadDataSet(filename)
N,M = dataset.shape

for i in xrange(N):
	dataset[i,:] = dataset[i,:]/1000
sio.savemat("/home/ydliu/文档/algorithm/GMM/TTestSamples.mat",{'newtest':dataset})