#!/usr/bin/env python
# coding=utf-8
from Gmm import *
import numpy as np
import math
import scipy.io as sio
import datetime

def lloadDataSet(filename):
	centroids1 = []
	gaussindex1 = []
	variance1 = []
	label1 = []
        #import the data in .mat
	dic = sio.loadmat(filename)
	print dic
	for key in dic:
		if key == 'centroids':
			centroids1.append(dic[key])
		if key == 'gaussindex':
			gaussindex1.append(dic[key])
		if key == 'variance':
			variance1.append(dic[key])
		if key == 'label':
			label1.append(dic[key])

	centroids = centroids1[0];
	gaussindex = gaussindex1[0]
	variance = variance1[0]
	label = label1[0]

	return centroids,gaussindex,variance,label
	#print localdata

        #the format is [1,N,M], so should perform locatedata[0]



begintime = datetime.datetime.now()

K =10

testfile = "/home/ydliu/文档/algorithm/GMM/TTestSamples.mat"
testlabel = "/home/ydliu/文档/algorithm/GMM/TestLabels.mat"
testData = loadDataSet(testfile)
testDataLabel = loadDataSet(testlabel)

trainfile = "/home/ydliu/文档/algorithm/GMM/data.mat"

centroids,gaussindex,variance,label = lloadDataSet(trainfile)



NN,MM = np.shape(testData)
M = MM
N = NN
print NN,MM
#comvariance = np.zeros((K,1))
#testClass = np.zeros((NN,1))
#examLabel = np.zeros((NN,1))
comvariance = [0 for i in xrange(K)]
testClass = [0 for i in xrange(NN)]
examLabel = [0 for i in xrange(NN)]

for k in xrange(K):
	core = np.power(2*math.pi,MM/2)*np.sqrt(np.linalg.det(variance[k,:,:]))
	comvariance[k] = 1/core
	#print comvariance[k,0]
for i in xrange(NN):
	prob = 0.0
	Inclass = 0
	for k in xrange(K):
		temp = testData[i,:]-centroids[k,:]
		temp.shape = (1,M)
		#print temp
		#print np.linalg.inv(variance[k,:,:]).shape
		a = (-1/2)*np.dot(np.dot(temp,np.linalg.inv(variance[k,:,:])),np.transpose(temp))

		b = np.exp(a)
		p = gaussindex[k]*comvariance[k]*b
		if p > prob:
			prob = p
			Inclass = k;
	examLabel[i] = label[Inclass]

	#computer the precious
rightoucnt = 0
np.savetxt("/home/ydliu/文档/algorithm/GMM/result.csv",examLabel)
for j in xrange(NN):
	if examLabel[j] == testDataLabel[j]:
		rightoucnt = rightoucnt + 1

precious = float(rightoucnt)/NN
print "the precious is:", precious
endtime = datetime.datetime.now()
print "the total time is:" ,endtime - begintime