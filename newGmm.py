#!/usr/bin/env python
# coding=utf-8

# from numpy import *
import time
import random
import scipy.io as sio
import numpy as np
import sys


#import the dataset
def loadDataSet(filename):
	localdata = []
        #import the data in .mat
	dic = sio.loadmat(filename)
	for key in dic:
		if type(dic[key]) == np.ndarray:
			# localdata = dic[key]
			localdata.append(dic[key]) 
	#print localdata

        #the format is [1,N,M], so should perform locatedata[0]
	dataSet = localdata[0]

	#print type(temp)
	return dataSet
#the initial centroids
def creatcentroids(dataSet,M,N,K):
	centroids = np.zeros((K,M))
	init = [13,7,0,6,1,26,16,9,10,12]
	for k in xrange(K):
		index = init[k]
               # index = random.randint(0,N)
                centroids[k,:] = dataSet[index,:]
	#print centroids
	return centroids

def getDistance(vector1,vector2):
	#print vector1,vector2
	#print "the sum is ", sum(np.power(vector2 - vector1,2))
	return np.sqrt(sum(np.power(vector2 - vector1,2)))

def kmeans(dataSet,K,N,M):
	#[N,M] = np.shape(dataSet)
	#print N,M
        clusterAssment = [0 for i in xrange(N)]
	#clusterAssment = np.zeros((N,1))
	# init centroids
	centroids = creatcentroids(dataSet,M,N, K)
	#print centroids

	clusterChaned = True

	while clusterChaned:
	 	clusterChaned = False
	 	computercentriods = np.zeros((K,M))
                computerindex = [0 for i in xrange(K)]
		#computerindex = np.zeros((K,1))
	 	#for each samples
	 	for i in xrange(N):
	 		mindistance = sys.maxint
	 		minindex = 0
	 		for k in xrange(K):
	 			distance = getDistance(dataSet[i,:],centroids[k,:])
	 			#print distance
	 			if distance < mindistance:
	 				mindistance = distance
	 				minindex = k
	 		#updata the cluster
	 		if clusterAssment[i] != minindex:
	 			clusterChaned = True
	 			clusterAssment[i] = minindex
	 			#print clusterAssment
	 	#update centroids
               # computerindex = [computerindex[clusterAssment[j]]+1 for j in xrange(N)]
	 	for j in xrange(N):
	 		currentindex = clusterAssment[j]
	 		#print "currentindex = ", currentindex
	 		computerindex[currentindex] = computerindex[currentindex] + 1
	 		computercentriods[currentindex,:] = computercentriods[currentindex,:] + dataSet[j,:]
	 	for k in xrange(K):
	 		#print computerindex[k,0]
	 		centroids[k,:] = computercentriods[k,:]/computerindex[k]
	 	#for k in range(K):
	 		# pointsIncluster = dataSet[nonezero(clusterAssment[:,0].A == k)[0]]
	 		# centroids[k,:] = mean(pointsIncluster,axis)
	#print centroids
	# for i in xrange(N):
	# 	print "data",i,"in cluster",clusterAssment[i,0]
	for k in xrange(K):
		print computerindex[k]
	return centroids, clusterAssment, computerindex

#make sure which label has highest frequency in each centiorid
def whichLabel(clusterAssment,trainlabel,K):
	returnlabel = [0 for i in xrange(K)]
	for k in xrange(K):
		#labelNum = np.zeros((K,1))
                labelNum = [0 for i in xrange(K)]
                listArray = [index for index in xrange(len(clusterAssment)) if clusterAssment[index] == k]
		#classArray = np.array(np.where(clusterAssment[:,0] == k))[0]
		#print type(classArray)
		for i in xrange(len(listArray)):
			index = trainlabel[listArray[i]]
			labelNum[index] = labelNum[index] + 1;
		label = np.argmax(np.array(labelNum))
		returnlabel[k] = label

	return returnlabel


#genetate the gauss index
def getGaussIndex(dataSet,centroids,clusterAssment,computerindex,N,M,K):
	gaussindex = [0 for i in xrange(K)]
	variance = np.zeros((K,M,M))
	for i in xrange(N):
		index = int(clusterAssment[i])
		#print index
		gaussindex[index] = gaussindex[index] + 1
		temp = dataSet[i,:]-centroids[index,:]
		temp.shape = (1,M)
		#print temp
		variance[index,:,:] = variance[index,:,:] + np.transpose(temp)*(temp)
        gaussindex = [float(i)/N for i in gaussindex]
	for j in xrange(K):
		variance[j,:,:] = variance[j,:,:]/computerindex[j]
	return gaussindex, variance





