import numpy
import random
from scipy import spatial
import os
import sys
import operator
import heapq
import math
import copy
import matplotlib.pyplot as plt

def run_knn(splits, k, method):
	
	if method == "Naive":
		return run_knn_naive(splits, k)
	
	if method == "SVD":
		return run_knn_SVD(splits, k)
	
	if method == "KDTree":
		return run_knn_KD(splits, k)

	if method == "Distance_Weight":
		return run_knn_distance_weight(splits, k)

	if method == "Feature_Selection":
		return run_knn_feature_selection(splits, k)



def run_knn_naive(splits, k):
	(trainX,trainY, validX,validY, testX, testY) = splits
	correct = 0
	for idx,query in enumerate(testX):
		pred = classify_tuple_naive(query, trainX,trainY, k)
		
		if testY[idx] == pred:
			correct+=1
	return (float(correct)/len(testX))



def classify_tuple_naive(query, trainX, trainY, k):
	dist = {}

	for idx, tup in enumerate(trainX):
		dist[idx]  = numpy.linalg.norm(tup - query)
		if dist[idx] == 0 :  return trainY[idx]

	knns = heapq.nsmallest(k, dist, key=dist.get)

	votes = {}
	for neighbour in knns:
		if trainY[neighbour] not in votes:
			votes[trainY[neighbour]] = 1	
		else:
			votes[trainY[neighbour]]+=1

	#print(votes)
	predicted_class = max(votes.iteritems(), key=operator.itemgetter(1))[0]
	return predicted_class


def run_knn_distance_weight(splits, k):
	(trainX,trainY, validX,validY, testX, testY) = splits
	correct = 0
	for idx,query in enumerate(testX):
		pred = classify_tuple_naive(query, trainX,trainY, k)
		
		if testY[idx] == pred:
			correct+=1
	print(float(correct)/len(testX))



def classify_tuple_distance(query, trainX, trainY, k):
	dist = {}

	for idx, tup in enumerate(trainX):
		dist[idx]  = numpy.linalg.norm(tup - query)
		if dist[idx] == 0 :  return trainY[idx]

	knns = heapq.nsmallest(k, dist, key=dist.get)
	votes = {}
	for neighbour in knns:
		if trainY[neighbour] not in votes:
			votes[trainY[neighbour]] = (1/(dist[neighbour]**2)) * 1	
		else:
			votes[trainY[neighbour]]+= (1/(dist[neighbour]**2)) * 1

	predicted_class = max(votes.iteritems(), key=operator.itemgetter(1))[0]
	return predicted_class





def run_knn_KD(splits, k):
	(trainX,trainY, validX,validY, testX, testY) = splits
	correct = 0
	KD = spatial.KDTree(trainX)
	print("KDTREE Built")
	for idx,query in enumerate(testX):
		pred = classify_tuple_KDTree(KD, query, trainY, k)

		if testY[idx] == pred:
			correct+=1


	print(float(correct)/len(testX))


def classify_tuple_KDTree(KD, query, trainY, k):
	distances,knns = KD.query(query, k=k)
	votes = {}

	for neighbour in knns:
		if trainY[neighbour] not in votes:
			votes[trainY[neighbour]] = 1	
		else:
			votes[trainY[neighbour]]+=1

	predicted_class = max(votes.iteritems(), key=operator.itemgetter(1))[0]

	return predicted_class





def run_knn_SVD(splits, k):
	(trainX,trainY, validX,validY, testX, testY) = splits
	correct = 0

	print(trainX)
	trainX_svd = numpy.linalg.svd(trainX, full_matrices=False)


	validX_svd = numpy.linalg.svd(numpy.concatenate((trainX, validX),axis=0), full_matrices=False)
	
	testX_svd = numpy.linalg.svd(numpy.concatenate((trainX, testX),axis=0), full_matrices=False)


	dimensions = find_svd_dimensions(trainX_svd, trainY, validX_svd, validY, k)


def find_svd_dimensions(trainX, trainY, validX, validY, k):
	u1, s1, v1 = trainX
	u2, s2, v2 = validX[0][len(trainY):], validX[1], validX[2]

	accuracies = {}


	for i in range(2, len(u1[0])+1):
		print("Folding into %d dimensions"%i)

		cs1, cs2 = copy.deepcopy(s1), copy.deepcopy(s2)

		cs1[i:] = 0
		cs2[i:] = 0
		train_recon = (numpy.dot(u1, numpy.diag(cs1)))[:,:i]

		for idx,ex in enumerate(train_recon):
			plt.text(ex[0], ex[1], trainY[idx])

		plt.show()

		valid_reconn = (numpy.dot(u2, numpy.diag(cs2)))[:,:i]
		correct = 0

		for idx,query in enumerate(valid_reconn): 
			pred_class = classify_tuple_naive(query, train_recon, trainY, k)

			if pred_class == validY[idx]:
				correct +=1

		accuracies[i] = float(correct)/len(validY)
		return max()

		
	print(accuracies)
		

	sys.exit()





