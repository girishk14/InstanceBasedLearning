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

	print(float(correct)/len(testX))
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
	return (float(correct)/len(testX))



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
	return (float(correct)/len(testX))


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
	print("Factorizing trainX")
	trainX_svd = numpy.linalg.svd(trainX, full_matrices=False)

	print("Factorizing validX and testX")
	validX_svd = numpy.linalg.svd(numpy.concatenate((trainX, validX),axis=0), full_matrices=False)	
	testX_svd = numpy.linalg.svd(numpy.concatenate((trainX, testX),axis=0), full_matrices=False)
	dimensions = find_svd_dimensions(trainX_svd, trainY, validX_svd, validY, k)
	accuracy = test_knn_SVD(trainX_svd, trainY, testX_svd, testY, k, dimensions)
	print("Final Acc  = ", accuracy)
	return accuracy
	#sys.exit()


def test_knn_SVD(trainX_svd, trainY, testX_svd, testY, k, dim):
	u1, s1, v1 = trainX_svd
	u2 ,s2,v2 = testX_svd[0], testX_svd[1], testX_svd[2]

	print(u2.shape, s2.shape, v2.shape)

	cs1, cs2 = copy.deepcopy(s1), copy.deepcopy(s2)
	correct = 0
	cs1[dim:] , cs2[dim:]  = 0, 0
	train_recon = (numpy.dot(u1, numpy.diag(s1)))[:,:dim]

	test_recon = (numpy.dot(u2, numpy.diag(s2)))[len(trainY):,:dim]

	print(train_recon.shape, test_recon.shape)
	raw_input()

	for idx, query in enumerate(test_recon):
		pred_class = classify_tuple_naive(query, train_recon, trainY, k)
		if pred_class == testY[idx]:
			correct +=1

	accuracy = float(correct)/len(testY)
	return accuracy

def find_svd_dimensions(trainX, trainY, validX, validY, k):
	dims = len(trainX[1])
	accuracies = {}
	for i in range(1, dims+1):
		accuracies[i] = test_knn_SVD(trainX, trainY, validX, validY, k, i)
		print(accuracies)
	return max(accuracies.iteritems(), key=operator.itemgetter(1))[0]






def run_knn_feature_selection(splits, k):
	(trainX,trainY, validX,validY, testX, testY) = splits	

	sel_set = set()
	features_left = set(xrange(0, len(trainX[0])))

	prev_acc = 0

#To identify the best features
	while True:
		acc_on_valid_set = {}
		for x in features_left:
			try_set = sel_set.union(set([x]))

			acc_on_valid_set[x] = classify_on_select_features(trainX, trainY, validX, validY, try_set, k)

		print(acc_on_valid_set)
		best_feature = max(acc_on_valid_set.iteritems(), key=operator.itemgetter(1))[0]
		if acc_on_valid_set[best_feature] > prev_acc:
			sel_set.add(best_feature);
			print(sel_set)
			features_left.remove(best_feature)
			prev_acc = acc_on_valid_set[best_feature]

		else:
			break


	print("Best featuers are :", sel_set)
	acc = classify_on_select_features(trainX, trainY, testX, testY, sel_set, k)
	print(acc)
	return acc

def classify_on_select_features(trainX, trainY, testX, testY, feature_set, k):
	
	trainX = trainX[:, list(feature_set)]
	testX = testX[:, list(feature_set)]
 
	correct = 0
	for idx,query in enumerate(testX):
		pred = classify_tuple_naive(query,trainX,trainY, k)
		
		if testY[idx] == pred:
			correct+=1

	return (float(correct)/len(testX))






