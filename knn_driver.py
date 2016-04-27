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

	if method == "Relief":
		return run_knn_relief(splits, k)


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
	trainX = numpy.concatenate((trainX, validX), axis=0)
	trainY = numpy.concatenate((trainY, validY), axis =0)


	train_U, train_S, train_Vt = numpy.linalg.svd(trainX, full_matrices=False)
	test_U, test_S, test_Vt = numpy.linalg.svd(numpy.concatenate((trainX, testX),axis=0), full_matrices=False)
	dimensions = find_svd_dimensions((train_U, train_S, train_Vt))
	print(dimensions)
	print(train_U.shape, train_S.shape, train_Vt.shape)
	
	train_recon = numpy.dot(numpy.dot(train_U[:, :dimensions], numpy.diag(train_S[:dimensions])), train_Vt[:dimensions, :])

	test_recon = numpy.dot(numpy.dot(test_U[:, :dimensions], numpy.diag(test_S[:dimensions])), test_Vt[:dimensions, :])

	test_recon =  test_recon[len(trainX):]


	if "--showplot" in sys.argv:
		for eg in range(0, len(trainX)):
			plt.text(train_U[eg,0], train_U[eg,1], trainY[eg])

		plt.show()


	accuracy = test_knn_SVD(train_recon, trainY, test_recon, testY, k)
	print("Final Acc  = ", accuracy)
	return accuracy

def test_knn_SVD(trainX_svd, trainY, testX_svd, testY, k):

	correct = 0 
	for idx, query in enumerate(testX_svd):
		pred_class = classify_tuple_naive(query, trainX_svd, trainY, k)
		if pred_class == testY[idx]:
			correct +=1

	accuracy = float(correct)/len(testY)
	return accuracy

def find_svd_dimensions(trainX_svd):
	U, S, V_t = trainX_svd
	e_sum = 0 
	red_sum = 0
	for eigen_val in S:
		e_sum += (eigen_val**2)
	for idx, eigen_val in enumerate(S):
		red_sum += (eigen_val**2)
		if red_sum > 0.9 * e_sum:
			return (idx + 1)





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



def run_knn_relief(splits, k):

	(trainX,trainY, validX,validY, testX, testY) = splits
	trainX = numpy.concatenate((trainX, validX))
	trainY = numpy.concatenate((trainY, validY))
	weight_vector = [0] * len(trainX[0])

	print(len(trainX))

	for i in range(0, len(trainX)):
		idx = random.randint(0 , len(trainX)-1)
		near_hit= find_nearest_neighbour(trainX, trainY, idx, True)
		near_miss= find_nearest_neighbour(trainX, trainY, idx, False)

		for feature in range(0, len(trainX[0])):
			hit_diff = trainX[near_hit][feature]
			miss_diff = trainX[near_miss][feature]
			ex = trainX[idx][feature]

			#print(ex, hit_diff, miss_diff)

			weight_vector[feature] = weight_vector[feature] - (ex - hit_diff)**2  + (ex -  miss_diff)**2

	weight_vector = [w/len(trainX) for w in weight_vector]


	correct = 0
	print(trainX[0])
	trainX = trainX * numpy.array(weight_vector)
	print(trainX[0])
	for idx,query in enumerate(testX):
		pred = classify_tuple_naive(query*weight_vector, trainX,trainY, k)
		
		if testY[idx] == pred:
			correct+=1

	print(float(correct)/len(testX))




def find_nearest_neighbour(trainX, trainY, choice_idx,  hit_or_miss):
	dist = {}
	for i,example in enumerate(trainX):
		if  i!=choice_idx and ((trainY[choice_idx] ==trainY[i])==hit_or_miss):
			dist[i] = numpy.linalg.norm(trainX[i] - trainX[choice_idx])
	return min(dist.iteritems(), key=operator.itemgetter(1))[0]








