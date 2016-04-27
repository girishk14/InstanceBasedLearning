import preprocess
import sys
import os
import numpy
import math
import statistics
import random
import knn_driver

sys.path.insert(0, 'decision_tree_comp/')
sys.path.insert(0, 'neural_network_comp/')

import dtree_driver
import neural_driver


def shuffle_order(a, b):	
	c = list(zip(a, b))
	random.shuffle(c)
	a, b = zip(*c)
	return a,b

def main():
	
	ip, op , metadata = preprocess.pre_process_stage1(sys.argv[1])
	ip,op = shuffle_order(ip, op)
	normalized_ip, normalized_op = preprocess.normalize(ip, op, metadata);

	knn_ip, knn_op = preprocess.normalize(ip, op, metadata, hot_encode = True)
	#neural_spec = [int(spec.strip()) for spec in sys.argv[2].split(",")]
	neural_spec = [4,5]
	neural_spec.append(len(normalized_op[0]))
	neural_spec.insert(0, len(normalized_ip[0]))
	learning_rate, momentum_rate = 0.10, 0.02# 0.001, 0.001

	knn_accs, comp_accs, mean_iter, mse  =  [0]*10, [0]* 10, 0,0

		
	if '--neural' in sys.argv:
		comp_accs = k_fold_validation_neural_net(normalized_ip, normalized_op, neural_spec, learning_rate, momentum_rate)
	


	elif '--dtree' in sys.argv:
		comp_accs = k_fold_validation_dtree(ip, op, metadata)


	k = int(sys.argv[2])

	knn_accs = k_fold_validation_knn(knn_ip, numpy.array(op), k, metadata)

	print(knn_accs)
	

	print("\n\n")
	print("Dataset Size : %d"%(len(ip)))
	print("Number of features : %d"%len(ip[0]))



	print("\nFold\t\t\tkNN\t\t\tDecision Tree/Neural Network")
	for fold in range(0,10):
			print( "%d \t\t\t %.2f \t\t\t %.2f"%(fold+1, knn_accs[fold], comp_accs[fold]))

	comp_mu, comp_ci = statistics.calc_confidence_interval(comp_accs)
	knn_mu, knn_ci = statistics.calc_confidence_interval(knn_accs)

	t_mu, t_ci = statistics.paired_t_test(comp_accs, knn_accs)

	print("\nConfidence interval for kNN classifier : %.3f   +/-   %.3f"%(knn_mu, knn_ci))
	print("Confidence interval for decison tree/neural network : %.3f   +/-   %.3f"%(comp_mu, comp_ci))


	print("Result of Paired T-Test : %.3f   +/-   %.3f"%(t_mu, t_ci))

	if 0 > t_mu - t_ci and 0<t_mu+t_ci:
		print("The two algorithms are statistically similar")

	else:
		print("The difference in the performance of the two algorithms is statistically significant")




def k_fold_generator(X, y, k_fold):
    subset_size = len(X) / k_fold  # Cast to int if using Python 3
    for k in range(k_fold):
        X_train = X[:(k * subset_size)] + X[(k + 1) * subset_size:]
        X_valid = X[(k * subset_size):][:subset_size]
        y_train = y[:(k * subset_size)] + y[(k + 1) * subset_size:]
        y_valid = y[(k * subset_size):][:subset_size]

        yield X_train, y_train, X_valid, y_valid

def k_fold_validation_neural_net(neural_ip, neural_op, neural_spec, learning_rate, momentum_rate):
	accs = []
	total_iterations = 0
	total_mse = 0
	print("Evaluating neural net")
	for trainX, trainY,testX, testY in k_fold_generator(neural_ip.tolist(), neural_op.tolist(), 10):
		print("Fold " + str(len(accs)+1))
		t_size = int((7/9.0) * len(trainX))
		splits = (numpy.array(trainX[0:t_size]), numpy.array(trainY[0:t_size]), numpy.array(trainX[t_size:]), numpy.array(trainY[t_size:]), numpy.array(testX), numpy.array(testY))
		acc,iters, mse = neural_driver.run_neural_network(splits, neural_spec, learning_rate, momentum_rate)
		accs.append(acc)
		total_iterations+=iters
		total_mse +=mse
	print("Accs", accs)
	return accs


def k_fold_validation_dtree(ip, op, metadata):
	print("Evaluating decision tree . . . ")
	accs = []
	for trainX, trainY,testX, testY in k_fold_generator(ip, op, 10):
		print("Fold %d"%(len(accs)+1))
		t_size = int((7/9.0) * len(trainX))
		splits = (trainX[0:t_size], trainY[0:t_size], trainX[t_size:], trainY[t_size:], testX, testY)
		accs.append(dtree_driver.run_decision_tree(metadata, splits))
	return accs


def downsample(trainX, trainY):
	
	sample_size = math.sqrt(len(trainX))*2#int(len(trainX))*0.2

	a = numpy.random.choice(numpy.arange(0, len(trainX)), size = sample_size)
	T_x = numpy.zeros((sample_size, len(trainX[0])))
	T_y = []
	i = 0

	for idx in a:
		T_x[i]= trainX[idx]
		T_y.append(trainY[idx])
		i+=1

	print(len(T_x), len(T_y))
	return T_x, T_y




def k_fold_validation_knn(knn_ip, knn_op, k, metadata):
	print("Evaluating KNN . . .")
	accs = []
	for trainX, trainY,testX, testY in k_fold_generator(knn_ip.tolist(), knn_op.tolist(), 10):
		print("Fold " + str(len(accs)+1))

		if len(trainX[0]) > 100 and len(trainX)>500: #more than 25 features
			print("Too much data, downsampling")
			trainX, trainY = downsample(trainX, trainY)

		t_size = int((7/9.0) * len(trainX))
		splits = (numpy.array(trainX[0:t_size]), numpy.array(trainY[0:t_size]), numpy.array(trainX[t_size:]), numpy.array(trainY[t_size:]), numpy.array(testX), numpy.array(testY))
		acc  =  knn_driver.run_knn(splits,k, sys.argv[3])
		accs.append(acc)
	return accs


if __name__ == '__main__':
	main()
