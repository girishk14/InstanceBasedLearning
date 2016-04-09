import neural_network
import sys
import os
import numpy
import math
import random



def run_neural_network(splits, neural_spec, learning_rate, momentum_rate):
	neural_net =  neural_network.NeuralNetwork(neural_spec, learning_rate, momentum_rate)
	(trainX,trainY, validX,validY, testX, testY) = splits
	iterations ,mse= train_neural(neural_net, trainX, trainY, validX, validY)
	acc = test_neural(neural_net, testX, testY)
	return acc, iterations, mse


def train_neural(neural_net, trainX, trainY, validX, validY):
	epoch_size = len(validX)

	for iteration in range(0, 20000):
		random_idx = numpy.random.randint(0, len(trainX))
		neural_net.feedForward(trainX[random_idx])
		neural_net.backPropagate(trainY[random_idx])

		if iteration%epoch_size == 0:
			mse = getMSE(neural_net, validX, validY)
			print("MSE on Epoch %d : %f"%(iteration/epoch_size, mse))
			if mse <= 0.0255:
				break
	print("Terminating after %d  iterations "%(iteration+1))
	return iteration+1, getMSE(neural_net, validX, validY)


def getMSE(neural_net, validX, validY):

	MSE = 0
	for idx in range(0, len(validX)):
		neural_net.feedForward(validX[idx])
		error = validY[idx] - neural_net.activations[len(neural_net.layers)-1][0:-1]
		MSE  = MSE + 0.5*sum(error * error)
	return MSE/float(len(validX))


def test_neural(neural_net, testX, testY):
	correct = 0
	for instanceX, instanceY in zip(testX, testY):
		targetY =  list(instanceY).index(1)
		predY = neural_classify_tuple(neural_net,instanceX)
		#print(predY, targetY)
		if predY==targetY: correct+=1

	return correct/float(len(testX))


def neural_classify_tuple(neural_net, X):
	neural_net.feedForward(X)
	op = neural_net.activations[len(neural_net.layers) -1]
	#print(op)
	return numpy.argmax(op[0:-1])
