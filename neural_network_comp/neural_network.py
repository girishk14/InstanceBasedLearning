import numpy
import os
import sys
import math
from copy import deepcopy
from itertools import izip

def sigmoid(y): return 1 / (1.0 + numpy.exp(-y))
def derivative_sigmoid(y): return y * (1-y)

class NeuralNetwork:
	def __init__(self, layers, learning_rate, momentum_rate):
		self.layers = [layer+1 for layer in layers]
		self.activations = [numpy.ones(layer_size) for layer_size in self.layers]
		self.weights = [numpy.random.randn(self.layers[i], self.layers[i+1]-1) for i in range(0, len(layers)-1)]
		self.delta_w = [numpy.zeros_like(weight_matrix) for weight_matrix in self.weights] #Reinitlize the deltas for this epoch
		self.learning_rate,self.momentum_rate = learning_rate,  momentum_rate

	def feedForward(self, inputs): #Now compute activations at each layer. Input layer activations are the same as the inputs. 
		self.activations[0][0:-1] = inputs 
		for l  in range(1, len(self.layers)):  #compute activations for the rest of the layers:
			self.activations[l][0:-1] = sigmoid(numpy.dot(self.activations[l-1], self.weights[l-1]))

	def compute_errors(self, outputs):
		self.unit_errors = [numpy.zeros(len(act_layer)-1) for act_layer in self.activations]
		op_layer = len(self.layers)-1
		self.unit_errors[op_layer] =  (outputs - self.activations[op_layer][0:-1]) * derivative_sigmoid(self.activations[op_layer][0:-1])
		for hiddenlayer in range(op_layer-1, 0, -1): #From the second last layer, to the the second layer (Hidden Layers) :
			self.unit_errors[hiddenlayer]  = numpy.dot(self.weights[hiddenlayer][0:-1],self.unit_errors[hiddenlayer+1]) * derivative_sigmoid(self.activations[hiddenlayer][0:-1])
	
	def compute_delta_w(self):
		for layerid in range(len(self.weights)-1,-1,-1):
			self.delta_w[layerid] = self.learning_rate * numpy.outer(self.activations[layerid], self.unit_errors[layerid+1]) + self.momentum_rate * self.delta_w[layerid]

	def update_weights(self):
		for l in range(0, len(self.weights)): self.weights[l]  = self.weights[l] + self.delta_w[l]

	def backPropagate(self, outputs):
		mse = self.compute_errors(outputs)
		self.compute_delta_w()
		self.update_weights()
	

#
