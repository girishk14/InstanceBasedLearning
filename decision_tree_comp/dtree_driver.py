import sys
import preprocess
import classification
import decision_tree
import pruning
import random
import decimal

global metadata


def run_decision_tree(metadata, splits):
    (trainX,trainY, validX,validY, testX, testY) = splits
    decision_tree.set_metadata(metadata)
    classification.set_metadata(metadata)
    d_tree = decision_tree.create_decision_tree(trainX,trainY)
    pruned_tree = pruning.reduced_error_pruning(d_tree, trainX, trainY, validX, validY) 
    return classification.get_classification_accuracy(pruned_tree, trainX, trainY, testX, testY)
  
 	

    
