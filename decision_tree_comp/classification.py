#This module contains routines for making classifications on the decision tree
import sys
import os
import math
import random
import pruning
import decision_tree
global metadata
def set_metadata(md):
	global metadata
	metadata = md

def majority_classifier_accuracy(root, test_examples):
	count = 0
	for y in test_examples:
		if y==root.majority_label:
			count = count + 1
	return (count/float(len(test_examples)))

def get_classification_accuracy(tree,  seen_dataset, seen_labels, unseen_dataset, unseen_labels):
	count = 0
	for i in range(0, len(unseen_dataset)):
        	if classify_tuple(tree, seen_dataset, seen_labels, unseen_dataset[i]) == unseen_labels[i]:
	        	count+=1
	return (count/float(len(unseen_dataset)))

def classify_tuple(root, dataset, labels, test_tuple): #Given the root of a decsion tree, and a new tuple, return its class
    trav = root
    while(trav.isLeaf is not True):
		curr_attr = trav.criteria['next_split_attr'] 
		if metadata['attr_types'][curr_attr] == 'c':
		    if test_tuple[curr_attr] < trav.children[0].criteria['parent_split_point']:
			trav = trav.children[0]
		    else:
		        trav = trav.children[1]
		else:
 		    flag = 0
		    for child in trav.children:
			if child.criteria['parent_split_point'] == test_tuple[curr_attr]:
			    trav = child
                            flag = 1
                    if flag==0: #If there is no way to classify this tuple in the tree
	                guess =  trav.majority_label
			return guess
    return trav.class_label


