from __future__ import print_function
import os
import sys
from copy import deepcopy
import math
import json
import random

try:
    import pydot
except ImportError:
    pass

global metadata
def set_metadata(md):
	global metadata
	metadata   = md

class node:
    def __init__(self, attr, split_point, partition, dataset, labels):
        self.children = []
        self.criteria = {"next_split_attr": attr, "parent_split_point":split_point}
        self.partition = partition
        self.class_label = None
	self.majority_label = get_class_majority(dataset , labels, partition)
	self.isLeaf = False

    def set_class_label(self):
        self.class_label = self.majority_label
	self.isLeaf = True

    def leafify(self, dataset, labels):
	self.set_class_label()
	
    def unleafify(self):
	self.isLeaf = False
	self.class_label = None


    def print_node(self):
        print(self.criteria)
        print(self.children)
        print(self.partition)
        print(self.class_label)
	print(self.isLeaf)
	print("\n")



def find_distribution(partition, labels):
    count = {}
         
    for i in partition:
        if labels[i] not in count.keys():
            count[labels[i]] = 1
        else:
            count[labels[i]]+=1
    dist_str = ''
    for class_label in count.keys():
	dist_str = dist_str + class_label + " : " + str(count[class_label]) + ", "


    return dist_str
	


def print_ASCII_tree(root, labels, depth = 0):
	if root.isLeaf: return
	for idx, child in enumerate(root.children):
		if metadata['attr_types'][root.criteria['next_split_attr']]=='d':
			lb = metadata['attr_names'][root.criteria['next_split_attr']] + ' = ' + str(child.criteria['parent_split_point']);
				
		else:
			lb = (metadata['attr_names'][root.criteria['next_split_attr']] + " < " + str(child.criteria['parent_split_point'])) if idx==0 else (metadata['attr_names'][root.criteria['next_split_attr']] + " > " + str(child.criteria['parent_split_point'])) 	
		class_distribution = find_distribution(child.partition, labels)
        	for i in range(0,depth): 
			print("   ", end="")
		print(lb + "---> " + class_distribution)
		print_ASCII_tree(child, labels, depth+1)
			
		
	

def create_decision_tree(dataset, labels):
     #Root Node has no split point or attribute
    attr_count = len(metadata['attr_types'])
    attr_list = set(range(0, attr_count))
    return generate_subtree(dataset, labels, range(0, len(dataset)),  attr_list )
	
def prune_attributes(dataset, labels, partition, attr_list):
    pruned_attr_list = []
    for attr in attr_list:
       attr_domain = set([dataset[instance][attr] for instance in partition])
       if len(attr_domain)>1:
           pruned_attr_list.append(attr)
    return pruned_attr_list
		

def generate_subtree(dataset, labels, partition, attr_list, parent_split_pt=None,depth = 0):
    #Stop building the subtree of there are no attributes left or if all the members are in the same class
    if len(set([labels[instance] for instance in partition]))==1: #There is only class of examples left
        root =  node(None, parent_split_pt, partition, dataset, labels)
        root.set_class_label()
        return root

    elif len(attr_list)==0: #There are no attributes left
        root = node(None, parent_split_pt, partition, dataset, labels)
        root.set_class_label()
        return root
 
    else: 
        pruned_attr_list = prune_attributes(dataset, labels, partition, attr_list) #Prune away attributes that have only a single value left in their domain

        if len(pruned_attr_list) == 0:
            root = node(None, parent_split_pt, partition, dataset, labels)
            root.set_class_label()  
            return root

        attr_list = pruned_attr_list
        best_attr, cont_split_pt = get_splitting_criteria(dataset,  labels, partition, attr_list)
	
        #Here we distinguish between discrete and conti
        root = node(best_attr, parent_split_pt, partition, dataset, labels)

        if(cont_split_pt==None): #A discrete attribute has been chosen to split!
            attr_domain = set([dataset[instance][best_attr] for instance in partition])	
            attr_list.remove(best_attr)		
    	  
            for x in attr_domain: #Create the new children by partitioning this attribute
                subpart = [instance for instance in partition if dataset[instance][best_attr]==x]
		root.children.append(generate_subtree(dataset, labels, subpart,deepcopy(attr_list), x, depth+1))

        else: #Continuos attribute, so binary branching
            l,r = [], []
            attr_list.remove(best_attr);
            for instance in partition:
                l.append(instance) if dataset[instance][best_attr]<cont_split_pt else r.append(instance)
            root.children.append(generate_subtree(dataset, labels, l,deepcopy(attr_list), cont_split_pt,depth+1))
            root.children.append(generate_subtree(dataset, labels, r,deepcopy(attr_list), cont_split_pt, depth+1))
    return root


def get_class_majority(dataset, labels, partition):
    lst  = [labels[instance] for instance in partition]
    return max(set(lst), key=lst.count)

def get_splitting_criteria(dataset, labels, partition, attr_list):
    S = compute_entropy(dataset, labels, partition);
    best_attr = -1
    best_gain = -sys.maxint
    best_split_pt = None
    for attribute in attr_list:
        split_pt = None
        if metadata['attr_types'][attribute] == 'd':
            S_A, split_pt = find_entropy_discrete(dataset, labels, partition, attribute)
        else:
            S_A, split_pt = find_entropy_continuous(dataset, labels, partition, attribute)

        if (S - S_A) > best_gain:
            best_attr = attribute
            best_gain = (S - S_A)
            best_split_pt = split_pt
	
    return best_attr, best_split_pt

def find_entropy_continuous(dataset, labels, partition, attribute):
    E = 0
    attr_domain =  set([dataset[instance][attribute] for instance in partition])
    split_pt = None
    min_E =  sys.maxint
    sorted_domain =  sorted(attr_domain)
    for x in range(0, len(sorted_domain)-1):
        mid = (sorted_domain[x] + sorted_domain[x+1])/2.0
        l,r = [], []
        for instance in partition:
            l.append(instance) if dataset[instance][attribute]<mid else r.append(instance)
        lweight  = float(len(l))/float(len(partition))
        rweight = float(len(r))/float(len(partition))
        E = lweight  * compute_entropy(dataset, labels, l) + rweight*compute_entropy(dataset, labels, r)
        if(E<min_E):
            min_E = E
            split_pt = mid 

   
    return min_E, split_pt





def find_entropy_discrete(dataset, labels, partition, attribute):
    E = 0
    attr_domain =  set([dataset[instance][attribute] for instance in partition])
    for x in attr_domain: #For each possible attribute value
        subpart = [instance for instance in partition if dataset[instance][attribute]==x]
        #Add the weighted entropy of every subpartiton
        E  = E +  (len(subpart)/float(len(partition))) * compute_entropy(dataset, labels, subpart)

    return E, None


def compute_entropy(dataset, labels, subpart):
    count = {}
         
    for i in subpart:
        if labels[i] not in count.keys():
            count[labels[i]] = 1
        else:
            count[labels[i]]+=1

    e = 0	
  
    for l in count.keys():
        pi = count[l]/float(len(subpart))
	#print(pi)
        e = e - (pi * math.log(pi, 2))
    return e




def visualize_tree(root, filename):
	#exec("import pydot")
	graph = pydot.Dot(graph_type='digraph')


	par = pydot.Node(str(root), label  = metadata['attr_names'][root.criteria['next_split_attr']])
	graph.add_node(par)
	visualize_subtree(root, par, graph)

	graph.write_png(filename)


def visualize_subtree(root, vizpar, graph): 
	
	if root.isLeaf:
		return None

	for idx, child in enumerate(root.children):
		if not child.isLeaf:
			vizchild = pydot.Node(str(child), label = metadata['attr_names'][child.criteria['next_split_attr']])

		else:
			vizchild = pydot.Node(str(child), label = child.class_label,style="filled", fillcolor="green")
		graph.add_node(vizchild)
	 
		if metadata['attr_types'][root.criteria['next_split_attr']]=='d':
			lb = str(child.criteria['parent_split_point']);
			
		else:
			lb = (" < " + str(child.criteria['parent_split_point'])) if idx==0 else (" > " + str(child.criteria['parent_split_point'])) 
			
		graph.add_edge(pydot.Edge(vizpar, vizchild,  label = lb, color="blue"))
		visualize_subtree(child, vizchild, graph)
						


