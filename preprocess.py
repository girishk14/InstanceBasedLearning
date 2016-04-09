def_author__ = 'Girish'
import json
import simplejson
import numpy
import sys
#This module contains routines to pre_process the 5 selected datasets, and obtain the metadata, features and lebels in the form that the decsion tree expects

'''
Format Description:


1. Dataset:  An array of tuples, where each tuple represents the feature vectors, i.e the n features of the examples
2. Labels: The classes corresponding to the examples in the dataset
3. Metadata ;This contains information about the data, i.e types, domains and statistical features of attributes
4. Missing Data: Replace missing data with the value having the highest probability, or the average of all values

'''
def mean(lst):
	sum = 0
#        print(lst)
	for val in lst:
		#print(val)
		if val.strip()!='?':
			sum = sum + float(val)
	return (sum/float(len(lst)))	

def most_common(lst):
    return max(set(lst), key=lst.count)

def pre_process_stage1(control_file): #Read in data, separate into data and labels, and fill in missing values
	dataset = []
	#The control file is to instruct the pre_processer on how to view the data
	labels = [] 
	with open(control_file) as data_file:    

		   metadata  = json.load(data_file)
	sep = metadata['sep'] if 'sep' in metadata.keys() else ','
	for f in  metadata['location']:
		with open(f, 'r') as ifile:
			for line in ifile:
				attrs = line.strip().split(sep)
				dataset.append([attr for i, attr in enumerate(attrs) if i!=metadata['class_position']])
				labels.append(attrs[metadata['class_position']])	
					
	#To compute means, and replace missing data with thse values		
	metadata['attr_mean'] = []
	for i, atype in enumerate(metadata['attr_types']):
		if atype=='c':
			metadata['attr_mean'].append(mean([instance[i] for instance in dataset]))
  
		else:
			metadata['attr_mean'].append(most_common([instance[i] for instance in dataset]))
	#print(metadata['attr_mean'])
	for example in dataset:
		for attr in range(0, len(metadata['attr_types'])):
		 	if example[attr].strip() == '?':
				example[attr] = metadata['attr_mean'][attr]	
				
			if metadata['attr_types'][attr] == 'c':
				example[attr] = float(example[attr])

	
	metadata['no_attrs']   = len(metadata['attr_types'])
	
	print("stage 1 complete",control_file )

	return dataset, labels, metadata


def normalize(dataset, labels, metadata, hot_encode=True):
	
	neural_input = [[] for x in range(len(dataset))]
	for attr in range(0,metadata['no_attrs']):
		if metadata['attr_types'][attr] ==  'c':
			attr_values = [tup[attr] for tup in dataset]
			normalized_column =  (numpy.array(attr_values)  - numpy.mean(attr_values))/numpy.std(attr_values)

			for i,tup in enumerate(neural_input):
				tup.append(normalized_column[i])

		else:
			attr_domain =  list(set([tup[attr] for tup in dataset]))
			for i,tup in enumerate(dataset):

				if hot_encode == True:
					hot_encoded = hot_encode_func(attr_domain, tup[attr])

				else:
					hot_encoded = tup[attr]
				neural_input[i].extend(hot_encoded)

	if hot_encode==True:
		neural_output = [[] for x in range(len(labels))]
		class_domain = list(set(labels))
		for i, label in enumerate(labels):
			hot_encoded = hot_encode_func(class_domain, label)
			neural_output[i].extend(hot_encoded)

	else:
		neural_output = labels

	return (numpy.array(neural_input), numpy.array(neural_output))

def hot_encode_func(domain, val):
	hot_encoded = [1 if val  == cat else 0 for cat in domain]
	return hot_encoded



	#In a neural network, continuous and discrete don't matter anymore post normalization. So lets no worry about losing that information


	
