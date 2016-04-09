CSCE 633 : Machine Learning (Spring 2016)
Project 1 : Neural Networks and Back-propagation

Author : Girish Kasiviswanathan (UIN : 425000392)


Installation
------------
This code has been written using Python 2.7 on Ubuntu, using VI Editor. 
This implementation makes use of the NumPy library for Python, which can be installed using the command 'pip install --user numpy'
 

Files and Directories Included
------------------------------
1. data : Folder containing all the datasets and their control files
2. decision_tree_comp: Folder containing modules from decision_tree project that are integrated in this project
3. driver.py, pre_process.py, neural_network.py, statistics.py: Core Python module
4. Decision_Tree_Report.pdf : Design documentation
5. Results.pdf : Results for sample runs 


Control Files
-------------

The control files for the 6 specified datasets have already been generated. You may need to make a new control file for testing on new datasets. JSON format is used so that we can define additional parsing parameters in future. 


This is the sample control file for the Iris Dataset: 

NOTE: ALL THE FOLLOWING ARE MANDATORY METADATA INFORMATION REQUIRED

{
 "attr_types": [   //The sequence of attributes is assumed to be same as that in the raw input
  "c", 
  "c", 
  "c", 
  "c"
 ], 
 "class_name": "Class",  //Holds the position of the class column in the raw data
 "class_position": 4, 
 "location": [
  "data/Iris/iris.data" //Location of the data. We can specify multiple locations by using a comma separator.
 ], 
 "attr_names": [
  "Sepal Length", 
  "Sepal Width", 
  "Petal Length", 
  "Petal Width"
 ], 
}



Running the Neural Network:
---------------------------
To execute the decision tree on some program, use the following command : 

python driver.py arg1 arg2 

arg1 : path to control file (string)
arg2 : hidden layer specification (comma separated string of numbers, within quotes)


For example, for the selected datasets,
python driver.py data/Iris/control.json "4"                  - Single hidden layer of 4 neurons
python driver.py data/BreastCancer/control.json "3,1,2"      - 3 hidden layers, of sizes 3,1 and 2
python driver.py data/Mushroom/control.json "0"              - 1 hidden layer of size 0 (empty)
python driver.py data/Pima/control.json  "100,100"           - 2 hiddens layers of 100 neurons each
python driver.py data/Phising/control.json "3"
python driver.py data/Car/control.json "4"


Running Decision Tree comparision
---------------------------------
Add the switch --dtree to the command. This run the decision tree also on the same folds and reports the accruacies. If this switch is not enabled, the decision tree accuracies default to 0.

For example,
python driver.py data/Iris/control.json "4" --dtree

Output
-------
The program reports the accuracies on each fold for the two algorithms, and finally the confidence interval, and score obtained from the paired-T test.

Sample Output
-------------
girishk14@ubuntu:~/ML/neural_network$ python driver.py data/Car/control.json "4" --dtree


Dataset Size : 1728
Average Number of Iterations for Neural Network : 19999.000
Average of Mean Squared Error for Neural Network : 0.07

Fold			Neural Network			Decision Tree
1 			 0.85 			 0.94
2 			 0.91 			 0.92
3 			 0.87 			 0.95
4 			 0.92 			 0.91
5 			 0.85 			 0.90
6 			 0.90 			 0.91
7 			 0.88 			 0.93
8 			 0.89 			 0.95
9 			 0.92 			 0.96
10 			 0.88 			 0.93

Confidence interval for neural network : 0.887   +/-   0.017
Confidence interval for decison tree : 0.929   +/-   0.015
Result of Paired T-Test : 0.042   +/-   0.022
The difference in the performance of the two algorithms is statistically significant

