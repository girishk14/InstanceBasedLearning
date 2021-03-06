CSCE 633 : Machine Learning (Spring 2016)
Project 3 : K-nearest Neighbours and Feature Selection	
Author : Girish Kasiviswanathan (UIN : 425000392)


Installation
------------
This code has been written using Python 2.7 on Ubuntu, using VI Editor. 
This implementation makes use of the NumPy library for Python, which can be installed using the command 'pip install --user numpy'
 

Dependencies
-------------
Numpy
Scipy
simplejson

Files and Directories Included
------------------------------
1. data : Folder containing all the datasets and their control files
2. decision_tree_comp: Folder containing modules from decision_tree project that are integrated in this project
3. neural_network_comp: Folder containing modules from neural_network projct for evaluation purposes
3. driver.py, pre_process.py, knn_driver.py,  statistics.py: Core Python modules
4. KNearestNeighbours_Report.pdf : Design documentation
5. Results_KNN.pdf : Results for sample runs 


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

"sep" : ',', 
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



Running the kNN classifier:
---------------------------
To execute the kNN classifier:

python driver.py arg1 arg2 


arg1 : path to control file (string)
arg2 : k  (number of neighbours to use)
arg3: Flavour of kN

The accepted value for arg2 is any positive integer lesser than the size of the training set

The accepted values for arg3 are :   Naive, Distance_Weighted, KDTree, Feature_Selection, SVD, and Relief, as explained in the design documentation



For example, for the selected datasets,
python driver.py data/Iris/control.json 5 Distance_Weighted             - Run Distance-Weighted 5-nearest-neighbours algorithm on Iris
python driver.py data/Arry/control.json 3 SVD                		- Run 3-NN with SVD on Arrhythmia dataset
python driver.py data/Sonar/control.json 1 Relief               	- Run 1-NN with relief feature weighting, on Sonar dataset
python driver.py data/Libra/control.json 73 KDTree               	- Construct KDTree for Libra dataset, and run 73-NN algorithm using it


Running Decision Tree or Neural Network comparision
---------------------------------
Add the switch --dtree to the command for decision tree, or --neural for neural network. Default parameters are used for the neural_network, as documented in Results_KNN.pdf.

Thiss run the corresponding algorithm also on the same folds for kNN and either neural/dtree and reports the accruacies and paired T-test scores. If this switch is not enabled, the dtree/neural accuracies default to 0.

For example,
python driver.py data/Arcene/control.json 5 Naive  --dtree         - Naive 5-NN V.S Decision Tree on Arcene

python driver.py data/Image/control.json 3 SVD  --neural   	   - 3-NN using SVD  V.S Neural Network  on Image


Visualization for SVD
---------------------
To see the two dimensional representation of data, you need to install matplotlib, and add the switch --showplot to the running command.

Output
-------
The program reports the accuracies on each fold for the two algorithms, and finally the confidence interval, and score obtained from the paired-T test.

Sample Output
-------------
girishk14@ubuntu:~/ML/InstanceBasedLearning$ python driver.py "data/BreastCancer/control.json" 5 Naive --dtree
('stage 1 complete', 'data/BreastCancer/control.json')
Evaluating decision tree . . . 
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
Fold 6
Fold 7
Fold 8
Fold 9
Fold 10
Evaluating KNN . . .
Fold 1
Running Naive kNN
Fold 2
Running Naive kNN
Fold 3
Running Naive kNN
Fold 4
Running Naive kNN
Fold 5
Running Naive kNN
Fold 6
Running Naive kNN
Fold 7
Running Naive kNN
Fold 8
Running Naive kNN
Fold 9
Running Naive kNN
Fold 10
Running Naive kNN



Time Taken : 1.354892
Dataset Size : 699
Number of features : 9

Fold			kNN			Decision Tree/Neural Network
1 			 0.97 			 0.97
2 			 0.97 			 0.99
3 			 0.97 			 0.93
4 			 0.94 			 0.91
5 			 0.97 			 0.96
6 			 0.99 			 0.97
7 			 0.93 			 0.91
8 			 0.94 			 0.88
9 			 0.96 			 0.94
10 			 0.97 			 0.87

Confidence interval for kNN classifier : 0.961   +/-   0.013
Confidence interval for decison tree/neural network : 0.933   +/-   0.027
Result of Paired T-Test : -0.028   +/-   0.023
The difference in the performance of the two algorithms is statistically significant
