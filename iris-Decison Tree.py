# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:06:12 2019

@author: SusarlaS


## -*- coding: utf-8 -*-

@author: SusarlaS
@iris classification problem  -- using Decision Trees (CART)  Supervised and Non Linear Algorithm
@ entropy to calculate the homogeneity of a sample. If the sample is completely homogeneous the entropy is zero and if the sample is an equally divided it has entropy of one.
# it follows a greedy approach.


I have used 2 approaches 
1. Plain Decision Tree Classifier using Entropy
2. Used Ensemble: Bagged Decision Tree

"""

# Import libraries
import pandas as pd

# LOAD DATA
input_file = 'C:/Users/susarlas/Desktop/data science/projects/classification/data files/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(input_file,names=names)
dataset.info()

# seperate target column
X = dataset.drop(['class'],axis=1)   # axis = 1 means dropping a column.
Y = dataset['class']


#Feature Importance
from sklearn.ensemble import ExtraTreesClassifier 
feature_model = ExtraTreesClassifier(n_estimators=100) 
feature_model.fit(X, Y) 
print(feature_model.feature_importances_)


#Split data into Training and Testing sets
from sklearn.model_selection import train_test_split
test_size = 0.20 
seed = 10
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# SAMPLING METHOD TRICK
from sklearn.model_selection import KFold 
kfold = KFold(n_splits=10, random_state=seed)
score = "accuracy"

# Model
from sklearn.tree import DecisionTreeClassifier 

model_DecTree = DecisionTreeClassifier(criterion='entropy',random_state = seed,min_samples_split=10)

#Train and check Tranining Quality using Cross_val_score
from sklearn.model_selection import cross_val_score
training_results = cross_val_score(model_DecTree, X_train, Y_train, cv=kfold)
training_results.mean()
# 92.50 is Training Accuracy.



# Fit the Model
model_DecTree.fit(X_train,Y_train)



#sample predict  1
predict_sample = model_DecTree.predict([[7.2, 3.5, 0.8, 1.6]])
# As per the Decision Tree the predicted output is Iris-versicolor

#7.1 3 5.9 2.1  belongs to IRIS-VIRGINICA lets test whether it is correctly classifying or not
#sample predict  2
predict_sample = model_DecTree.predict([[7.1, 3, 5.9, 2.1]])
# output is VIRGINICA . Correctly classified.

# Now Predict all Test of X_TEST  and cross validate with Y_TEST
predictions = model_DecTree.predict(X_test) 

predictions_score = model_DecTree.score(X_test,Y_test)
print(predictions_score)


# Checking Predictions Accuracy using Cross Validation Report
# method 1 : Using confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(Y_test, predictions)
print(matrix)

#method 2 : Using Classification Report
from sklearn.metrics import classification_report
report = classification_report(Y_test, predictions) 
print(report)
# 97% accuracy is achieved from classification report


#method 3: accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)
# 96.7% from accuracy score


# Here i have tuned Hyper Parameters after careful multiple trails. I learnt best to use entropy than gini

# TRAINING ACCURACY :   92.50 %  ()
# PREDICTION ACCURACY : 96.7% 





###################################################################################################################################

'''
using      Ensemble Bagged Decision Tree 

'''

#######################################################################################################################################



# Using the Ensemble Bagging
from sklearn.ensemble import BaggingClassifier 
cart = DecisionTreeClassifier()
num_trees = 100
model_Bagging_DecisionTree = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed) 



#Train and check Tranining Quality using Cross_val_score
from sklearn.model_selection import cross_val_score
training_results_bag = cross_val_score(model_Bagging_DecisionTree, X_train, Y_train, cv=kfold)
training_results_bag.mean()

# 93.3 is Training Accuracy.



# Fit the Model
model_Bagging_DecisionTree.fit(X_train,Y_train)


#sample predict  1
predict_sample3 = model_Bagging_DecisionTree.predict([[7.2, 3.5, 0.8, 1.6]])
print(predict_sample3)
# As per the Bagged Decision Tree the predicted output is Iris-setosa which is actually correct

#7.1 3 5.9 2.1  belongs to IRIS-VIRGINICA lets test whether it is correctly classifying or not
#sample predict  2
predict_sample4 = model_Bagging_DecisionTree.predict([[7.1, 3, 5.9, 2.1]])
print(predict_sample4)
# output is VIRGINICA . Correctly classified.

# Now Predict all Test of X_TEST  and cross validate with Y_TEST
predictions1 = model_Bagging_DecisionTree.predict(X_test) 

predictions_score1 = model_Bagging_DecisionTree.score(X_test,Y_test)
print(predictions_score1)

# Got 100% predictions_score


# Checking Predictions Accuracy using Cross Validation Report
# method 1 : Using confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(Y_test, predictions1)
print(matrix)

#method 2 : Using Classification Report
from sklearn.metrics import classification_report
report = classification_report(Y_test, predictions1) 
print(report)
# 100% accuracy is achieved from classification report


#method 3: accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions1)
# 100% from accuracy score




'''
Hence using Bagged Decision Trees we have improved the performance
'''






#Saving our Model using Pickel
from pickle import dump 
from pickle import load 

  
 # save the model to disk 
filename = 'iris_Bagged_DecisionTree.sav' 
dump(model_DecTree, open(filename, 'wb'))

 
# load the model from disk 
loaded_model = load(open(filename, 'rb')) 
result = loaded_model.score(X_test, Y_test) 
print(result)



# Gini is used for Continuous variables 
# Entropy is used for classification (it takes some time because it uses a log function)









