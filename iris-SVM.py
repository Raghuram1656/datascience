# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:42:22 2019

@author: SusarlaS
@iris classification problem 
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
from sklearn.svm import SVC
model_svc = SVC(gamma='auto')

#Train and check Tranining Quality using Cross_val_score
from sklearn.model_selection import cross_val_score
training_results = cross_val_score(model_svc, X_train, Y_train, cv=kfold)
training_results.mean()
# 97.49 is Training Accuracy.



# Fit the Model
model_svc.fit(X_train,Y_train)



#sample predict  1
predict_sample = model_svc.predict([[7.2, 3.5, 0.8, 1.6]])
# As per the SVM the predicted output is Iris-setosa 

#7.1 3 5.9 2.1  belongs to IRIS-VIRGINICA lets test whether it is correctly classifying or not
#sample predict  2
predict_sample = model_svc.predict([[7.1, 3, 5.9, 2.1]])
# output is VIRGINICA . Correctly classified.

# Now Predict all Test of X_TEST  and cross validate with Y_TEST
predictions = model_svc.predict(X_test) 

predictions_score = model_svc.score(X_test,Y_test)
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

# 100% Prediction Accuracy is achieved from classification_report

#method 3: accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)
# 96.7% from accuracy score


# TRAINING ACCURACY :   97.49 %
# PREDICTION ACCURACY : 100% 


#Saving our Model using Pickel
from pickle import dump 
from pickle import load 

  
 # save the model to disk 
filename = 'iris_finalized_model_SVM.sav' 
dump(model_svc, open(filename, 'wb'))

 
# load the model from disk 
loaded_model = load(open(filename, 'rb')) 
result = loaded_model.score(X_test, Y_test) 
print(result)













