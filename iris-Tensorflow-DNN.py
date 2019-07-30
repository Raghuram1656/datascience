# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:05:20 2019

@author: SusarlaS
"""

# 
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# LOAD DATA
input_file = 'C:/Users/susarlas/Desktop/data science/projects/classification/data files/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(input_file,names=names)
dataset.info()



# seperate target column
X = dataset.drop(['class'],axis=1)   # axis = 1 means dropping a column.
Y = dataset['class']

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)

#Split data into Training and Testing sets
from sklearn.model_selection import train_test_split
test_size = 0.20 
seed = 10
X_train, X_test, Y_train, Y_test = train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)


# SAMPLING METHOD TRICK
from sklearn.model_selection import KFold 
kfold = KFold(n_splits=10, random_state=seed)
score = "accuracy"



#########################################   HIGH LEVEL TENSORFLOW API ###################################

## High Level TENSORFLOW API
import tensorflow as tf

config = tf.contrib.learn.RunConfig(tf_random_seed=42) # not shown in the config

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,400], n_classes=10,
                                         feature_columns=feature_cols, config=config)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf) # if TensorFlow >= 1.1  A WRAPPER AROUND TENSORFLOW FOR SKLEARN

dnn_clf.fit(X_train, Y_train, batch_size=50, steps=40000)





#Training Accuracy
from sklearn.model_selection import cross_val_score
training_results = cross_val_score(dnn_clf, X_train, Y_train, cv=kfold)
training_results.mean()




# Predict

from sklearn.metrics import accuracy_score
Y_PRED= dnn_clf.predict(X_test)


predictions_score = dnn_clf.score(X_test,Y_test)
print(predictions_score)




#Accuracy Score
accuracy_score(Y_test, Y_PRED['classes'])

## 96.6 % Accuracy


###
predict_sample = dnn_clf.predict([[7.2, 3.5, 0.8, 1.6]])