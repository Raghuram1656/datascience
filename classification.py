#importing common python libraries
import numpy  as np
import scipy
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

#load libraries for Models
import sklearn
from sklearn import model_selection

#for Data Preparation and Modeling pipeline to stop data leakage
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#Dimensionality Reduction, Encoding, Splitting
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#Feature Selection
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 

#classification Algorithms libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


#Evaluation Metrics and Methods libraries for classification
#logarithemic loss, Area under ROC curve,classification accuracy and the below
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#Logarithemic losss: to evaluate  the predictions of probabilities of membership to a given class
#Area under ROC : for Binary classification problems
#True Positive rate also called 'Recall' also called sensitivity
#True Negative rate also called Specificity
#for Area under Roc curve, Logarithemic loss and classification accuracy we use KFold and cross_validation also
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score


#Saving Final Model
from pickle import dump 
from pickle import load 

##################################### IRIS DATASET CLASSIFICATION PROBLEM ##########################


# Problem Definition : Predict the class of the flower based on available attributes.

#read  input file
input_file = 'C:/Users/susarlas/Desktop/data science/projects/classification/data files/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(input_file,names=names)
dataset = df


# dimensionality or shape
print(dataset.shape)

#info
print(dataset.info())
# head PEEK
print(dataset.head(20))
#describe
print(dataset.describe())
#types
types = dataset.dtypes
print(types)

##Descriptive Statistics
dataset.describe()

#classification table
class_counts = dataset.groupby('class').size() 

#Fiding out the Missing Values
dataset.isnull().any().sum()


#Univariate Analysis
dataset.plot(kind='box', subplots=False, layout=(2,2), sharex=False, sharey=False) 
plt.show()


#Standardization
#Scaled_dataset  = 

#Non Standardization 
# Split-out validation dataset 
array = dataset.values 
X = array[:,0:4] 
Y = array[:,4] 
validation_size = 0.20 
seed = 7 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


#Multiple Algorithms for Classification 
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto'))) 

# evaluate each model in turn 
results = []
names = []

for name,model in models:
    kfold = KFold(n_splits = 5, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)

# With this it is very clear that KNN is best
#lets go ahead and use KNN and predict 

# Make predictions on validation dataset 
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train) 



knn.score(X_validation,Y_validation)
predictions = knn.predict(X_validation) 




print(accuracy_score(Y_validation, predictions)) 
print(confusion_matrix(Y_validation, predictions)) 
print(classification_report(Y_validation, predictions))



#SVM
#Create a svm Classifier
model2 = SVC(gamma='auto')
model2.fit(X_train,Y_train)
prediction2 = model2.predict(X_validation)
print(classification_report(Y_validation,prediction2))
print(accuracy_score(Y_validation, prediction2)) 

model2.score(X_validation,Y_validation)


# Using KNN, SVM with  K-fold Cross-Validation on training data.
# knn
model_knn = KNeighborsClassifier()
model_knn_fit = model_knn.fit(X_train,Y_train)
knn_score = model_knn.score(X_validation,Y_validation)  #0.90
#k fold for knn
knn_kfold = KFold(n_splits=10, random_state=7) 

#results of training
knn_results = cross_val_score(model_knn_fit, X_train,Y_train, cv=knn_kfold) 
print(knn_results.mean()*100.0)

#predict and see the matching results
knn_prediction = model_knn_fit.predict(X_validation)
print(classification_report(Y_validation,knn_prediction))




#svm 
model_svm= SVC(gamma='auto')
#fit models
model_svm_fit = model_svm.fit(X_train,Y_train)
#get scores
svm_score = model_svm.score(X_validation,Y_validation)  #0.93


#validate now using K-fold Cross-Validations
svm_kfold = KFold(n_splits=25, random_state=8) 
#results of Training
svm_results = cross_val_score(model_svm_fit, X_train,Y_train, cv=svm_kfold) 
print(svm_results.mean()*100.0)

#predict and see the results
svm_prediction = model_svm_fit.predict(X_validation)
print(classification_report(Y_validation,svm_prediction))




################# Still to do

# 1. Features Selection
#Use PCA or ExtraTreeClassifier to know the importance of each attribute and decide
# 2. Dimensionality Reduction using different Methods

# 3. Improving Performance with Ensembles
     #ExtraTreeClassifier is an easy Ensemble which is also used as a Feature Importance tool

# 4. Improving Performance with Algorithm Tuning



# Feature Importance using Extra Tree Classifier

from sklearn.ensemble import ExtraTreesClassifier 
# feature extraction 
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X_train, Y_train) 
print(model.feature_importances_)


#Ensembles

#Bagging  Random Fores



# parameters for Random forest ensemble
from sklearn.ensemble import RandomForestClassifier
num_trees = 100 
max_features = 3 
# parameters for Kfold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
kfold_randomforest = KFold(n_splits=10, random_state=7) 
model_randomforest = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
randomforest_results = cross_val_score(model_randomforest, X_train, Y_train, cv=kfold) 
print(randomforest_results.mean())







#Decision Tree Classifier
num_trees = 16
max_features = 4 
from sklearn import tree
kfold_decisiontree = KFold(n_splits=10, random_state=7) 
model_decisiontree = tree.DecisionTreeClassifier(criterion="gini",max_features=max_features,random_state = 7)
decisiontree_results = cross_val_score(model_decisiontree , X_train, Y_train, cv=kfold) 
print(decisiontree_results.mean())







# Boosting AdaBoost
from sklearn.ensemble import AdaBoostClassifier 
num_trees = 30
seed=7 

#sampling method for training 
kfold_ada = KFold(n_splits=25, random_state=seed) 
model_ada = AdaBoostClassifier(n_estimators=num_trees, random_state=seed) 

results_ada = cross_val_score(model_ada, X_train, Y_train, cv=kfold_ada) 
print(results_ada.mean())

model_ada.fit(X_train,Y_train)

predicted_label = model_ada.predict([[7.2, 3.5, 0.8, 1.6]])


# Now predict 
ada_predict = model_ada.predict(X_validation)


# now check the prediction accuracy:

confusion_matrix(Y_validation,ada_predict)
report = classification_report(Y_validation, ada_predict) 

print(report)
#























