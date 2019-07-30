# in pandas, y default string data type is stored as object
# strings missing values will be with '.' and integers with 'NaN' 
# to replace such values we can use single command like
# df = pd.read_csv("data/cereal.csv", skiprows = 1, na_values = ['no info', '.'])

# go through this  blog for detailed csv read in pandas
 # https://www.marsja.se/pandas-read-csv-tutorial-to-csv/
 
 
 #Data Preprocessing Techiques
 #https://towardsdatascience.com/data-pre-processing-techniques-you-should-know-8954662716d6
 
 
# for one hot encoding 
#https://www.coursera.org/lecture/competitive-data-science/categorical-and-ordinal-features-qu1TF
# https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding
# https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
# https://www.datacamp.com/community/tutorials/categorical-data
#https://medium.com/hugo-ferreiras-blog/dealing-with-categorical-features-in-machine-learning-1bb70f07262d

# when to use dimensionality reduction?
# dimensionality reduction techniques for regression and classification?
# to overcome curse of dimensionality  we can use "Feature Hashing , Hashing Trick"
# Hashing refer https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512



# https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/
# good one
#  https://medium.freecodecamp.org/you-need-these-cheat-sheets-if-youre-tackling-machine-learning-algorithms-45da5743888e
# https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/
#https://www.analyticsvidhya.com/blog/2015/09/full-cheatsheet-machine-learning-algorithms/


#scaling
#standardization
#Normalization



#########visualization
# Numeric variables use box plots



## Classification:
    #Linear Classification       : LDA,LOGISTIC REGRESSION,LOGISTIC REGRESSION MULTI CLASS
    #Non Linear Classification   : CART,KNN,NAIVE BAYES, SVM, C4.5, C5.0,
 
    
    
# Classfication Report    
#TN / True Negative: case was negative and predicted negative
#TP / True Positive: case was positive and predicted positive
#FN / False Negative: case was positive but predicted negative
#FP / False Positive: case was negative but predicted positive



#Precision – What percent of your predictions were correct?
#Precision – Accuracy of positive predictions.
#Precision = TP/(TP + FP)
#
#Recall – What percent of the positive cases did you catch? 
#Recall: Fraction of positives that were correctly identified.
#Recall = TP/(TP+FN)


### When we run K-FOLD cross-validation,then we can summerize using a mean and standard deviation
# For k-fold we always use a mean value (beacuse severeal folds will have several low-high values)


#Feature Selection:
# Irrelevent or unneccessary featrures during training may influence the performance of the model. So
#it is must to eliminate unneccessary features    
# FEATURE Extraction: transformation of raw data into features suitable for modeling;
# feature transformation transformation of data to improve the accuracy of the algorithm;
# feature Selection    removing unnecessary features.


#kfold scoring = 'accuracy'
precision_macro,recall_macro


 https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019
    
#For categorical variables we use PCA as Feature Selection    , but ExtraTreeClassifier is very easy to understand

# Ensembles : 
#    a set of samples are taken from the data and will be worked on different modes for output, at last all the 
#    outputs are merged to find the final output(prediction). 
#     Bagging  : Random Forest and Extra Tree Classifiers
#     Boosting :
#     Voting   :
#    Difference between Decision Tree and Random Forest is :  Samples of the training dataset are taken with replacement, but the trees are constructed in a way that reduces the correlation between individual classiﬁers. Speciﬁcally, rather than greedily choosing the best split point in the construction of each tree, only a random subset of features are considered for each split



             

    
    
    
 # Standardization,Normalization and Binarization are 3 techniques if the values are not on same scale.
 # 
    
    

#To Save our final model we can use either Pickle or Joblib



# 13+ algoirthms explained
#https://www.newtechdojo.com/list-machine-learning-algorithms/

#clustering 
#https://towardsdatascience.com/unsupervised-learning-with-python-173c51dc7f03






# after understanding new strurcture

# LOAD DATA
# FEATURE SELECTION and Dimensionality Reduction
# SAMPLING METHOD TRICK
# PICK A MODEL
# TRAIN IT AND TRACK TRAINING SUCCESS RATE BY CROSS_VAL_SCORE
# FIT THE MODE
# PREDICT THE RESULT 
#Model evaluation: quantifying the quality of predictions USE Cross_
# CHECK THE ACCCUACY USING CLASSIFICAION REPORT/CLASSIFICATION ACCURACY/CONFUSION MATRIX
# IMPROVE THE RESULTS USING HYPERPARAMETER TUNING

# TRY DIFFENT ALGORITHMS USING BAGGING AND BOOSTING 
 

#CREATE A PIPELINE FOR A FLOW OF TASKS LIKE, HAVING A SAMPLING METHOD,



#Grid Search Algorithms can be used on Logistic Regrission and SVM. It is basically 
# developed for Neural Networkds

Entropy:
#    OK, let’s take an example to understand this better. When you use fuel to run your car, a perfectly ordered petrol (compact energy) is converted/dissipated to disordered forms of energy like heat, sound, vibrations etc. The work is generated in the process to run the engine of the car. The more disordered or random the energy is the harder/impossible it is to extract purposeful work out of it. So I guess, we care about work and not energy. In other words, higher the entropy or randomness of a system the harder it is to convert it to meaningful work. Physicists define the entropy of a system with the following formula

