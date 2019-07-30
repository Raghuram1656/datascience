# Python Project Template Filler
# 1. Prepare Problem 
# a) Load libraries 
import pandas as pd
# b) Load dataset
input_file = 'C:/Users/susarlas/Desktop/data science/projects/classification/data files/iris.csv'
df = pd.read_csv(input_file,skiprows = 1) 
#to remove the first row if it is not a correct row
#Missing values filler
df = pd.read_csv(input_file, skiprows = 1, na_values = ['no info', '.'])

print(df.info())
# 2. Summarize Data 
# a) Descriptive statistics 
# b) Data visualizations


# 3. Prepare Data 
# a) Data Cleaning 
# b) Feature Selection 
# c) Data Transforms


# 4. Evaluate Algorithms 
# a) Split-out validation dataset 
# b) Test options and evaluation metric 
# c) Spot Check Algorithms 
# d) Compare Algorithms


# 5. Improve Accuracy 
# a) Algorithm Tuning 
# b) Ensembles


# 6. Finalize Model 
# a) Predictions on validation dataset 
# b) Create standalone model on entire training dataset 
# c) Save model for later use
