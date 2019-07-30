# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:18:22 2019

@author: SusarlaS
"""

#Dropping rows and columns in Pandas
import pandas as pd
import numpy as np
raw_data = {'name': ['siva', 'raghuram', 'susarla', 'rajaram'],
        'age': [20, 19, 22, 21],
        'favorite_color': ['blue', 'red', 'yellow', "green"],
        'grade': [88, 92, 95, 70]}

df = pd.DataFrame(raw_data, index = ['name1', 'name2', 'name3', 'name4'])

#drop a row by name
df.drop(['name2'])  # raghuram row name 'name2' dropped




#drop a row by number
df.drop(df.index[-1])     #-1 is last row number  0 is the first row

#drop a column by name
df.drop('age',axis =1)

#drop column by number
df.drop(df.columns[2],axis=1)


#Select rows from a Pandas DataFrame based on values in a column
df.loc[df['age'] == 20]

df.loc[df['age'] > 20]

search_colors = ['pink','green']
df.loc[df['favorite_color'].isin(search_colors)]
       
       
# Rename a column
df.rename(columns = {'age': 'vayasu'},inplace = True)       

df.rename(columns = {'vayasu': 'age'},inplace = True)      


#Create a new column
df['new column'] = 0


# updating row value based on column values  
df.loc[df['new column'] == 0,'new column'] = 500


#Change the order of columns in Pandas dataframe
df = df[['favorite_color','grade','name','new column','age']]






 