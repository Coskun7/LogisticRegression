#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:33:08 2024

@author: mali
"""

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/mali/Downloads/heart_2020_cleaned.csv')


object_columns = df.select_dtypes(include='object').columns


for column in object_columns:
    unique_values = df[column].dropna().unique()  
    if set(unique_values) == {'Yes', 'No'}:  
        df[column] = df[column].apply(lambda x: 1 if x == 'Yes' else 0)
        

df.GenHealth = [
    4 if each == 'Excellent' 
    else 3 if each == 'Very good' 
    else 2 if each == 'Good' 
    else 1 if each == 'Fair' 
    else 0 
    for each in df.GenHealth
]
df.Sex = [1 if each == 'Male' else 0 for each in df.Sex]

df.drop(['AgeCategory','Race','Diabetic'],axis=1,inplace=True)

x = df.iloc[:,1:15].to_numpy()
y = df.iloc[:,0].to_numpy()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_ = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_,y,test_size=0.2,random_state=42)

from sklearn import linear_model 

log_reg = linear_model.LogisticRegression(random_state=7,max_iter=200)

log_reg.fit(x_train, y_train)

print(log_reg.score(x_test, y_test))
print(log_reg.score(x_train,y_train))

y_pred = log_reg.predict(x_test)

print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred - y_test)) * 100))