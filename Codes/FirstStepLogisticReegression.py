#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:52:42 2024

@author: mali
"""
#%% import lbraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math 
np.set_printoptions(precision=2)

#%% load data

df=pd.read_csv('/Users/mali/Downloads/data.csv')

print(df.columns)

df.drop(['id','Unnamed: 32'],axis=1,inplace=True)
#%% list comprehension

df.diagnosis = [ 1 if each == 'M' else 0 for each in df.diagnosis]
y=df.iloc[:,0].to_numpy()
x=df.iloc[:,1:].to_numpy()

#%% normalization 

x = (x-np.min(x))/(np.max(x)-np.min(x))

#%% split 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#%% sigmoid function 

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
    g = 1/(1+np.exp(-z))
    
    return g

#%% parameter initialize
w_init = np.full_like(x_test[1],0.01)
b_init = 0.0

#%% Sigmoid test 

x_vec = x[0,:]

print(sigmoid(np.dot(x_vec,w_init)+b_init))

#%% Cost Function 

def compute_cost(x,y,w,b):
    """
 Computes cost

 Args:
   X (ndarray (m,n)): Data, m examples with n features
   y (ndarray (m,)) : target values
   w (ndarray (n,)) : model parameters  
   b (scalar)       : model parameter
   
 Returns:
   cost (scalar): cost
 """
    
    m = x.shape[0]
    
    cost = 0.0
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i],w)+b)
        
        cost += y[i]*np.log(f_wb_i) + (1-y[i])*np.log(1-f_wb_i)
        
    cost = -cost/m
    
    return cost

#%% Gradient Descent

def compute_gradient(x,y,w,b):
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = x.shape
    
    dj_dw = np.zeros(n)
    dj_db = 0.0
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i],w)+b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i*x[i,j]
            
        dj_db += err_i
        
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw,dj_db

def gradient_descent(x,y,w,b,alpha,iterations):
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    
    J_history=[]
    index=[]
    
    for i in range (iterations):
        dj_dw,dj_db = compute_gradient(x, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i%100 == 0:
            cost = compute_cost(x, y, w, b)
            J_history.append(cost)
            index.append(i)
            print("Cost after iteration %i : %f"%(i,cost))
            
    plt.plot(index,J_history)
    plt.xlabel('Index')
    plt.ylabel('Cost')
    plt.show()       
    return w,b

#%% Prediction

def predict(x,w,b):
    
    m=x.shape[0]
    y_prediction = np.zeros(m)
    for i in range (m):
        f_wb_i = sigmoid(np.dot(x[i],w)+b)
        
        if f_wb_i > 0.5:
            y_prediction[i] = 1
            
    
    return y_prediction

#%% Logistic Regression Model

def Logistic_Regression(x,y,w,b,alpha,iterations):
    
    w,b=gradient_descent(x, y, w, b, alpha, iterations)
    
    y_prediction = predict(x, w, b)
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction - y_test)) * 100))
    



Logistic_Regression(x_test, y_test, w_init, b_init, alpha=0.1, iterations=10000)

#%% with skit learn

from sklearn import linear_model 

log_reg = linear_model.LogisticRegression(random_state=7,max_iter=200)

log_reg.fit(x_train, y_train)

print(log_reg.score(x_test, y_test))
print(log_reg.score(x_train,y_train))

y_pred = log_reg.predict(x_test)

print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred - y_test)) * 100))


    
    
    
    
    
    




        
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


        
    




































    
    