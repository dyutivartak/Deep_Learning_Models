import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def func(X,w,b):
    return np.dot(X,w) + b

def sig(yin):
    return 1/(1+ math.exp(-yin))

def mse(yhat,y):
    mse=0
    for i in range(len(y)):
        mse+=(yhat[i]-y[i])**2
    return mse/2

def plot_mse(mse_list,epochs):
    x=[i for i in range(epochs)]
    plt.plot(x,mse_list)

def stochastic_gd (X,Y):
    w,b,c,epoch = -2,-2,1,300
    mse_list=[]
    for i in range(epoch):
        result = []
        for x,y in zip(X,Y):
            yhat = sig(func(x,w,b))
            delw = c*(y-yhat)*yhat*(1-yhat)*x
            delb = c*(y-yhat)*yhat*(1-yhat)
            w += delw
            b += delb
            result.append(yhat)
            if (len(result)==len(Y)):
              mse_list.append(mse(result,Y))
    #print(f"Epoch: {i+1}\tWeight: {w}, Bias: {b}\t y_hat: {result}\t MSE: {mse_list[-1]}")
    plot_mse(mse_list,epoch)
    print(f"\n\nFinal weights: {w}\tBias {b}\t\tMean Squared Error: {mse_list[-1]}")

X = [0.5,2.5]
Y = [0.2,0.9]
stochastic_gd(X,Y)
