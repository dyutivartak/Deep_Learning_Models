import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

X = np.array([0.5, 2.5]) 
Y = np.array([0.2, 0.9])

def sigmoid(x, w, b): 
  y_in = np.dot(w, x) + b 
  y_hat = 1/(1 + np.exp(-y_in)) 
  return y_hat

def error(y, y_hat): 
  err = np.array((y-y_hat)**2).mean() 
  return err 

def delta_w(x, y, y_hat, c): 
  dw = c*(y_hat-y)*y_hat*(1-y_hat)*x
  return dw 

def delta_b(y, y_hat, c): 
  db = c*(y_hat-y)*y_hat*(1-y_hat)
  return db

def batch_gd(x, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  err_list = [] 
  W = [] 
  B = [] 
  for i in range(epochs): 
    temp_w = 0 
    temp_b = 0
    for x, y in zip(X, Y):
      y_hat = sigmoid(x, w, b) 
      temp_w += delta_w(x, y, y_hat, c) 
      temp_b += delta_b(y, y_hat, c) 
    temp_w = temp_w/len(Y) 
    temp_b = temp_b/len(Y) 
    w += temp_w 
    b += temp_b 
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight ==> {w} and Bias ==> {b}") 
  return W, B, err_list

wt_bgd, b_bgd, err_bgd = batch_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_bgd) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_bgd, err_bgd) 
plt.xlabel("Weight") 
plt.ylabel("Error") 
plt.show()
