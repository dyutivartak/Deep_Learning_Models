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

def mini_batch_gd(X, Y, epochs): 
  batch_size = 1 
  w = -2 
  b = -2 
  c = 1 
  err_list = [] 
  W = [] 
  B = [] 
  for i in range(epochs): 
    temp_dw = 0 
    temp_db = 0 
    counter = 0
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b) 
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
      counter += 1 
      if(counter % batch_size == 0):  
        w += temp_dw 
        b += temp_db 
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight ==> {w} and Bias ==> {b}") 
  
  return W, B, err_list

wt_mb, bias_mb, err_mb = mini_batch_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_mb) 
plt.xlabel("Epochs") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_mb, err_mb) 
plt.xlabel("Weights") 
plt.ylabel("Error") 
plt.show()
