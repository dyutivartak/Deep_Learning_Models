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

def adadelta(X, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  beta = 0.9
  eta = 1 
  eps = 2  
  vw = 0 
  vb = 0 
  W = [] 
  B = [] 
  err_list = [] 
  for i in range(epochs): 
    temp_dw = 0 
    temp_db = 0 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b) 
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
    
    vw = beta*vw + (1-beta)*temp_dw*temp_dw 
    vb = beta*vb + (1-beta)*temp_db*temp_db 

    w = w - (eta*temp_dw)/(np.sqrt(vw) + eps) 
    b = b - (eta*temp_db)/(np.sqrt(vb) + eps) 

    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b))) 
    print(f"After epoch {i+1}: Weight is {w} and Bias is {b}") 
  return W, B, err_list

wt_adadelta, bias_adadelta, err_adadelta = adadelta(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_adadelta) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_adadelta, err_adadelta) 
plt.xlabel("Weights") 
plt.ylabel("Error") 
plt.show()
