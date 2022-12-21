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

def nag_gd(X, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  eta = 1 
  beta = 0.9 
  W = [] 
  B = [] 
  err_list = [] 
  prev_vw = 0 
  prev_vb = 0 
  for i in range(epochs): 
    temp_dw = 0
    temp_db = 0 
    vw = w - beta*prev_vw  
    vb = b - beta*prev_vb 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, vw, vb) 
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
    v_w = beta*prev_vw + eta*temp_dw 
    v_b = beta*prev_vb + eta*temp_db 
    w = w - v_w 
    b = b - v_b 
    prev_vw = vw 
    prev_vb = vb 
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight is {w} and Bias is {b}") 
  return W, B, err_list

wt_nag, bias_nag, err_nag = nag_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_nag) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_nag, err_nag) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()
