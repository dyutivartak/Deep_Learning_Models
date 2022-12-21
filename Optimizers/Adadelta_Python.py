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

def adagrad_gd(X, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  eta = 1 
  eps = 2 
  err_list = [] 
  W = [] 
  B = [] 
  v_w = 0 
  v_b = 0 
  for i in range(epochs): 
    temp_dw = 0 
    temp_db = 0 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b)
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 

    v_w = v_w + temp_dw**2 
    v_b = v_b + temp_db**2 

    w = w - (eta*temp_dw)/(np.sqrt(v_w + eps)) 
    b = b - (eta*temp_db)/(np.sqrt(v_w + eps)) 

    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b))) 
    print(f"After epoch {i+1}: Weight is {w} and Bias is {w}") 
  return W, B, err_list

wt_adagrad, bias_adagrad, err_adagrad = adagrad_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_adagrad) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_adagrad, err_adagrad) 
plt.xlabel("Weight") 
plt.ylabel("Error") 
plt.show()
