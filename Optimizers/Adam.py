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

def adam_gd(X, Y, epochs): 
  w, b, c = -2, -2, 1 
  beta1, beta2 = 0.45, 0.85
  eta = 1 
  mt_w, mt_b = 0, 0 
  vt_w, vt_b = 0, 0
  eps = 2 
  err_list = [] 
  W = [] 
  B = [] 
  for i in range(1, epochs+1): 
    temp_dw = 0 
    temp_db = 0 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b)
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
    mt_w = beta1*mt_w + (1-beta1)*temp_dw
    mt_b = beta1*mt_b + (1-beta1)*temp_db 
    vt_w = beta2*vt_w + (1-beta2)*temp_dw*temp_dw
    vt_b = beta2*vt_b + (1-beta2)*temp_db*temp_db 
    mt_hat_w = mt_w/(1-beta1**i) 
    vt_hat_w = vt_w/(1-beta2**i) 
    w = w - (eta*mt_hat_w)/(np.sqrt(vt_hat_w) + eps)
    mt_hat_b = mt_b/(1-beta1**i) 
    vt_hat_b = vt_b/(1-beta2**i) 
    b = b - (eta*mt_hat_b)/(np.sqrt(vt_hat_b) + eps)
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight is {w} and Bias is {b}") 
  return W, B, err_list
  return W, B, err_list

wt_adam, bias_adam, err_adam = adam_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_adam) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_adam, err_adam) 
plt.xlabel("Weights") 
plt.ylabel("Error") 
plt.show()
