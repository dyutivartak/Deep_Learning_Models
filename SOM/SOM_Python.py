!pip install minisom
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from minisom import MiniSom 

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Credit_Card_Applications.csv') 
data 

data.shape

data.info() 

X = data.iloc[:, 1:14].values 
y = data.iloc[:, -1].values 
pd.DataFrame(X) 

from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0, 1)) 
X = sc.fit_transform(X) 
pd.DataFrame(X) 

som_grid_rows = 10 
som_grid_columns = 10 
iterations = 20000 
sigma = 1 
learning_rate = 0.5 
som = MiniSom(x = som_grid_rows, y = som_grid_columns, input_len=13, sigma=sigma, learning_rate=learning_rate) 
som.random_weights_init(X) 
som.train_random(X, iterations) 
som.distance_map() 

from pylab import plot, axis, show, pcolor, colorbar, bone 
bone() 
pcolor(som.distance_map().T)       # Distance map as background 
colorbar() 
show() 
bone() 
pcolor(som.distance_map().T) 
colorbar() #gives legend 
markers = ['o', 's']                 # if the observation is fraud then red circular color or else green square 
colors = ['r', 'g'] 
for i, x in enumerate(X): 
    w = som.winner(x) 
    plot(w[0] + 0.5, 
         w[1] + 0.5, 
         markers[y[i]], 
         markeredgecolor = colors[y[i]], 
         markerfacecolor = 'None', 
         markersize = 10, 
         markeredgewidth = 2) 
show() 

mappings = som.win_map(X) 
mappings 
mappings.keys() 
len(mappings.keys()) 
mappings[(9,8)] 
frauds = np.concatenate((mappings[(0,9)], mappings[(8,9)]), axis = 0) 
frauds 
frauds1 = sc.inverse_transform(frauds) 
pd.DataFrame(frauds1) 
