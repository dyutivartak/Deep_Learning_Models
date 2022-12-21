import numpy as np
x1 = np.array([0,0])
x2 = np.array([0,1])
x3 = np.array([1,0])
x4 = np.array([1,1])
x = [x1,x2,x3,x4]
w = [1,1]
b = -1
for i in x:
    y = np.dot(i,w) + b
    if y >= 0:
        print(f"OR {i[0],i[1]}: 1")
    else:
        print(f"OR {i[0],i[1]}: 0")