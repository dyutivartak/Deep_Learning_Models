import numpy as np
x1 = np.array([0])
x2 = np.array([1])
x = [x1,x2]
w = [-1]
b = 0
for i in x:
    y = np.dot(i,w) + b
    if y >= 0:
        print(f"NOT {i}: 1")
    else:
        print(f"NOT {i}: 0")