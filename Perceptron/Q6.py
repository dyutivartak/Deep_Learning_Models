import numpy as np
def perceptron(w,x,b):
    v = np.dot(w,x) + b
    if v >= 0:
        return 1
    else:
        return 0
  
def not_function(x):
    w_not = [-1]
    b_not = 0
    return perceptron(x, w_not, b_not)
  
def and_function(x):
    w = [1, 1]
    b_and = -2
    return perceptron(x, w, b_and)
  
def or_function(x):
    w = [1, 1]
    b_or = -1
    return perceptron(x, w, b_or)
  
def xor_function(x):
    y1 = and_function(x)
    y2 = or_function(x)
    y3 = not_function(y1)
    final_x = np.array([y2, y3])
    ans = and_function(final_x)
    return ans
  
x1 = np.array([0, 0])
x2 = np.array([0, 1])
x3 = np.array([1, 0])
x4 = np.array([1, 1])
x = [x1,x2,x3,x4]
for i in x:
    print(f"XOR {i[0],i[1]}: {xor_function(i)}")
