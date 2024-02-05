

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plot the data points from the given X array, 

# the first 3 belong to class 1 and the last 3 belong to class 2
X = [(1,1), (1,2), (2,1), (0,0), (1,0), (0,1)]


 
def plot_data(X):
    x_coord_class1 = [points[0] for points in X[:3]]    
    y_coord_class1 = [points[1] for points in X[:3]]
    x_coord_class2 = [points[0] for points in X[3:]]
    y_coord_class2 = [points[1] for points in X[3:]]
    
    plt.xlabel('x')
    plt.ylabel('y')
       
    plt.scatter(x_coord_class1, y_coord_class1, color='blue', label='Class 1')
    plt.scatter(x_coord_class2, y_coord_class2, color='red', label='Class 2')
    plt.legend(loc = 'upper left')
    
    
    plt.show()
if __name__ == "__main__":
    plot_data(X)
