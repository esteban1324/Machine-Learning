import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

import plotly as py 
import plotly.graph_objs as go

from sklearn import preprocessing 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 


# load and process data here 
df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:, 3:]


# building dendrogram to find optimal number of clusters 
plt.figure(figsize=(16,8))
d_plot = sch.dendrogram(sch.linkage(X, method='ward', metric='euclidean'))
plt.title('hierarchical cluster dendrogram')
plt.xlabel('customers')
plt.ylabel('Distance')
plt.show()

# building a hierarchal cluster model to implement and classify clusters.  



