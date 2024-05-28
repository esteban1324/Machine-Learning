import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 


# load and process data here 
df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, 3:]


# building dendrogram to find optimal number of clusters 
def plot_dendrogram(X):
    plt.figure(figsize=(16,8))
    d_plot = sch.dendrogram(sch.linkage(X, method='ward', metric='euclidean'))
    plt.title('hierarchical cluster dendrogram')
    plt.xlabel('customers')
    plt.ylabel('Distance')
    plt.show()

#plot_dendrogram(X)

# plot to show the clusters based on the two features 
def scatter_plot(X):
    data = X.values
    plt.scatter(data[:, 0], data[:, 1])
    plt.title('scatter plot')
    plt.xlabel('annual income')
    plt.ylabel('spending score')
    plt.show()
    
#scatter_plot(X)

# function to perform hierarchical clustering 
def clustering(X):
    hc = AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward')
    y_hc = hc.fit_predict(X.values)

    return y_hc

# plot the cluster labels with the scatter plots, show there is some clustering 
def plot_result(X, assignments):
    data = X.values
    plt.scatter(data[assignments==0, 0], data[assignments==0, 1], s=100, c='cyan')
    plt.scatter(data[assignments==1, 0], data[assignments==1, 1], s=100, c='yellow')
    plt.scatter(data[assignments==2, 0], data[assignments==2, 1], s=100, c='green')
    plt.scatter(data[assignments==3, 0], data[assignments==3, 1], s=100, c='red')
    plt.scatter(data[assignments==4, 0], data[assignments==4, 1], s=100, c='black')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('cluster assignements')
    plt.show()

# correlation matrix calc 
def heatmap(X):
    sns.clustermap(X, row_cluster=False, col_cluster=True, cmap='viridis',figsize=(7,7))
    plt.title('heatmap')
    plt.show()


def heatmap2(X):
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
    
    
# sillhouette plot for hierarchal clustering 

# calling functions
data = X.values
predicted = clustering(X)
plot_result(X, predicted)
heatmap(data)
heatmap2(X)