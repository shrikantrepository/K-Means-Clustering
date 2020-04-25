import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

""" Problem statement is we need to cluster this dataset in some groups based on the Annual Income
and Spending Score to understand spending patter of people based on annual salary"""

# Here we are going to take only two independent variable 'Annual Income and Spending score' 
X= dataset.iloc[:,[3,4]].values    

# We need to find out how many clusters we can use here to get solution of our problem statment 
# In order to find out what is optimal number of cluster we will use elbow method
# This is called as elbow method because shape of plot is like human bone elbow 
# Once we plot elbow method then we need to select a 
from sklearn.cluster import KMeans

wcss=[]         # Within cluster sum of squared 
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)  #Kmeans++ - Algorithm used to initilise the centroids
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia gives a values of WCSS for each cluster as the for loop goes

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

""" By using elbow method we have consutructed a clusters which shows wcss value
wcss scores is basically summation of ecludian distance between the points that are actuall scattered
You will get two main diversion on the curve where we get a minimal number of WCSS value and based on that 
we will select how manay number of clusters are required"""

#Fitting K-MEans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)   # number of clusters = 5 as per elbow method
y_kmeans=kmeans.fit_predict(X)

#Visualize the clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

""" Yellow points in the above scatter plots are centroids which is a mean of all values falls
within that respective cluster.
Conclusion - People who are earning between 20-40 spending less ie 0-40. There is a group of 
people who earn between 20-40 but spend very aggresively. same way for other clusters."""

