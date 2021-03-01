import csv
import json
import math
import geopy.distance
from geopy.distance import geodesic
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import copy
from matplotlib.pyplot import figure
import pandas as pd
from numpy.random import randn
from pandas import Series
import random
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
np.seterr(divide='ignore', invalid='ignore')
radius_km = [[40, 80, 120], [4, 4, 6]]
description = "econ"
check_goodness = True


def manager(radius, k):
    start_time = time.time()
    df = pd.read_csv('z:/Desktop/Research Data India/Spring-2021/Clustering_Datasets/Economic_Only_Clustering_Data_'+str(radius)+'.csv')
    #columns = df.columns
    #columnRelation = []
    #colStats = []
    #for column in columns:
    #    colStats.append([column, df[column].max(),df[column].min(),df[column].std(),df[column].median(),df[column].mean()])

    #for stats in colStats:
    #    print(stats)

    #for i in range(0, len(columns)-1):
    #    for j in range(0, len(columns)-1):
    #        if i != j and dbscan_clustering(df, i, j, 0) <= 10:
    #            print(columns[i])
    #            print(columns[j])
    #            if [columns[i], columns[j]] not in columnRelation and [columns[j], columns[i]] not in columnRelation:
    #                columnRelation.append([column1, column2])
    #            print(dbscan_clustering(df, i, j, 1))
    kmeans_clustering(df, radius, k)
    print("--- %s seconds ---" % (time.time() - start_time))

def kmeans_clustering(df, radius, k):
    X = pd.DataFrame(df)
    cluster_results = copy.deepcopy(X)
    df.fillna(0, inplace=True)
    scale = MinMaxScaler()
    scale.fit(X)
    scaled_data = pd.DataFrame(scale.transform(X))

    if check_goodness:
        kmeans_goodness(scaled_data, 15, radius)

    clusters = kmeans_cluster(scaled_data, k)[1]

    cluster_results['Cluster'] = clusters

    cluster_results.to_csv("z:/Desktop/Research Data India/Spring-2021/Clustering_Outputs/"+description+"_kmeans_"+str(k)+"_"+str(radius)+"km.csv", index=False)
    #print(cluster_results)

def kmeans_cluster(scaled_data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    clusters = kmeans.fit_predict(scaled_data)
    centers = kmeans.cluster_centers_
    inter_cluster_dist = 0
    intra_cluster_dist = 0
    dist_to_centroid = 0
    counter = 0

    for i in range(k):
        for j in range(k):
            if i != j:
                inter_cluster_dist += np.sum(np.square(centers[i]-centers[j]))
        counter += i
        #print(i, counter)
    if(counter == 0):
        counter = 1
    inter_cluster_dist = inter_cluster_dist/counter

    for i in range(k):
        for dataPoint in scaled_data[clusters==i]:
            #print(data[clusters==i])
            dist_to_centroid += np.sum(np.square(dataPoint-centers[i]))
        intra_cluster_dist += dist_to_centroid/len(scaled_data[clusters==i])

    goodness = (inter_cluster_dist + (k/intra_cluster_dist))
    print("goodness", k, inter_cluster_dist, intra_cluster_dist, goodness)
    #print(centers_new)
    return (goodness, clusters)

def kmeans_goodness(scaled_data, kRange, radius):
    rangeK = range(1,kRange+1)
    goodnesses = []
    
    for k in rangeK:
        #goodnesses.append(kmeans_custom(scaled_data.dropna(), k))
        goodnesses.append(kmeans_cluster(scaled_data, k)[0])

    plt.plot(rangeK, goodnesses, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Goodness of clusters')
    plt.axis([0, 15, 0, 6])
    plt.title('Goodness plot at '+str(radius)+' km')
    plt.savefig("z:/Desktop/Research Data India/Spring-2021/Clustering_Outputs"+description+"_Goodness_Plot_"+str(radius)+".png")
    plt.show()

def kmeans_custom(df, k):
    #print(df)
    dff = df.values[:, 0:4]
    data = pd.DataFrame(dff).dropna()
    
    n = data.shape[0]
    
    #c = data.shape[1]
    inter_cluster_dist = 0
    intra_cluster_dist = 0

    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", category=RuntimeWarning)
    #    mean = np.nanmean(data, axis = 0)
    #std = np.std(data, axis = 0)
   
    centers = data.sample(n=k).to_numpy()
    print(centers)
    centers_old = np.zeros(centers.shape)
    centers_new = deepcopy(centers)

    clusters = np.zeros(n)
    distances = np.zeros((n,k))

    error = np.linalg.norm(centers_new - centers_old)
    


    while error != 0:
        for i in range(k):
            distances[:,i] = np.linalg.norm(data - centers_new[i], axis=1)
        clusters = np.argmin(distances, axis = 1)
        centers_old = deepcopy(centers_new)
        
        for i in range(k):
            with warnings.catch_warnings():
                #if len(data[clusters==i]) <= 1:
                #    x=2
                #else:
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centers_new[i] = np.mean(data[clusters == i], axis=0)
                #print(data[clusters==i])
                #print(clusters)
                print(centers_old)
                
        error = np.linalg.norm(pd.DataFrame(centers_new) - pd.DataFrame(centers_old))
        print(k, error)
    #print(k)

    for i in range(k):
        for j in range(k):
            if i != j:
                inter_cluster_dist += np.linalg.norm(centers_new[i]-centers_new[j])
    inter_cluster_dist = inter_cluster_dist/k

    for i in range(k):
        for dataPoint in data[clusters==i]:
            #print(data[clusters==i])
            intra_cluster_dist += np.linalg.norm(dataPoint-centers_new[i])

    #intra_cluster_dist = intra_cluster_dist/k

    goodness = (inter_cluster_dist/(k/2) + (k/intra_cluster_dist))
    print("goodness", k, inter_cluster_dist, intra_cluster_dist, goodness)
    #print(centers_new)
    return goodness

def dbscan_clustering(df, column1, column2, visualize):
    df.fillna(0, inplace=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    df_normalized = normalize(scaled_data)
    df_normalized = pd.DataFrame(df_normalized)
    pca = PCA(n_components = 2) 
    df_principal = pca.fit_transform(df_normalized) 
    df_principal = pd.DataFrame(df_principal) 
    df_principal.columns = [column1, column2]

    x = .1
    y = 10
    db_default = DBSCAN(eps = x, min_samples = y).fit(df_principal)
    labels = db_default.labels_
    
    if visualize == 1:
        visualize_dbscan(labels, df_principal, column1, column2)

    return max(labels)

def visualize_dbscan(labels, clustered, column1, column2):
    colors = [random_color() for x in range(-1,max(labels))]
    colors[-1] = 'black'
    color_array = [colors[label] for label in labels]
    print(clustered.head())
    plt.figure(figsize =(9, 9)) 
    plt.scatter(clustered[column1], clustered[column2], c = color_array) 
    plt.show()
    plt.clf()

def random_color():
    color = [random.uniform(0.0, 1.0),random.uniform(0.0, 1.0),random.uniform(0.0, 1.0), 1]
    return tuple(color)

for radius in radius_km[0]:
    i = int((radius/40)-1)
    manager(radius, radius_km[1][i])