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
import random
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
radius_km = [40, 80, 120]

def only_deaths(df, radius):
    events_death = df[df.deaths != '']
    events_death = events_death[events_death.deaths != '1']
    events_death = events_death[events_death.deaths != '2']
    events_death = events_death[events_death.deaths != '3']
    events_death = events_death[events_death.deaths != '4']
    print(events_death)
    events_death.to_csv('z:/Desktop/Research Data India/Spring-2021/Clustering_Datasets/Deaths_Only_Clustering_Data_'+str(radius)+'.csv')

def only_economic(df, radius):
    has_economic = df[df.economic_loss!='']
    events_economic = has_economic
    events_economic.to_csv('z:/Desktop/Research Data India/Spring-2021/Clustering_Datasets/Economic_Only_Clustering_Data_'+str(radius)+'.csv')

def manager(radius):
    start_time = time.time()
    df = pd.read_csv('z:/Desktop/Research Data India/Spring-2021/Datasets/Timeline_Data_Output_Trimmed'+str(radius)+'.csv', na_filter=False)
    df.fillna('')
    only_deaths(df, radius)
    only_economic(df, radius)

for radius in radius_km:
    manager(radius)