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
import datetime as dt
import geocoder
from geopy.geocoders import Nominatim
from geopy.geocoders import GoogleV3
import googlemaps
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta

sourceGDelt = 'Z:/Desktop/Research Data India/Spring-2021/Source_Data/GDELT_State_Filtered_Events.csv'
sourceDes = 'Z:/Desktop/Research Data India/Spring-2021/Source_Data/DesInventar_Data.xlsx'
protestEvents = ['140','141', '144', '145', '153', '173', '175', '1411', '1412', '1413', '1414', '1441',  '1442',  '1443',  '1444', '1451', '1452', '1453',  '1454',  '1723',  '1724']
aidEvents = ['73']
govEvents = ['1323', '1324', '153', '1663', '1723', '1724', '173', '175']
ranges = [40, 80, 120]

print("Loading Data...")

Disasters = pd.read_excel(sourceDes).fillna('')
Gdelt = pd.read_csv(sourceGDelt).fillna('')

print("Data Loaded")

Gdelt['Date'] = Gdelt['Date'].apply(lambda date: dt.datetime(int(str(date)[0:4]), int(str(date)[4:6]), int(str(date)[6:8])))
Gdelt = Gdelt[(Gdelt['Date'] >= dt.datetime(2006,1,1)) & (Gdelt['Date'] < dt.datetime(2014,1,1))]
Disasters.rename(columns = {"Date (YMD)": "Date"}, inplace = True) 
Disasters['Date'] = Disasters['Date'].apply(lambda date: dt.datetime(int(date.split("/")[0]), int(date.split("/")[1]) if int(date.split("/")[1]) in range(1,12) else 1 , 28 if int(date.split("/")[2]) > 28 else 1 if int(date.split("/")[2]) < 1 else int(date.split("/")[2])))
Disasters = Disasters[(Disasters['Date'] >= dt.datetime(2006,1,1)) & (Disasters['Date'] < dt.datetime(2013,1,1))]

locations = copy.deepcopy(Disasters[['State', 'District', 'Block']])
locations['latLong'] = ''
geocodedLoc = locations.drop_duplicates()

geoDisasters = pd.read_csv('z:/Desktop/Research Data India/Spring-2021/Clustering_Datasets/Geocoded_DesInventar.csv')


trimmedDisasters = copy.deepcopy(Disasters[['Date', 'Deaths', 'Injured', 'Houses Destroyed', 'Houses Damaged', 'Directly affected', 'Indirectly Affected', 'Duration (d)', 'Health sector', 'Agriculture', 'Water supply', 'Sewerage', 'Industries', 'Communications', 'Transportation', 'Power and Energy', 'Relief', 'Economic loss (infrastructure)', 'Economic loss (w/Agriculture)', 'Event']])
trimmedDisasters['Lat'] = ''
trimmedDisasters['Long'] = ''
for index, disaster in geoDisasters.iterrows():
    #print(str(geoDisasters.loc[index, 'latLong']).split(", ")[0].replace("[",""))
    print(str(geoDisasters.loc[index, 'latLong']).split(", "), '####', geoDisasters.loc[index, 'latLong'])
    if geoDisasters.loc[index, 'latLong'] != 'nan' and geoDisasters.loc[index, 'latLong'] != '' and geoDisasters.loc[index, 'latLong'] is not None:
        trimmedDisasters.loc[index, 'Lat'] = str(geoDisasters.loc[index, 'latLong']).split(", ")[0].replace("[","")
        trimmedDisasters.loc[index, 'Long'] = str(geoDisasters.loc[index, 'latLong']).split(", ")[1].replace("]","")
    else:
        trimmedDisasters.loc[index, 'Lat'] = ''
        trimmedDisasters.loc[index, 'Long'] = ''

Gdelt = Gdelt[Gdelt['CAMEOCode'] in protestEvents or Gdelt['CAMEOCode'] in aidEvents or Gdelt['CAMEOCode'] in govEvents]
Gdelt = Gdelt[['Date', 'CAMEOCode', 'ActionGeoLat', 'ActionGeoLong']]
Gdelt.columns = ['Date', 'CameoCode', 'Lat', 'Long']

def geocodeLocation(index, row):
    try:
        #html = requests.get(url, stream=True)
        #open(f'{file_name}.json', 'wb').write(html.content)
        #print("row:", row, "printed#######################")
        geocodedLoc.loc[index, 'latLong'] = geocoder.bing(geocodedLoc.loc[index, 'Block']+', '+geocodedLoc.loc[index, 'District']+', '+geocodedLoc.loc[index, 'State']+', India', key='AvXEL2IXYy89DtKTFqDlt5eujV_JyPT9lQ564nEPJgmETY-ye9Yj59DvYn02p-GK').latlng
        #geocodedLoc.loc[(geocodedLoc['State'] == row[0]) & (geocodedLoc['District'] == row[1]) & (geocodedLoc['Block'] == row[2])] = row
        print("row:", row, "printed#######################")
        return row
    except requests.exceptions.RequestException as e:
        return e

def runner(variable):
    threads= []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for index, row in geocodedLoc.iterrows():
            #file_name = uuid.uuid1()
            threads.append(executor.submit(geocodeLocation(index, row)))
           
        #for task in as_completed(threads):
            #print(task.result) 
            #geocodedLoc.loc[(geocodedLoc['State'] == task.result[0]) & (geocodedLoc['District'] == task.result[1]) & (geocodedLoc['Block'] == task.result[2])] = task.result[3]
    return variable

def geocodingData(Disasters):
    print("Geocoding Data...")
    start_time = time.time()

    runner(1)

    print("Geocoding Done; Process Took %s Seconds" % (time.time() - start_time))
    geocodedLoc.to_csv('z:/Desktop/Research Data India/Spring-2021/Clustering_Datasets/Geocoded_DesInventar_distinct.csv')

def generateTimelines(trimmedDisasters, Gdelt, kmRange):

    timelines = []
    
    for disaster in trimmedDisasters.iterrows():
        distFilter = geodesic((disaster['Lat'], disaster['Long']),(Gdelt['Lat'],Gdelt['Long'])).km < kmRange
        dateFilter = disaster['Date'] <= Gdelt['Date'] <= (disaster['Date'] + relativedelta(years=1))
        timelineEvents = Gdelt[dateFilter and distFilter]
        timelineEvents = pd.concat([disaster, timelineEvents]).reset_index(drop = True)
        if len(timelineEvents) > 1:
            timelines.append(copy.deepcopy(timelineEvents))
        timelineEvents.clear()
    return timelines

def mainFunction(kmRange):

    timelines = generateTimelines(trimmedDisasters, Gdelt, kmRange)

    print(timelines)
#print(Disasters['latLong'])

for kmRange in ranges:
    mainFunction(kmRange)