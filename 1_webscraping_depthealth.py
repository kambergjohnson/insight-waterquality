import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import time
import urllib.request
import json
import sys

def webscraping_depthealth(html_file, key, position):
    '''Takes a downloaded html file from the Hawaii Department of Health
    website and converts it into a convenient pandas dataframe.
    Input:
    html_file=string containing path to downloaded .html
    key=string proceeding start of data
    position=number of keys in document before start of data'''
    #decode html
    soup = BeautifulSoup(open(html_file), "html.parser")
    #create a table starting after position number of keys
    table = soup.find_all(key)[position] 
    df = pd.read_html(key)[0]
    #labels the columns
    df.columns = df.iloc[0]
    #remove first row because it contains column info
    df = df[df.index != 0]
    return df

def numericise(df,column_list):
    '''Takes a list of column names from dataframe and makes the columns numeric'''
    for x in column_list:
        df[x] = pd.to_numeric(df[x], errors='coerce')
    return df

def day_to_unix(df, date):
    '''Takes date column of dataframe and converts into unix time'''
    df['Datetime'] = df[date].apply(lambda t: pd.Timestamp(t))
    df['Datetime'] = pd.to_datetime(df['Datetime']).astype(int)
    #remove possible nans
    df = df[np.isfinite(df['Datetime'])]
    df = df.reset_index()
    #convert from microseconds to seconds (more usable)
    df['Datetime'] = df['Datetime'].apply(lambda x:'%10.f' % (x/1000000))
    df['Datetime'] = df['Datetime'].apply(lambda x:'%10.f' % (float(x)/1000))
    return df

def no_nans(df,column_list):
    '''Remove rows with nans for particular columns we care about'''
    for x in column_list:
        df = df[np.isfinite(df[x])]
    df = df.reset_index()
    return df

def feasible(df, column, greaterthan, lessthan):
    '''Remove columns that do not have feasible values'''
    df = df[(df[column] > greaterthan) & (df[column] < lessthan)]
    return df

def lists_for_darksky_api(df, datetime, latitude, longitude):
    '''Create lists from date column, latitude column, longitude column to query darksky api'''
    unixday = 86400
    date_list = []
    latitude_list = []
    longitude_list = []
    
    for index, row in df.iterrows():
        date_list.append(row[datetime])
        latitude_list.append(row[latitude])
        longitude_list.append(row[longitude])
    return date_list, latitude_list, longitude_list

def query_darksky(key, date_list, latitude_list, longitude_list):
    '''Obtain weather information from darksky.
    Inputs:
    key = darksky private key
    date_list = list of dates in unix
    latitude_list = list of latitudes
    longitude_list = list of longitudes'''
    precipIntensity_1 = []
    precipIntensityMax_1 = []
    temperatureHigh_1 = []
    temperatureLow_1 = []
    dewPoint_1 = []
    humidity_1 = []
    pressure_1 = []
    windSpeed_1 = []
    windBearing_1 = []
    cloudCover_1 = []

    for n in range(len(date_list)):
        #for every date/lat/long group create url to query darksky
        temp_list = []
        temp_list2 = []
        temp_list.append(str(latitude_list[n]))
        temp_list.append(str(longitude_list[n]))
        temp_list.append(str(date_list[n]))
        data = ','.join(temp_list)
        temp_list2.append('https://api.darksky.net/forecast/')
        #next comes the darksky private key
        temp_list2.append(key)
        #next comes the date/lat/long
        temp_list2.append(data)
        data2 = ''.join(temp_list2)
        #finally the request
        full_data = json.loads(urllib.request.urlopen(data2).read().decode())

        precipIntensity_1.append(full_data['daily']['data'][0]['precipIntensity'])
        precipIntensityMax_1.append(full_data['daily']['data'][0]['precipIntensityMax'])
        temperatureHigh_1.append(full_data['daily']['data'][0]['temperatureHigh'])
        temperatureLow_1.append(full_data['daily']['data'][0]['temperatureLow'])
        dewPoint_1.append(full_data['daily']['data'][0]['dewPoint'])
        humidity_1.append(full_data['daily']['data'][0]['humidity'])
        pressure_1.append(full_data['daily']['data'][0]['pressure'])
        windSpeed_1.append(full_data['daily']['data'][0]['windSpeed'])
        windBearing_1.append(full_data['daily']['data'][0]['windBearing'])
    
    return precipIntensity_1, precipIntensityMax_1, temperatureHigh_1, temperatureLow_1, dewPoint_1, humidity_1, pressure_1, windSpeed_1, windBearing_1, cloudCover_1
        
def add_list_to_df(df, list_of_columns_to_add):
    '''Take lists and add to dataframe'''
    for x in list_of_columns_to_add:
        series = pd.Series(x)
        df[str(x)] = series.values
    return df

def main():
    #parse depthealth html file for data
    data_html = sys.argv[1]
    df = webscraping_depthealth(data_html, 'table', 1)

    #make sure everything is numbers
    df = numericise(['CP Result','Dissolved Oxygen', 'Dissolved Oxygen Saturation', 'pH', 'Location Identifier', 'Ent Results', 'Lat Dec Deg', 'Long Dec Deg', 'Salinity', 'Turbidity'])

    #get unix date/time
    df = day_to_unix(df, 'Date')

    #remove nans
    df = no_nans(df, ['Lat Dec Deg', 'Long Dec Deg', 'Datetime'])

    #remove not feasible stuff
    df = feasible(df, 'Lat Dec Deg', 21, 22)
    df = feasible(df, 'Long Dec Deg', -159, -157)
    df = feasible(df, 'Datetime', 946651738, 1514731738)

    #make lists of information needed for darksky api
    date_list, latitude_list, longitude_list = lists_for_darksky_api(df, 'Datetime', 'Lat Dec Deg', 'Long Dec Deg')

    #query darksky
    key = sys.argv[2]
    precipIntensity_1, precipIntensityMax_1, temperatureHigh_1, temperatureLow_1, dewPoint_1, humidity_1, pressure_1, windSpeed_1, windBearing_1, cloudCover_1 = query_darksky(key, date_list, latitude_list, longitude_list)

    #add darksky info to df
    df = add_list_to_df(df, [precipIntensity_1, precipIntensityMax_1, temperatureHigh_1, temperatureLow_1, dewPoint_1, humidity_1, pressure_1, windSpeed_1, windBearing_1, cloudCover_1])

    #save final df as a tsv
    df.to_csv('WaterQualityWeatherData.tsv', sep='\t')

if __name__ == '__main__':
    main()