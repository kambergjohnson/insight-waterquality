import pandas as pd
import numpy as np
import re
import sys
from math import radians, sin, cos, acos
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

def categorize(df, category_2):
    '''Categorize the samples based on whether they are safe or not'''
    cats = []
    for i in range(len(df)):
        if df.loc[i]['Ent Results'] >= 130:
            if df.loc[i]['CP Result'] >= 2:
                cats.append(1)
            else: cats.append(0)
        else:
            cats.append(0)
    df[category_2] = cats
    return df
    
def convert_month_to_continuous(df):
    '''Finds months and makes it into a continuous function'''
    #finds months
    month = []
    for x in df['Date']:
        month.append(re.findall(r"(\d+)/", x)[0])
    df['Month'] = month
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
    df = df[np.isfinite(df['Month'])]
    #turns month number into continuous
    df['sin_month'] = np.sin(2*np.pi*df['Month']/12)
    df['cos_month'] = np.cos(2*np.pi*df['Month']/12)
    return df

def dist_waikiki(df):
    '''Determines distance from waikiki'''
    wai_dist = []
    for i in range(len(df)):
        lat = df['Lat Dec Deg'][i]
        long = df['Long Dec Deg'][i]
        slat = radians(21.271483) #waikiki latitude
        slon = radians(-157.823031) #waikiki longitude
        elat = radians(lat)
        elon = radians(long)
        dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
        wai_dist.append(dist)
    df['waikiki_distance'] = wai_dist
    return df

def main():
    path = sys.argv[1]
    df = pd.read_csv(path, sep='\t', header=0)
    #categorize based on whether there are safe or hazardous levels of bacteria
    df = categorize(df, 'category_2')
    #turn months into continuous
    df = convert_month_to_continuous(df)
    #distance from waikiki
    df = dist_waikiki(df)
    
    #save final df as a tsv
    df.to_csv('WaterQualityWeatherDataforMODEL.tsv', sep='\t')
    
if __name__ == '__main__':
    main()