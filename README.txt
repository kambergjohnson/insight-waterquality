## README

Example code from my Water Quality project as part of my Insight Data Science Fellowship

Goal: The goal of this project was to created an algorithm that could predict bacterial
contamination in Hawaii Beach Water. 

(1) The data was acquired from the Hawaii State Department of Health and the Dark Sky API:
- You will need to download data from that website (http://health.hawaii.gov/cwb/). 
Otherwise sample data for Oahu samples (2004-2017) is provided (WaterQualityData_1.html). 
- You will need a acquire a dark sky API key (https://darksky.net/dev). 
I have provided some sample data (WaterQualityProcessedDailyWeatherforModel.tsv) if you want
to repeat the analysis and bypass this step. 

In order to software prerequisites (see below), you will need to pas the WaterQualityData_1.html 
(or something similar) as an argument. 

(2) To prepare for machine learning, some feature engineering was performed. Here is a sample.
Samples were categorized based off of weather or not there were contaminated based off of the
Hawaii department of Health recommendations. Month information was transformed intoa continuous
function and the flat distance from Waikiki beach was calculated. 
In order to software prerequisites (see below), you will need to pass a tsv value similar 
to what would be output from step1 as an argument. 

I have provided some sample data (WaterQualityProcessedDailyWeatherforModel.tsv) if you want
to repeat the analysis and bypass this step. 

(3) To finally predict, this code splits into training and testing, undersamples from 
the minority case and uses gradient boosted decision tree as the machine learning algorithm.
-You will need to pass WaterQualityProcessedDailyWeatherforMODEL.tsv or something similar
as an argument. 

Prerequisites: 
python 3, pandas, numpy, and sklearn. 