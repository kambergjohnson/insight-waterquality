# insight-waterquality
Gradient Boosted Machine Learning for Vacation Contamination (insight project)

What does it do?
At a high level, this code takes a pandas dataframe similar to provided (WaterQualityProcessedDailyWeatherforMODEL.tsv).
It splits the code into training and testing, undersamples from the minority case and 
uses a gradient boosted decision tree to predict categories. 

Prerequisites: 
python 3, pandas, numpy, and sklearn. 

To run, provide the path to the dataframe as the first argument. 
