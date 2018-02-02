import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def train_test_split(df, percenttrain):
    '''Takes a dataframe and randomly splits into train and test sets'''
    numitems = len(df)
    numtrain = int(numitems*percenttrain)
    numtest = numitems - numtrain

    #scramble up the indicies in the df so that it is random
    df = df.sample(frac=1).reset_index(drop=True)
    traindf = df[0:numtrain]
    testdf = df[numtrain:]
    traindf = traindf.reset_index(drop=True)
    testdf = testdf.reset_index(drop=True)
    return traindf, testdf

def undersample(traindf, cat):
    '''Takes a training dataframe and creates a new training dataframe with balanced categories'''
    #identify minority number, indicies,and create sample
    minorityN = len(traindf[traindf[cat] == 1]) # get the total count of contaminated group
    minority_indices = traindf[traindf[cat] == 1].index
    minority_sample = traindf.loc[minority_indices]
    
    #identify majority indicies, sample same number as minority sample
    majority_indices = traindf[traindf[cat] == 0].index
    # randomly sample from majority group
    random_indices = np.random.choice(majority_indices, minorityN, replace=False) 
    majority_sample = traindf.loc[random_indices]
    
    #put the samples together to make a new dataframe
    samples = [minority_sample, majority_sample]
    traindf_balanced = pd.concat(samples)
    return traindf_balanced

def cross_validation_for_trees(maxtrees, leafs, traindf, features, category):
    '''Determines optimal amount of trees and leafs'''
    trees = range(1,maxtrees)
    leafs = leafs
    tree_scores = []
    for t in trees:
        clf = RandomForestClassifier(n_estimators=t, min_samples_leaf=leafs)
        scores = cross_val_score(clf, traindf[features],traindf[category], cv=5, scoring='recall')
        tree_scores.append(scores.mean())
    return tree_scores

def gradientboostedtree(trees, leafs, traindf_balanced, testdf, features):
    '''Machine Learning Classification with a Gradient Boosted Tree'''
    clf = GradientBoostingClassifier(n_estimators=trees, min_samples_leaf=leafs)
    rtree = clf.fit(traindf_balanced[features],traindf_balanced['category_2'])
    predictions = clf.predict(testdf[features])
    return predictions, clf
    
def calculate_accuracy(traindf_balanced, testdf, cat, predictions):
    numtrain = len(traindf_balanced)
    numtest = len(testdf)
    numyes = len(testdf[testdf[cat] == 1])
    numno = len(testdf[testdf[cat] == 0])
    correct = 0
    misclassification = 0
    true_positive = 0 #aka sensitivitity or recall
    false_positive = 0 #type 1 error
    specificity = 0 #when its no, how often does it say no, or true negatives
    false_negative = 0 #type 2 error
    predicted_yes = 0

    for i in range(numtest):
        if predictions[i] == testdf.loc[i][cat]: correct +=1
        if predictions[i] != testdf.loc[i][cat]: misclassification +=1
        if predictions[i] == 1 and testdf.loc[i][cat] ==1: true_positive +=1
        if predictions[i] == 1 and testdf.loc[i][cat] ==0: false_positive +=1 
        if predictions[i] == 0 and testdf.loc[i][cat] ==0: specificity +=1
        if predictions[i] == 0 and testdf.loc[i][cat] ==1: false_negative +=1
        if predictions[i] == 1: predicted_yes +=1
    print('Accuracy:', float(correct)/float(numtest), numyes, numno)
    print('Misclassification/Error Rate', float(misclassification)/float(numtest))
    print('True Positive Rate/Recall...most important', float(true_positive)/float(numyes), true_positive, numyes)
    print('False Positive Rate', float(false_positive)/float(numno))
    print('Specificity/True Negatives', float(specificity)/float(numno))
    print('Precision/correct when predict yes', float(true_positive)/float(predicted_yes), true_positive, predicted_yes)
    print('Prevalence', float(numyes)/float(numtest))

def feature_importance(clf):
    importance = clf.feature_importances_
    return importance

def main():
    #open up dataframe
    path = sys.argv[1]
    df = pd.read_csv(path, sep='\t', header=0)
    
    #split training and test
    traindf = train_test_split(df, 0.90)[0]
    testdf = train_test_split(df, 0.90)[1]

    traindf_balanced = undersample(traindf, 'category_2')
    
    #features for machine learning
    features = ['sin_month', 'cos_month','Salinity', 'Turbidity', 'precipIntensity', 'humidity', 'precipIntensityMax','precipIntensityMax_1', 'precipIntensityMax_2', 'precipIntensityMax_3']
    
    tree_scores = cross_validation_for_trees(1000, 10, traindf, features, 'category_2')
    
    #machine learning
    predictions, clf = gradientboostedtree(1000, 10, traindf_balanced, testdf, features)
    
    #calculate accuracy
    calculate_accuracy(traindf_balanced, testdf, 'category_2', predictions)

    list_feature_importance = feature_importance(clf)
    
if __name__ == '__main__':
    main()