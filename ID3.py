from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import ceil, floor, log2, pi
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import statistics
import math

def calculate_metrics(tp, tn, fn, p, n, fp):
    # calculate the accuracy, error rate, sensitivity, specificity, and precision for the selected classifier in reference to the corresponding test set.
    accuracy = tp + tn /(p+n)
    error_rate = fp + fn /(p + n)
    sensitivity = tp/ p
    precision = tp/ (tp+fp)
    specificity = tn/n

    display_metrics(accuracy, error_rate, sensitivity, precision, specificity)

def display_metrics(accuracy, error_rate, sensitivity, precision, specificity):
    print(f'Accuracy: {accuracy}, Error_rate:{error_rate}, Sensitivity:{sensitivity}, Precision:{precision}, specificity:{specificity}')

def mc(columnName,training_set):
    column = training_set[columnName]
    probs = column.value_counts(normalize=True)
    messageConveyed = -1*np.sum(np.log2(probs)*probs)
    # print(f'mc {messageConveyed}')
    return messageConveyed

def isUnique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def ID3(root,training_set,test_set):

    mcDictionary = {}
    # Step 1- Calculate MC (Message Conveyed) for the given data set in reference to the class attribute
    print(f'Step 1- Calculate MC (Message Conveyed) for the given data set in reference to the class attribute')
    # MC = -p1*log2(p1) - p2*log2(p2)
    # For n classes MC = -p1log2(p1) - p2*log2(p2)-...-pn*log2(pn)

    # For each column calculate the gain.
    numberOfColumns = 0
    print('***********************************')
    print('For each column calculate the gain.')
    for (columnName, columnData) in training_set.iteritems():
        messageConveyed = mc(columnName,training_set)
        mcDictionary.update({columnName:round(messageConveyed)})
        numberOfColumns+=1
    print('***********************************')
    print(f'numberOfColumns {numberOfColumns}')
    print(f'mcDictionary {mcDictionary}')


    # The column with the highest gain is the root.
    print(f'The column with the highest gain is the root.')
    values = mcDictionary.values()
    max_value = max(values)
    print(f'The max value is {max_value}')
    # print(f'The max value, {max_value}, is associated with column {columnWithMaximumInformationGain}')
    val_list = list(values)
    columnWithMaximumInformationGain = list(mcDictionary.keys())[list(mcDictionary.values()).index(max_value)]
    print(f'The max value, {max_value}, is associated with column {columnWithMaximumInformationGain}')

    # select the max value from the gain array
    # this is the new root
    root =  columnWithMaximumInformationGain
    print(f'root is {root}')
    print("******************************************")
    print("**************   ROOT   ******************")
    print(f"TF is {root}**********************")
    print("******************************************")
    print(f'isUnique = {isUnique(training_set[root])}')
    if(isUnique(training_set[root])):
        return   
    
    # Step 2 - Repeat for every attribute
    print(f'Step 2 - Repeat for every attribute')
    # Loop 1
    attribute = ""
    maximum       = 0 
    for (F, columnData) in training_set.iteritems():
        print(f'processing attribute {F}')
        # Loop 2
        total = 0
        uniques = training_set[F].unique()
        for k in uniques:
            print(f'processing branch {k} for {F}')
            # Calculate MC for column
            messageConveyed = mc(F,training_set)

            # Calculate the weight for F
            F_D    = training_set[F].count()
            TF_D   = training_set[root].count()

            weight = F_D/TF_D
            total = weight*messageConveyed
        gain = abs(mc(root,training_set) - total)
        if(gain > maximum):
            attribute = F
            maximum   = gain 
        print(f"gain: {gain} for {F}")
    
    print(f'attribute {attribute} has the max gain of {gain}')
    print(f'removing {attribute}')
    root = attribute
    print(f'new root {root} has branches {training_set[root].unique()}')
    print(f'root is {root}')
    print("******************************************")
    print("**************   ROOT   ******************")
    print(f"TF is {root}**********************")
    print("******************************************")
    unique_values = training_set[root].unique()
    datasets = []
    for unique_value in unique_values:
        print(f'processing for file : {unique_value} ')
        df_1 = training_set[training_set[attribute] > unique_value]
        datasets.append(df_1)

    # del training_set[attribute]
    
    # Step 3 - Examine dataset of each leaf
    print(f'Step 3 - Examine dataset of each leaf')
    print(f'number of datasets {len(datasets)}')
    print("*****************")
    print("printing datasets")
    print("*****************")
    dataframes = {}
    for df in datasets:
        print(f'Step 4 - for {attribute} dataset check is marked "split"')
        if(df[attribute].is_unique):
            print(f'all values are the same no split')
        else:
            print(f'values are not unique perform split')
            dataframes.update({attribute:df})
    print(dataframes)
    
    for attribute in dataframes:
        print(f"processing {attribute}")
        ID3(root,dataframes[attribute],test_set)
        
        
    print("*****************")

# prompt user to select either ID3 or Bayes classifier.
selection = "ID3" #= input("Please enter your selection for either ID3 or Bayes classification: ")
threshold = 0.9     #= input("Please enter a threshold: ")
g         = 0.05    #= input("Please enter a value for g: ")

root = ""
if(selection == "ID3"):
    # use the training set to predict the test set.
    # use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.
    test_set = pd.read_csv("Assignment_2--Test_set_for_ID3.csv")
    training_set = pd.read_csv("Assignment_2--Training_set_for_ID3.csv")
    ID3(root,training_set,test_set)
