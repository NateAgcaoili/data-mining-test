import pandas as pd
import numpy as np
import numpy as np
import math



# Step 1- Calculate MC (Message Conveyed) for the given dataset (let us call it file TF) in reference to  the class attribute 
# MC(TF) = -p1*log2(p1) - p2*log2(p2) 
def mc(classAttribute,attribute,training_set):
    column = training_set[classAttribute]

    if attribute:
        column = training_set[training_set[classAttribute] == attribute] 

    probs = column.value_counts(normalize=True)
    messageConveyed = -1*np.sum(np.log2(probs)*probs)
    return messageConveyed

def wmc(classAttribute,attribute,training_set):
    attributeCount = len(training_set[training_set[classAttribute] == attribute].index)
    total          = len(training_set[classAttribute].index)
    print(f'{attributeCount}/{total}')
    return attributeCount/total


def ID3(root,training_set,test_set):

    highestGainAttribute = ""
    highestGainValue     = -math.inf
    for classAttribute, values in training_set.iteritems():
        messageConveyed = mc(classAttribute, attribute=None, training_set=training_set)
        print(f"{classAttribute} mc: {messageConveyed}")

        attributes = training_set[classAttribute].unique()
        print(f"{classAttribute}\n")
        weightedMessageConveyed = 0
        for attribute in attributes:
            weight = wmc(classAttribute, attribute, training_set)
            messageConveyed = mc(classAttribute, attribute, training_set)
            print(f"wmc({attribute}) = {weight}")
            weightedMessageConveyed += weight*messageConveyed

        print(f'wmc({classAttribute}) = {weightedMessageConveyed}')
        gain = messageConveyed - weightedMessageConveyed
        print(f'MC - wmc({classAttribute}) = {messageConveyed} - {weightedMessageConveyed} = {gain}')
        if gain > highestGainValue:
            highestGainAttribute = classAttribute
            highestGainValue     = gain
    
    print(f'winner is {highestGainAttribute} with gain of {highestGainValue}')
    root = highestGainAttribute
    leaves = training_set[root].unique()
    splits = {}
    for leaf in  leaves:
        print(f'leaf: {leaf} of root: {root}')
        if training_set[training_set[root] == leaf][root].is_unique:
            print(f'all of the records for leaf: {leaf} are the same. NO SPLIT')
            splits.update({leaf:"no split"})
            return
        else:
            print(f'all of the records for leaf: {leaf} are NOT the same. SPLIT')
            splits.update({leaf:"split"})

    for leaf,split in splits.items():
        if split == "split":
            print(f"setting {leaf} as the new dataset")
            if root in training_set:
                training_set = training_set[training_set[root] == leaf].drop(columns=root)
                ID3(root,training_set,test_set)

# use the training set to predict the test set.
# use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.
test_set_ID3 = pd.read_csv("Assignment_2--Test_set_for_ID3.csv")
training_set_ID3 = pd.read_csv("Assignment_2--Training_set_for_ID3.csv")

# prompt user to select either ID3 or Bayes classifier.
selection = "ID3" #= input("Please enter your selection for either ID3 or Bayes classification: ")
threshold = 0.9   #= input("Please enter a threshold: ")
g         = 0.05   #= input("Please enter a value for g: ")

root = ""
if(selection == "ID3"):
    print('***********************************')
    print('TRAINING SET')
    print(training_set_ID3)
    print('***********************************')
    
    print('***********************************')
    print('TEST SET')
    print(test_set_ID3)
    print('***********************************')
    ID3(root,training_set_ID3,test_set_ID3)