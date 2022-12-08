import pandas as pd
import numpy as np
import math

# DELETE
def calculate_metrics(training_set,test_set,classAttribute,classValue):
    print(classAttribute)
    print(classValue)
    # calculate the accuracy, error rate, sensitivity, specificity, and precision for the selected classifier in reference to the corresponding test set.
    print(training_set[training_set[classAttribute] == classValue].index)
    tp = len(training_set[training_set[classAttribute] == classValue].index)
    fp = len(test_set[test_set[classAttribute] == classValue].index)
    tn = len(training_set[training_set[classAttribute] == classValue].index) 
    fn = len(test_set[test_set[classAttribute] != classValue].index)
    p  = tp + fp
    n  = tn + fn
    print(f" \t      \t\t {classValue} \t not {classValue} \t \t TOTAL")
    print(f" \t      \t\t  \t  \t \t ")
    print(f" \t      \t {classValue} \t {tp}  \t {fp} \t {p}")
    print(f" \t not  \t {classValue} \t {fn}  \t {tn} \t {n}")
    print(f" \t total\t\t {tp+fn} \t {fn+tn}  \t {p+n} \t")

    accuracy = tp + tn /(p+n)
    error_rate = fp + fn /(p + n)
    sensitivity = tp/ p
    precision = tp/ (tp+fp)
    specificity = tn/n

    display_metrics(accuracy, error_rate, sensitivity, precision, specificity)

def get_confusion_matrix(train_set, test_set, class_attribute, class_value):
    confusion_matrix = {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0
    }
    confusion_matrix["TP"] = len(train_set[train_set[class_attribute] == class_value].index)
    confusion_matrix["FP"] = len(test_set[test_set[class_attribute] == class_value].index)
    confusion_matrix["TN"] = len(train_set[train_set[class_attribute] == class_value].index)
    confusion_matrix["FN"] = len(test_set[test_set[class_attribute] != class_value].index)

    return confusion_matrix

def get_accuracy(confusion_matrix):
    accuracy = (confusion_matrix["TP"] + confusion_matrix["TN"]) / len(test_set)

    return accuracy

def get_error_rate(confusion_matrix):
    error_rate = (confusion_matrix["FP"] + confusion_matrix["FN"]) / len(test_set)

    return error_rate

def get_sensitivity(confusion_matrix):
    sensitivity = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])
    
    return sensitivity

def get_specificity(confusion_matrix):
    specificity = confusion_matrix["TN"] / (confusion_matrix["TN"] + confusion_matrix["FP"])

    return specificity

def get_precision(confusion_matrix):
    precision = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FP"])

    return precision

def print_metrics(class_attribute, accuracy, error_rate, sensitivity, specificity, precision):
    print("{:<16} {:<10} {:<12} {:<12} {:<12} {:<12}".format('Class Attribute', 'Accuracy', 'Error Rate', 'Sensitivity', 'Specificity', 'Percision'))
    print("{:<16} {:<10} {:<12} {:<12} {:<12} {:<12}".format(class_attribute, round(accuracy, 3), round(error_rate, 3), round(sensitivity, 3), round(specificity, 3), round(precision, 3)))

# DELETE
def display_metrics(accuracy, error_rate, sensitivity, precision, specificity):
    print(f'Accuracy: {accuracy}, Error_rate:{error_rate}, Sensitivity:{sensitivity}, Precision:{precision}, specificity:{specificity}')

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
    return attributeCount/total

def ID3(root,training_set,test_set, threshold, g):

    highestGainAttribute = ""
    highestGainValue     = -math.inf
    for classAttribute, values in training_set.items():
        # Step 1- Calculate MC (Message Conveyed) for the given dataset (let us call it file TF) in reference to  the class attribute 
        # MC(TF) = -p1*log2(p1) - p2*log2(p2) 
        messageConveyed = mc(classAttribute, attribute=None, training_set=training_set)

        attributes = training_set[classAttribute].unique()
        weightedMessageConveyed = 0
        # Step 2- Calculate Gain for every attribute in the training set .
        for attribute in attributes:
            weight = wmc(classAttribute, attribute, training_set)
            messageConveyed = mc(classAttribute, attribute, training_set)
            weightedMessageConveyed += weight*messageConveyed

        gain = messageConveyed - weightedMessageConveyed
        if gain > highestGainValue:
            highestGainAttribute = classAttribute
            highestGainValue     = gain
    
    root = highestGainAttribute
    leaves = training_set[root].unique()
    splits = {}
    # K is the total number of branches in the subtree
    k = len(leaves)
    print(f"SELECTED ROOT: {root}")
    # Step 3- Examine dataset of each leaf.
    for leaf in  leaves:
        if training_set[training_set[root] == leaf]["Volume"].is_unique:
            splits.update({leaf:"no split"})
            return
        else:
            splits.update({leaf:"split"})
    classValues = None
    # Step 4- For each leaf’s dataset that is marked “Split” Do.
    for leaf,split in splits.items():
        if root in training_set:
            c1 = len(training_set[training_set[root] == leaf].index)
            F1 = len(training_set[root].index)
            # N is the number of records in the Test set that are correctly classified by the rules extracted from the tree before removal of a subtree. 
            N = len(test_set[test_set[root] == leaf].index)
            # Q is the total number of records in the test set
            Q = len(test_set[root].index)
            alpha = c1/F1
            #print(f"leaf :{leaf} -> ")
            if split == "split" and alpha < threshold:
                #calculate_metrics(training_set,test_set,root,leaf)
                confusion_matrix = get_confusion_matrix(training_set, test_set, root, leaf)
                # calculating metrics
                accuracy = get_accuracy(confusion_matrix)
                error_rate = get_error_rate(confusion_matrix)
                sensitivity = get_sensitivity(confusion_matrix)
                specificity = get_specificity(confusion_matrix)
                precision = get_precision(confusion_matrix)
                # displaying metrics
                print_metrics(root, accuracy, error_rate, sensitivity, specificity, precision)
                training_set = training_set[training_set[root] == leaf].drop(columns=root)
                test_set     = test_set[test_set[root] == leaf].drop(columns=root)
                # M is the number of records in the Test set that are correctly classified by the rules extracted from the tree after removal of the subtree.
                M = len(test_set.index)
                # If (N-M)/Q  gK, then the subtree can be removed.
                if (N-M)/Q < g * k:
                    continue
                # Go to Step 1;
                ID3(root,training_set,test_set,threshold,g)
            else:
                print("end")

    

# use the training set to predict the test set.
# use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.
train_set = pd.read_csv("Assignment_2--Training_set_for_ID3.csv")
test_set = pd.read_csv("Assignment_2--Test_set_for_ID3.csv")

def prob_continous_value(A, v, classAttribute, dataset, x):
    # calcuate the average for all values of A in dataset with class = x
    a = dataset[dataset[classAttribute] == x][A].mean()
    # calculate the standard deviation for all values A in dataset with class = x
    stdev = 1
    stdev = dataset[dataset[classAttribute] == x][A].std()
    v = dataset[A].iloc[0]
    if stdev == 0.0:
        stdev = 0.00000000000001
    return (1/(math.sqrt(2*math.pi)*stdev))*math.exp(-((v-a)*(v-a))/(2*stdev*stdev))

def main(t1, g):
    root = ""
    ID3(root, train_set, test_set, t1, g)