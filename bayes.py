from csv import reader
from math import sqrt
from math import exp
from math import pi

# CSV loader
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
		#print('[%s] => %d' % (value, i))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Predicting the classes of each row of the test set
def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        #print(result)
        predictions.append(result)
    return predictions

# Converting volume_index_dictionary to key volume values
def convert_predictions(predictions, volume_index_dictionary):
    converted_predictions = []
    for i in range(len(predictions)):
        converted_prediction = list(volume_index_dictionary.keys())[list(volume_index_dictionary.values()).index(predictions[i])]
        converted_predictions.append(converted_prediction)
    return converted_predictions

# Getting confusin matrix per volume entry (1 - 6)
def get_confusion_matrices(test_set, predictions):
    confusion_matrices = {
        1: {
           "TP": 0,
           "FP": 0,
           "TN": 0,
           "FN": 0
        },
        2: {
           "TP": 0,
           "FP": 0,
           "TN": 0,
           "FN": 0
        },
        3: {
           "TP": 0,
           "FP": 0,
           "TN": 0,
           "FN": 0
        },
        4: {
           "TP": 0,
           "FP": 0,
           "TN": 0,
           "FN": 0
        },
        5: {
           "TP": 0,
           "FP": 0,
           "TN": 0,
           "FN": 0
        },
        6: {
           "TP": 0,
           "FP": 0,
           "TN": 0,
           "FN": 0
        }
    }

    # For each volume
    for j in range(6):
        current_volume = j + 1
        for i in range(len(test_set)):
            volume_check = int(test_set[i][-1])
            prediction_check = int(predictions[i])
            # if the test set's row prediction volume is equal to the current_volume
            if  volume_check == current_volume:
                # if the prediction is correct
                if volume_check == prediction_check:
                    # increment current_volume's true positive in confusion matrix
                    confusion_matrices[current_volume]["TP"] += 1
                # prediction is incorrect, should have predicted 1
                else:
                    # increment volume 1's false negative
                    confusion_matrices[current_volume]["FN"] += 1
            # volume is something other than the current_volume
            else:
                # if prediction is correct and not the current_volume
                if  volume_check == prediction_check:
                    # increment current_volume's true negative in confusion matrix
                    confusion_matrices[current_volume]["TN"] += 1
                # prediction was not equal to 
                else: 
                    confusion_matrices[current_volume]["FP"] += 1
    
    return confusion_matrices

def get_accuracies(test_set, confusion_matrices):
    accuracies = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    for i in range(6):
        current_confusion_matrix = confusion_matrices[i + 1]
        accuracy = (current_confusion_matrix["TP"] + current_confusion_matrix["TN"]) / len(test_set)
        accuracies[i + 1] = accuracy
    
    return accuracies

def get_error_rates(test_set, confusion_matrices):
    error_rates = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }

    for i in range(6):
        current_confusion_matrix = confusion_matrices[i + 1]
        error_rate = (current_confusion_matrix["FP"] + current_confusion_matrix["FN"]) / len(test_set)
        error_rates[i + 1] = error_rate
    
    return error_rates

def get_sensitivities(confusion_matrices):
    sensitivities = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }

    for i in range(6):
        current_confusion_matrix = confusion_matrices[i + 1]
        sensitivity = current_confusion_matrix["TP"] / (current_confusion_matrix["TP"] + current_confusion_matrix["FN"])
        sensitivities[i + 1] = sensitivity
    
    return sensitivities

def get_specificities(confusion_matrices):
    specificities = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }

    for i in range(6):
        current_confusion_matrix = confusion_matrices[i + 1]
        specificity = current_confusion_matrix["TN"] / (current_confusion_matrix["TN"] + current_confusion_matrix["FP"])
        specificities[i + 1] = specificity
    
    return specificities

def get_precisions(confusion_matrices):
    precisions = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }

    for i in range(6):
        current_confusion_matrix = confusion_matrices[i + 1]
        precision = current_confusion_matrix["TP"] / (current_confusion_matrix["TP"] + current_confusion_matrix["FP"])
        precisions[i + 1] = precision
    
    return precisions

def print_metrics(accuracies, error_rates, sensitivities, specificities, precisions):
    print("{:<8} {:<10} {:<12} {:<12} {:<12} {:<12}".format('Volume', 'Accuracy', 'Error Rate', 'Sensitivity', 'Specificity', 'Percision'))
    for i in range(6):
        volume = i + 1
        print("{:<8} {:<10} {:<12} {:<12} {:<12} {:<12}".format(volume, round(accuracies[volume], 3), round(error_rates[volume], 3), round(sensitivities[volume], 3), round(specificities[volume], 3), round(precisions[volume], 3)))


def main():
    # Make a prediction with Naive Bayes
    filename = 'Assignment_2--Training_set_for_Bayes.csv'
    test_file = 'Assignment_2--Test_set_for_Bayes.csv'
    train_set = load_csv(filename)
    test_set = load_csv(test_file)
    for i in range(len(train_set[0])-1):
        str_column_to_float(train_set, i)

    for i in range(len(test_set[0])-1):
        str_column_to_float(test_set, i)
    # convert class column to integers
    volume_index_dictionary = str_column_to_int(train_set, len(train_set[0])-1)
    # fit model
    model = summarize_by_class(train_set)
    predictions = get_predictions(model, test_set)
    converted_predictions = convert_predictions(predictions, volume_index_dictionary)
    confusion_matrices = get_confusion_matrices(test_set, converted_predictions)
    # calculating metrics
    accuracies = get_accuracies(test_set, confusion_matrices)
    error_rates = get_error_rates(test_set, confusion_matrices)
    sensitivities = get_sensitivities(confusion_matrices)
    specificities = get_specificities(confusion_matrices)
    precisions = get_precisions(confusion_matrices)
    # displaying metrics
    print_metrics(accuracies, error_rates, sensitivities, specificities, precisions)