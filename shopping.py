import csv
import sys
import calendar
import statistics
import numpy as np
#from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    #model = train_model(X_train, y_train)
    #predictions = model.predict(X_test)
    #sensitivity, specificity = evaluate(y_test, predictions)
    
    # Train self implemented model and make predictions
    knn = KNN(1) #initialize objekt knn
    knn.fit(X_train,y_train) # "train" model - but no model is built or trained
    pred = knn.predict(X_test) # predict on X_test dataset
    sensitivity, specificity, f1_measure = evaluate(y_test, pred) #evaluate model with f1_measure
    
    #calculate F1 score from sklearn package for control
#    f1 = f1_score(y_test, pred)

    # Print results
    print(f"Correct: {(y_test == np.array(pred)).sum()}")
    print(f"Incorrect: {(y_test != np.array(pred)).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"F1-Measure: {100 * f1_measure:.2f}%")
#   print(f"Precision: {100 * precision:.2f}%") # for control reasons
#   print(f"F1: {100*f1:.2f}%") # for control reasons


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)

# self implented KNN-Classifier:

# first, define helper functions:

def euclidean_distance(point, X): #get euclidean distance from a point to the whole dataset
    X = np.array(X) #list to array
    return np.sqrt(np.sum((point - X)**2, axis = 1)) #result is an array with len(X)

def most_k_nearest_labels(l): #get the most occurring/ most common element from the given list 
    return statistics.mode(l)

class KNN():
    def __init__(self, k=1):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        neighbors = []
        for x in X_test: #iterate through every point in X_test
            distances = euclidean_distance(x, self.X_train) #distances is an array with len(X_train)
            y_sort = [y for _,y in sorted(zip(distances, self.y_train))] 
            #zip function takes iterables, aggregates them in a tuple
            #sorted sorts in ascending order the tuples (distances, y_train)
            #iterate through tuples and only take the labels (y) and store it in y_sort as a list
            neighbors.append(y_sort[:self.k]) 
            #take the the k-th nearest neighbors labels of x and store them in the neighbors list 
        return list(map(most_k_nearest_labels, neighbors))  #take the one label which is most common for every x and give them back as a list


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = float(0)
    specificity = float(0)
    false_positive = float(0)

    total_positive = float(0)
    total_negative = float(0)

    for label, prediction in zip(labels, predictions):

        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1 #true positives

        if label == 0:
            total_negative += 1
            if label == prediction:
                specificity += 1 #true negatives
        
        if label == 0: #calculate false_positives necessary for f1_measure
            if label != prediction:
                false_positive += 1
        
    #false_positive = total_negative - specificity #another way to compute the false_positives
    precision = sensitivity / (sensitivity + false_positive) #calculation of precision, necessary for f1_measure
    sensitivity /= total_positive
    specificity /= total_negative
    f1_measure = 2 * ((precision * sensitivity) / (precision + sensitivity))
    
    return sensitivity, specificity, f1_measure

if __name__ == "__main__":
    main()
