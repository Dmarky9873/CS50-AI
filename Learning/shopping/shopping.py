import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """Loads shopping data from a CSV file `filename` and converts it into a list of
    evidence lists and a list of labels. 

    Returns:
        tuple: Returns a tuple (`evidence`, `labels`).

        `evidence` will be a list of lists, where each list contains the
        following values, in order:
        - index: 0 | Administrative, an integer
        - index: 1 | Administrative_Duration, a floating point number
        - index: 2 | Informational, an integer
        - index: 3 | Informational_Duration, a floating point number
        - index: 4 | ProductRelated, an integer
        - index: 5 | ProductRelated_Duration, a floating point number
        - index: 6 | BounceRates, a floating point number
        - index: 7 | ExitRates, a floating point number
        - index: 8 | PageValues, a floating point number
        - index: 9 | SpecialDay, a floating point number
        - index: 10 | Month, an index from 0 (January) to 11 (December)
        - index: 11 | OperatingSystems, an integer
        - index: 12 | Browser, an integer
        - index: 13 | Region, an integer
        - index: 14 | TrafficType, an integer
        - index: 15 | VisitorType, an integer 0 (not returning) or 1 (returning)
        - index: 16 | Weekend, an integer 0 (if false) or 1 (if true)

    `labels` will be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    # Opens the CSV file
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)

        evidences = []
        labels = []

        # Adds everything to the coresponding lists and cleans the data
        for line in list(reader)[1:]:
            evidence = []
            evidence.append(int(line[0]))
            evidence.append(float(line[1]))
            evidence.append(int(line[2]))
            evidence.append(float(line[3]))
            evidence.append(int(line[4]))
            evidence.append(float(line[5]))
            evidence.append(float(line[6]))
            evidence.append(float(line[7]))
            evidence.append(float(line[8]))
            evidence.append(float(line[9]))
            evidence.append(getMonthInt(line[10]))
            evidence.append(int(line[11]))
            evidence.append(int(line[12]))
            evidence.append(int(line[13]))
            evidence.append(int(line[14]))
            evidence.append(getVisitorTypeInt(line[15]))
            evidence.append(getBooleanInt(line[16]))

            evidences.append(evidence)

            labels.append(getBooleanInt(line[17]))

    return (evidences, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    raise NotImplementedError


def getMonthInt(month: str):
    if month == 'Jan':
        return 0
    elif month == 'Feb':
        return 1
    elif month == 'Mar':
        return 2
    elif month == 'Apr':
        return 3
    elif month == 'May':
        return 4
    elif month == 'Jun':
        return 5
    elif month == 'Jul':
        return 6
    elif month == 'Aug':
        return 7
    elif month == 'Sep':
        return 8
    elif month == 'Oct':
        return 9
    elif month == 'Nov':
        return 10
    elif month == 'Dec':
        return 11


def getVisitorTypeInt(visitorType: str):
    if visitorType == 'Returning_Visitor':
        return 1
    return 0


def getBooleanInt(boolean: str):
    if boolean == 'TRUE':
        return 1
    return 0


if __name__ == "__main__":
    main()
