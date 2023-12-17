import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import math
from statistics import mean
import matplotlib.pyplot as plt
# import time

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def get_average_accuracy(testSize, model, x_train, y_train, x_test, y_test):
    trainingAccuracies = []
    testingAccuracies = []
    for i in range(testSize):
        print("Test", str(i + 1) + '/' + str(testSize) + ':')
        trainingAccuracy = model.fit(x_train, y_train,
                                     epochs=EPOCHS, verbose=1).history['accuracy'][EPOCHS - 1]
        testingAccuracy = model.evaluate(x_test,  y_test, verbose=0)[1]
        print("Training Accuracy =", trainingAccuracy,
              "\nTesting Accuracy =", testingAccuracy)
        trainingAccuracies.append(trainingAccuracy)
        testingAccuracies.append(testingAccuracy)

    trainingAvg = round(mean(trainingAccuracies)*100, 1)
    testingAvg = round(mean(testingAccuracies)*100, 1)
    overfittingIndex = testingAvg - trainingAvg
    trialNums = [x for x in range(testSize)]
    print("\n\nTesting Average Accuracy:", str(testingAvg) + '%')
    print("Training Average Accuracy:", str(trainingAvg) + '%')
    print("Overfitting Index:", round(overfittingIndex, 1))
    
    testingAvgList = []    
    tra
    for i in range(testSize):
        testingAvgList.append(testingAvg)

    plt.plot(trialNums, testingAccuracies, label="Tests")
    plt.plot(trialNums, trainingAccuracies, label="Training")
    plt.plot(trialNums, [testingAvg]*testSize, label="Testing Average")
    plt.plot(trialNums, [trainingAvg]*testSize, label="Training Average")
    plt.ylabel("Trial #")
    plt.xlabel(r"% Accuracy")
    plt.legend()
    plt.savefig("tests.jpg")
    print("Testing results saved as tests.jpg")


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    # model.evaluate(x_test,  y_test, verbose=2)
    get_average_accuracy(15, model, x_train, y_train, x_test, y_test)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def load_data(data_dir):
    """Loads image data from directory `data_dir`.

    Assumes `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.



    Args:
        data_dir (_type_): The directory of all the data.

    Returns:
        tuple: Returns tuple `(images, labels)`. `images` will be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` will
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    def incNum(file: str, first: bool):
        """Takes `file` in format `XXXXX_XXXXX.ppm` and returns
        `(XXXXX + 1)_(XXXXX + 1).ppm`

        Args:
            file (str): The file that we will be manipulating.
            first (bool): Whether or not we manipulating the first or second number.
            * If `True` we are manipulating the first number.
            * If `False` we are manipulating the second number.

        Returns:
            str: `file` with either the first or second numbers incremented.
        """
        # Determines the number we will be manipulating.
        if first:
            number = int(file[:5]) + 1
        else:
            number = int(file[6:11]) + 1

        # Makes the new string of the number with the right amount of leading zeros.
        numDigits = len(str(number))
        newNum = '0'*(5 - numDigits) + str(number)

        # Returns the filename with the new number added to the proper place.
        if first:
            return newNum + file[5:]
        return file[:6] + newNum + file[11:]

    # Creates a new sign dictionary where we will temporarily store all our values.
    signs = dict()

    # Goes through each sign and adds all its images to the dictionary formatted as specified in documentation.
    for i in range(NUM_CATEGORIES):
        # Creates places to store the path and images of each sign
        signs[i] = {
            # Records the path
            'path': os.path.join(data_dir, str(i)),
            # Empty list for the images
            'images': list()
        }

        currFile = '00000_00000.ppm'
        os.system('cls')
        printProgressBar(i, NUM_CATEGORIES,
                         prefix="Loading files...\nCurrent sign: " + str(i))

        def getCurrFilePath(file: str):
            """Gets the current file's path given `file`.

            Args:
                file (str): The name of the file who's path we are trying to find.

            Returns:
                str: File path of `file`.
            """
            currFilePath = os.path.join(signs[i]['path'], file)
            return currFilePath

        # Creates a list to contain the images of sign i.
        signs[i]['images'] = list()

        # Goes through all images of sign i and adds them to the dictionary.
        gotAllImages = False
        while not gotAllImages:
            # Reads the image and resizes it to 300x300 (width, height).
            image = cv2.imread(getCurrFilePath(currFile))
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Adds the image as a numpy ndarray to the images list of sign i.
            signs[i]['images'].append(np.asarray(image))

            # Increments to the next file. If we have ran through each file we move to the next sign.
            if not os.path.exists(getCurrFilePath(incNum(currFile, False))):
                if not os.path.exists(getCurrFilePath(incNum(currFile, True))):
                    gotAllImages = True
                else:
                    currFile = incNum(currFile, True)
            else:
                currFile = incNum(currFile, False)

    images = list()
    labels = list()

    # Goes through each image and adds it to the images list and adds the coresponding sign to the label list at the same index.
    for sign in signs:
        for image in signs[sign]['images']:
            images.append(image)
            labels.append(sign)

    # Returns the two lists as a tuple.
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Create a neural network
    model = tf.keras.models.Sequential([
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), activation="relu"
        ),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        # Adds hidden layers
        tf.keras.layers.Dense(
            math.sqrt(((IMG_WIDTH*IMG_HEIGHT*3)*NUM_CATEGORIES)), activation="relu"),
        tf.keras.layers.Dense(
            math.sqrt(((IMG_WIDTH*IMG_HEIGHT*3)*NUM_CATEGORIES)), activation="relu"),

        tf.keras.layers.Dense(
            math.sqrt(((IMG_WIDTH*IMG_HEIGHT*3)*NUM_CATEGORIES)), activation="relu"),

        tf.keras.layers.Dense(
            math.sqrt(((IMG_WIDTH*IMG_HEIGHT*3)*NUM_CATEGORIES)), activation="relu"),

        # Add an output layer with output units for all signs
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
