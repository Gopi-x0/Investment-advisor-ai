'''
Demo code for CPSC 383
Works with bakechefData.csv dataset
For use in tutorials

Author: Janet Leahy
Jun. 3, 2025
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

print("TensorFlow version:", tf.__version__)
print("\n\n Messages complete! Let's start... \n\n")


##############################################

FEATURE_VEC_SIZE = 0

# given a list with the features and label for a sample (i.e. a list of strings representing a row of the csv),
# this function converts it to a numeric feature vector (row vector) and an integer label
# returns the tuple (feature_vector, label)
def getDataFromSample(sample):

    # WS10: FILL THIS IN!

    # feature vector needs to be a (1,FEATURE_VEC_SIZE) numpy array
    # label just needs to be an integer
    
    # you'll have to remove this later, this is just so the code runs for now and you can test up to this point
    quit()
    
    # return as a tuple:   
    #return (feature_vector, label)



##############################################

# reads number of samples, features, and their labels from the given file
# and converts lists of features to feature vectors
# returns (features_array, labels_array) as a tuple
# where features_array is a (n, FEATURE_VEC_SIZE) numpy array and labels_array is a (n,) numpy array

def readData(filename):
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row
        
        n = 0
        features = []
        labels = []
        
        for row in reader:
            # convert row to feature vector + numeric label
            feature_vector, label = getDataFromSample(row)
            
            # append to arrays of all data points and labels
            features.append(feature_vector)
            labels.append(label)
            
            # increment number of data points read
            n = n + 1

    print(f"Number of data points read: {n}")
    
    # convert to numpy arrays of the appropriate dimensions
    features_array = np.concatenate(features, axis = 0)
    labels_array = np.array(labels)
    
    return (n, features_array, labels_array)


################################################

# reads the data from the bakechefData.csv file,
# passes the rows to be encoded into feature vectors,
# and divides the data into training and testing sets
def prepData():

    # here, n = number of samples in CSV (should be 6000)
    # features and labels are (n, FEATURE_VEC_SIZE) and (FEATURE_VEC_SIZE,) arrays
    # containing the feature vectors and labels for the entire dataset
    n, features, labels = readData('bakechefData.csv')

    # WS10: FILL THIS IN!
    
    # split features into trainingFeatures and testingFeatures

    # split labels into trainingLabels and testingLabels


    # For now, we just define these values to be the entire dataset so the code runs
    # but you'll have to remove this later
    trainingFeatures = features
    trainingLabels = labels
    testingFeatures = features
    trainingLabels = labels

    # package both the training data/labels and the testing data/labels as tuples before returning
    trainingData = (trainingFeatures, trainingLabels)
    testingData = (testingFeatures, testingLabels)
    return (trainingData, testingData)


###################################################

# this line lets you test your getDataFromSample function on an input of your choice
# just be careful that your example matches the formatting on the spreadsheet
# you will test your neural net on this input later

my_vector, some_label = getDataFromSample(['beef', 'Y', 'brown', "['cucumber', 'carrots', 'cilantro', 'lettuce', 'jalapeno']", '1'])
print(f"Encoded feature vector: {my_vector}")
print(f"Encoded label: {some_label}")


# read and prepare data from spreadsheet
(x_train, y_train),(x_test, y_test) = prepData()

'''
# check that the data shapes are what you expected
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
'''

model = tf.keras.models.Sequential([
    tf.keras.Input(shape = (FEATURE_VEC_SIZE,)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])
#print(model.summary())


loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
# from_logits = True because output values are not probabilities (i.e. softmax is not built in to the net) 


model.compile(optimizer = tf.keras.optimizers.Adam(),
                loss=loss_fn,
                metrics = ['accuracy']) 


# train and then test the model
model.fit(x_train, y_train, epochs = 10)
model.evaluate(x_test, y_test, verbose = 2)


# show predictions on input of our choice after training
output = model(my_vector).numpy()
print(f"Raw output: {output}")
prob = tf.nn.sigmoid(output).numpy()
print(f"Probability after sigmoid: {prob}")

