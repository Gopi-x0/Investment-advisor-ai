# ------------------------------------------------------------
# penguins.py
# Name: Puneet Dhawan (30205034)
# Date: June 18, 2025
# Course: CPSC 383
# Semester: Spring 2025
# 
# the following code is from bakechef_net_starter and was adjusted to suit this task
# ------------------------------------------------------------

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

print("TensorFlow version:", tf.__version__)
print("\n\n Messages complete! Let's start... \n\n")

##############################################

FEATURE_VEC_SIZE = 11 #Changes for each feature used as data. 

# given a list with the features and label for a sample (i.e. a list of strings representing a row of the csv),
# this function converts it to a numeric feature vector (row vector) and an integer label
# returns the tuple (feature_vector, label)
def getDataFromSample(sample):

    # Holds the Species as a constant
    # label just needs to be an integer
    penguin_species = {'Adelie': [1, 0, 0], 'Chinstrap': [0, 1, 0], 'Gentoo': [0, 0, 1]}
    label = np.array(penguin_species[sample[1]])

    penguin_islands = {'Torgersen': [1, 0, 0], 'Biscoe': [0, 1, 0], 'Dream': [0, 0, 1]}
    penguin_genders = {'NA': [1, 0, 0], 'male': [0, 1, 0], 'female': [0, 0, 1]}
    
    # feature vector needs to be a (1,FEATURE_VEC_SIZE) numpy array
    culmen_length = float(sample[3]) 
    culmen_depth = float(sample[4])
    flipper_length = float(sample[5])
    body_mass = float(sample[6])
    island = penguin_islands[sample[2]]
    gender = penguin_genders[sample[7]]
    year = int(sample[8]) - 2007 
    feature_vector = np.array([[culmen_length, culmen_depth, flipper_length, body_mass, year] + island + gender])
    
    # return as a tuple:   
    return (feature_vector, label)

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
    labels_array = np.vstack(labels)
    
    return (n, features_array, labels_array)

################################################

# reads the data from the penguins.csv file,
# passes the rows to be encoded into feature vectors,
# and divides the data into training and testing sets
def prepData():

    # here, n = number of samples in CSV (should be 6000)
    # features and labels are (n, FEATURE_VEC_SIZE) and (FEATURE_VEC_SIZE,) arrays
    # containing the feature vectors and labels for the entire dataset
    n, features, labels = readData('C:/Users/dhawa/OneDrive/Desktop/UNI/CPSC 383/penguins/penguins.csv')

    #indices are randomly shuffled, then those indices are reallocated to the features and labels
    indices = np.arange(n)
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    # split features into trainingFeatures and testingFeatures
    # split labels into trainingLabels and testingLabels
    # Will get the sample size and create the 5:1 training_testing_ratio for testing and training, data is then split with that value
    training_testing_ratio = int(n * (5 / 6))
    trainingFeatures = features[:training_testing_ratio]
    trainingLabels = labels[:training_testing_ratio]
    testingFeatures = features[training_testing_ratio:]
    testingLabels = labels[training_testing_ratio:]

    # package both the training data/labels and the testing data/labels as tuples before returning
    trainingData = (trainingFeatures, trainingLabels)
    testingData = (testingFeatures, testingLabels)
    return (trainingData, testingData)


###################################################

# read and prepare data from spreadsheet
(x_train, y_train),(x_test, y_test) = prepData()

# check that the data shapes are what you expected
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
#x_test = (x_test - mean) / std  

model = tf.keras.models.Sequential([
    tf.keras.Input(shape = (FEATURE_VEC_SIZE,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3)                            #output only 3 for this case, the 3 species
])
print(model.summary())


loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# SparseCategoricalCrossentropy as we are working with 0,1,2 and not binary. Catergorical will complex the task
# from_logits = True because output values are not probabilities (i.e. softmax is not built in to the net) 

model.compile(optimizer = tf.keras.optimizers.Adam(),
                loss=loss_fn,
                metrics = ['accuracy']) 

# train and then test the model
model.fit(x_train, y_train, epochs = 8)
model.evaluate(x_test, y_test, verbose = 2)


