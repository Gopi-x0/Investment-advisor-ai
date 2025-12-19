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

FEATURE_VEC_SIZE = 11 #constant for the number of feature being utilized but can change for each feature used as data. 

# given a list with the features and label for a sample (i.e. a list of strings representing a row of the csv),
# this function converts it to a numeric feature vector (row vector) and an integer label
# returns the tuple (feature_vector, label)
def getDataFromSample(sample):

    #Penguin species is encoded with one hot encoding, 
    penguin_species = {'Adelie': [1, 0, 0], 'Chinstrap': [0, 1, 0], 'Gentoo': [0, 0, 1]}
    label = np.array(penguin_species[sample[1]])

    #Penguin gender and island is encoded with one hot encoding 
    penguin_islands = {'Torgersen': [1, 0, 0], 'Biscoe': [0, 1, 0], 'Dream': [0, 0, 1]}
    penguin_genders = {'NA': [1, 0, 0], 'male': [0, 1, 0], 'female': [0, 0, 1]}
    
    # feature vector needs to be a (1,FEATURE_VEC_SIZE) numpy array
    #Gets the island based on the given encoding 
    island = penguin_islands[sample[2]]
    #Gets all numerical data
    culmen_length = float(sample[3])  
    culmen_depth = float(sample[4])
    flipper_length = float(sample[5])
    body_mass = float(sample[6])
    #Gets the gender based on the given encoding 
    gender = penguin_genders[sample[7]]
    #Gets the year
    year = int(sample[8]) 
    #an array is created with all the extracted data
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

    # here, n = number of samples in CSV (should be 342)
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
    # For this case, outputing 285 (343*5/6)
    training_testing_ratio = int(n * (5 / 6))
    #trains based on 285 data points
    trainingFeatures = features[:training_testing_ratio]
    trainingLabels = labels[:training_testing_ratio]
    #tests based on 57 data points
    testingFeatures = features[training_testing_ratio:] 
    testingLabels = labels[training_testing_ratio:]

    # package both the training data/labels and the testing data/labels as tuples before returning
    trainingData = (trainingFeatures, trainingLabels)
    testingData = (testingFeatures, testingLabels)
    return (trainingData, testingData)


###################################################

# read and prepare data from spreadsheet
(x_train, y_train),(x_test, y_test) = prepData()

#Will go through all of x train and its each of its column which includes its features. by doing this it will find a Mean and STD.
#After Mean and Standard Deviation is found, its utilized to get a standardized version through the Z-score formula. Which helps in stopping any data biases 
mean = np.mean(x_train, axis=0)                     
std = np.std(x_train, axis=0)
standardized_x_train = (x_train - mean) / std
standardized_x_test = (x_test - mean) / std  

model = tf.keras.models.Sequential([
    tf.keras.Input(shape = (FEATURE_VEC_SIZE,)),        #Input layer, takes in a feature vector based on number of features(11)
    tf.keras.layers.Dense(128, activation='relu'),      #Hidden layer 1, Works with 128 Neurons and the activation function ReLU 
    tf.keras.layers.Dense(64, activation='relu'),       #Hidden layer 2, Works with 64 Neurons and the activation function ReLU 
    tf.keras.layers.Dense(32, activation='relu'),       #Hidden layer 3, Works with 32 Neurons and the activation function ReLU 
    tf.keras.layers.Dense(16, activation='relu'),       #Hidden layer 4, Works with 16 Neurons and the activation function ReLU 
    tf.keras.layers.Dense(3)                            #output only 3 for this case, the 3 species
])
print(model.summary())

# CategoricalCrossentropy as we are working with one hot encoding and not binary. 
# from_logits = True because output values are not probabilities (i.e. softmax is not built in to the net) 
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

#Learning rate for adam is set to default as I found no difference between using it and not. 
model.compile(optimizer = tf.keras.optimizers.Adam(),
                loss=loss_fn,
                metrics = ['accuracy']) 

#train and then test the model
#Epochs set to 12, but can be increased as increasing the epoch will only further allow the neural network to learn and train itself to find a connection
model.fit(standardized_x_train, y_train, epochs = 12)
model.evaluate(standardized_x_test, y_test, verbose = 2)


