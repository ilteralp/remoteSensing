# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:40:44 2019

@author: melike
"""

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras.applications import VGG16
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Conv1D, MaxPooling1D, BatchNormalization, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten, Dropout, Reshape
from keras import losses
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
train_dataset=np.loadtxt("C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\train_spectral.arff",delimiter=",")
test_dataset=np.loadtxt("C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\test_spectral.arff",delimiter=",")

# Split into input (X) and output (Y) variables
train_X=train_dataset[:,0:144]
test_X=test_dataset[:,0:144]
train_Y=train_dataset[:,144]
test_Y=test_dataset[:,144]

NUM_TR_SAMPLES = train_X.shape[0]  # 2382
NUM_TEST_SAMPLES = test_X.shape[0] # 12197
NUM_FEATURES = train_X.shape[1]    # 144
NUM_CLASSES = 15
print("NUM_TR_SAMPLES " + str(NUM_TR_SAMPLES) + "\nNUM_FEATURES " + str(NUM_FEATURES) + "\nNUM_TEST_SAMPLES " + str(NUM_TEST_SAMPLES))

print(train_X[0])
# Normalize data
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)
print(train_X[0])

# Shuffle data
train_X, train_Y = shuffle(train_X, train_Y)

# Start class labels from 0 for one-hot encoding. Otherwise one-hot creates 16 classes. 
train_Y = train_Y - 1
test_Y = test_Y - 1

# Reshape model input
train_Y = to_categorical(train_Y, num_classes=NUM_CLASSES)
test_Y = to_categorical(test_Y, num_classes=NUM_CLASSES)
print("to_categorical " + str(train_Y.shape), str(test_Y.shape))
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
"""
train_X = train_X.reshape((NUM_TR_SAMPLES, NUM_FEATURES, 1))
train_Y = train_Y.reshape((NUM_TR_SAMPLES, NUM_CLASSES))
test_X = test_X.reshape((NUM_TEST_SAMPLES, NUM_FEATURES, 1))
test_Y = test_Y.reshape((NUM_TEST_SAMPLES, NUM_CLASSES))
"""

train_X = np.reshape(train_X, (NUM_TR_SAMPLES, NUM_FEATURES, 1))
train_Y = np.reshape(train_Y, (NUM_TR_SAMPLES, NUM_CLASSES))
test_X = np.reshape(test_X, (NUM_TEST_SAMPLES, NUM_FEATURES, 1))
test_Y = np.reshape(test_Y, (NUM_TEST_SAMPLES, NUM_CLASSES))
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


#Instantiate an empty model
spectral_model = Sequential()
spectral_model.add(Conv1D(filters=64, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_size=14, strides=2, padding='valid'))

#spectral_model.add(BatchNormalization())
spectral_model.add(Activation('sigmoid'))

# Max Pooling
spectral_model.add(MaxPooling1D(pool_size=2, strides=2))

# 2nd Convolutional Layer
spectral_model.add(Conv1D(filters=128, kernel_size=5, strides=2, padding='valid'))

# 3rd Convolutional Layer
spectral_model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='valid'))

# Passing it to a Fully Connected layer
spectral_model.add(Flatten())

# 1st Fully Connected Layer
spectral_model.add(Dense(1024))
spectral_model.add(Activation("sigmoid"))

# Classification Layer
spectral_model.add(Dense(NUM_CLASSES, activation='softmax'))

# Print model
spectral_model.summary()

# Compile model
#spectral_model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
sgd = SGD(lr=0.1, decay=1e-6, momentum=1.9)
spectral_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

# Fit network
epochs, verbose, batch_size = 100, 1, 32
spectral_model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

# Evaluate model
_, accuracy = spectral_model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=verbose)
