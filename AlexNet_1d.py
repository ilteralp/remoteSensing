# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:18:24 2019

@author: melike

To Do: 
    
1. Should padding be like as in https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py ?
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
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Load data
train_dataset=np.loadtxt("C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\norm_train_spectral.arff",delimiter=",")
test_dataset=np.loadtxt("C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\norm_test_spectral.arff",delimiter=",")

# Split into input (X) and output (Y) variables
train_X=train_dataset[:,0:144]
test_X=test_dataset[:,0:144]
train_Y=train_dataset[:,144]
test_Y=test_dataset[:,144]

NUM_TR_SAMPLES = train_X.shape[0]  # 2382
NUM_TEST_SAMPLES = test_X.shape[0] # 12197
NUM_FEATURES = train_X.shape[1]    # 144
NUM_CLASSES = 15

# Shuffle data
train_X, train_Y = shuffle(train_X, train_Y)

# Start class labels from 0 for one-hot encoding. Otherwise one-hot creates 16 classes. 
train_Y = train_Y - 1
test_Y = test_Y - 1

# Labels to categorical
cat_train_Y = to_categorical(train_Y, num_classes=NUM_CLASSES)
cat_test_Y = to_categorical(test_Y, num_classes=NUM_CLASSES)
print("to_categorical " + str(cat_train_Y.shape), str(cat_test_Y.shape))
print(train_X.shape, cat_train_Y.shape, test_X.shape, cat_test_Y.shape)

# Reshape model input
train_X = np.reshape(train_X, (NUM_TR_SAMPLES, NUM_FEATURES, 1))
cat_train_Y = np.reshape(cat_train_Y, (NUM_TR_SAMPLES, NUM_CLASSES))
test_X = np.reshape(test_X, (NUM_TEST_SAMPLES, NUM_FEATURES, 1))
cat_test_Y = np.reshape(cat_test_Y, (NUM_TEST_SAMPLES, NUM_CLASSES))
print(train_X.shape, cat_train_Y.shape, test_X.shape, cat_test_Y.shape)

#Instantiate an empty model
spectral_model = Sequential()

# 1st Convolutional Layer
spectral_model.add(Conv1D(filters=96, input_shape=(train_X.shape[1], train_X.shape[2]), 
                          kernel_size=12, strides=2, padding='valid', activation='relu'))

# Max Pooling
spectral_model.add(MaxPooling1D(pool_size=4, strides=3))

# Normalization
spectral_model.add(BatchNormalization())

# 2nd Convolutional Layer
spectral_model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'))

# Max Pooling
spectral_model.add(MaxPooling1D(pool_size=10, strides=1))

# Normalization
spectral_model.add(BatchNormalization())

# 3rd Convolutional Layer
spectral_model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'))

# 4th Convolutional Layer
spectral_model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'))

# 5th Convolutional Layer
spectral_model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))

# Max Pooling
spectral_model.add(MaxPooling1D(pool_size=3, strides=2))

# Dropout
spectral_model.add(Dropout(0.5))

# Passing it to a Fully Connected layer
spectral_model.add(Flatten())

# 1st Fully Connected Layer
spectral_model.add(Dense(4096, activation='relu'))

# Dropout
spectral_model.add(Dropout(0.5))

# 2nd Fully Connected Layer
spectral_model.add(Dense(4096, activation='relu'))

# Dropout
spectral_model.add(Dropout(0.5))

# Classification Layer
spectral_model.add(Dense(NUM_CLASSES, activation='softmax'))

# Print model
spectral_model.summary()

# Compile model
spectral_model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

# Fit network
epochs, verbose, batch_size = 10, 1, 32
spectral_model.fit(train_X, cat_train_Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

# Evaluate model
_, accuracy = spectral_model.evaluate(test_X, cat_test_Y, batch_size=batch_size, verbose=verbose)

# Get predictions
pred = spectral_model.predict(test_X, batch_size=batch_size, verbose=verbose)
y_classes = pred.argmax(axis=-1)
kappa = cohen_kappa_score(y_classes, test_Y)
print("kappa: " + str(kappa))