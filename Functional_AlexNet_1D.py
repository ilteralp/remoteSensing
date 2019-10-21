# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:40:49 2019

@author: melike
"""

from keras.layers import Input, Dense, Conv1D, Flatten
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers.merge import concatenate
from keras import losses
from sklearn.utils import shuffle
import numpy as np
np.random.seed(0)  # Set a random seed for reproducibility


"""
Returns spectral and spatial train and test sets
"""
BATCH_SIZE = 32
EPOCHS = 3
SHUFFLE = False
VERBOSE = 1
SEED_SPECT_VAL = 20
SEED_SPAT_VAL = 30

# Spectral data
TRAIN_SPECT_PATH = "C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\train_spectral.arff"
TEST_SPECT_PATH = "C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\test_spectral.arff"
SPECT_LABEL_INDEX = 144

# Load data
train_spect_dataset=np.loadtxt(TRAIN_SPECT_PATH,delimiter=",")
test_spect_dataset=np.loadtxt(TEST_SPECT_PATH,delimiter=",")

# Split into input (X) and output (Y) variables
train_spect_X=train_spect_dataset[:,0:SPECT_LABEL_INDEX]
test_spect_X=test_spect_dataset[:,0:SPECT_LABEL_INDEX]
train_spect_Y=train_spect_dataset[:,SPECT_LABEL_INDEX]
test_spect_Y=test_spect_dataset[:,SPECT_LABEL_INDEX]

# Get number of examples
NUM_SPECT_TR_SAMPLES = train_spect_X.shape[0]  # 2382 or 2832
NUM_SPECT_TEST_SAMPLES = test_spect_X.shape[0] # 12197 for both
NUM_SPECT_FEATURES = train_spect_X.shape[1]    # 144 or 145
NUM_CLASSES = 15

# ================================================================

# Spatial data
TRAIN_SPAT_PATH = "C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\train_spatial.arff"
TEST_SPAT_PATH = "C:\\Users\\melike\\AnacondaProjects\\Tunus\\data\\test_spatial.arff"
SPAT_LABEL_INDEX = 145

# Load data
train_spat_dataset=np.loadtxt(TRAIN_SPAT_PATH,delimiter=",")
test_spat_dataset=np.loadtxt(TEST_SPAT_PATH,delimiter=",")

# Split into input (X) and output (Y) variables
train_spat_X=train_spat_dataset[:,0:SPAT_LABEL_INDEX]
test_spat_X=test_spat_dataset[:,0:SPAT_LABEL_INDEX]
train_spat_Y=train_spat_dataset[:,SPAT_LABEL_INDEX]
test_spat_Y=test_spat_dataset[:,SPAT_LABEL_INDEX]

# Get number of examples
NUM_SPAT_TR_SAMPLES = train_spat_X.shape[0]  # 2382 or 2832
NUM_SPAT_TEST_SAMPLES = test_spat_X.shape[0] # 12197 for both
NUM_SPAT_FEATURES = train_spat_X.shape[1]    # 144 or 145

# Shuffle data
train_spect_X, train_spect_Y = shuffle(train_spect_X, train_spect_Y, random_state=SEED_SPECT_VAL)
print("spect_y" + str(train_spect_Y))

# Start class labels from 0 for one-hot encoding. Otherwise one-hot creates 16 classes. 
train_spect_Y = train_spect_Y - 1
test_spect_Y = test_spect_Y - 1

# Labels to categorical
cat_spect_train_Y = to_categorical(train_spect_Y, num_classes=NUM_CLASSES)
cat_spect_test_Y = to_categorical(test_spect_Y, num_classes=NUM_CLASSES)

# Reshape model input
train_spect_X = np.reshape(train_spect_X, (NUM_SPECT_TR_SAMPLES, NUM_SPECT_FEATURES, 1)) # This is channels-last repr. 
cat_spect_train_Y = np.reshape(cat_spect_train_Y, (NUM_SPECT_TR_SAMPLES, NUM_CLASSES))
test_spect_X = np.reshape(test_spect_X, (NUM_SPECT_TEST_SAMPLES, NUM_SPECT_FEATURES, 1))
cat_spect_test_Y = np.reshape(cat_spect_test_Y, (NUM_SPECT_TEST_SAMPLES, NUM_CLASSES))
print("SPECT")
print(train_spect_X.shape, cat_spect_train_Y.shape, test_spect_X.shape, cat_spect_test_Y.shape)

# First input model
spect_input = Input(shape=(train_spect_X.shape[1], train_spect_X.shape[2]), name='spect_input')
convl1_spect = Conv1D(filters=96, kernel_size=12, strides=2, padding='valid', activation='relu')
print("conv spect weights")
print(convl1_spect.get_weights())
spect_out = convl1_spect(spect_input)
flat1 = Flatten()(spect_out)

# Shuffle data
train_spat_X, train_spat_Y = shuffle(train_spat_X, train_spat_Y, random_state=SEED_SPAT_VAL)
print("spat_y " + str(train_spat_Y))

# Start class labels from 0 for one-hot encoding. Otherwise one-hot creates 16 classes. 
train_spat_Y = train_spat_Y - 1
test_spat_Y = test_spat_Y - 1

# Labels to categorical
cat_spat_train_Y = to_categorical(train_spat_Y, num_classes=NUM_CLASSES)
cat_spat_test_Y = to_categorical(test_spat_Y, num_classes=NUM_CLASSES)

# Reshape model input
train_spat_X = np.reshape(train_spat_X, (NUM_SPAT_TR_SAMPLES, NUM_SPAT_FEATURES, 1)) # This is channels-last repr. 
cat_spat_train_Y = np.reshape(cat_spat_train_Y, (NUM_SPAT_TR_SAMPLES, NUM_CLASSES))
test_spat_X = np.reshape(test_spat_X, (NUM_SPAT_TEST_SAMPLES, NUM_SPAT_FEATURES, 1))
cat_spat_test_Y = np.reshape(cat_spat_test_Y, (NUM_SPAT_TEST_SAMPLES, NUM_CLASSES))
print("SPAT")
print(train_spat_X.shape, cat_spat_train_Y.shape, test_spat_X.shape, cat_spat_test_Y.shape)


# Second input model
spat_input = Input(shape=(train_spat_X.shape[1], train_spat_X.shape[2]), name='spat_input')
convl1_spat = Conv1D(filters=96, kernel_size=13, strides=2, padding='valid', activation='relu')(spat_input)
print("conv spat weights")
#print(convl1_spat.get_weights())
flat2 = Flatten()(convl1_spat)

# Merge input models
merge = concatenate([flat1, flat2])

# interpretation model
hidden = Dense(10, activation='relu')(merge)
output = Dense(15, activation='sigmoid', name='main_output')(hidden)
model = Model(inputs=[spect_input, spat_input], outputs=output)
# summarize layers
#print(model.summary())
# plot graph
plot_model(model, to_file='multiple_inputs.png')

# Compile model
model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

# Fit model 
model.fit([train_spect_X, train_spat_X], [cat_spect_train_Y, cat_spat_train_Y], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=SHUFFLE)
#model.fit({'spect_input': train_spect_X, 'spat_input': spat_input}, {'main_output': output}, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=SHUFFLE)
#model.fit([train_spect_X, train_spat_X], output, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, shuffle=SHUFFLE)
