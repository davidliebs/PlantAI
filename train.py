import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
import pickle
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler

# define X_train and X_test into variable
os.chdir('../Files')

##################################
# Defining variables
img_size=50
img_type=3
conv_size = 3
pool_matrix = 2

nb_classes=10

################################

# Loading in files
f = open('X_train', 'rb')
X_train=pickle.load(f)
f.close()

f = open('X_test', 'rb')
X_test = pickle.load(f)
f.close()

f = open('y_train', 'rb')
y_train = pickle.load(f)
f.close()

f = open('y_test', 'rb')
y_test = pickle.load(f)
f.close()

###########################################################################
# formatting data to compute class weights
y_train = y_train.argmax(1)

# Setting the weights
# class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# formatting data to train on CNN
y_train = np_utils.to_categorical(y_train, nb_classes)


#########################################################################

# Creating the neural network
model=Sequential()
# 5,7,12,17

model.add(Conv2D(5, (conv_size,conv_size), input_shape=(img_size, img_size, img_type), padding='same'))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(pool_matrix, pool_matrix)))

model.add(Conv2D(7, (conv_size,conv_size), padding='same'))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(pool_matrix, pool_matrix)))

model.add(Conv2D(12, (conv_size,conv_size), padding='same'))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(pool_matrix, pool_matrix)))

model.add(Conv2D(17, (conv_size,conv_size), padding='same'))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(pool_matrix, pool_matrix)))

model.add(Flatten())

model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=7, class_weight='balanced', validation_data=(X_test, y_test))

tf.keras.models.save_model(model, 'PlantAI.model', overwrite=True, include_optimizer=True)
