import numpy as np
import subprocess
import pickle
import random
import cv2
import os
import matplotlib.pyplot as plt
import subprocess
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.utils import shuffle

os.chdir('../Dataset')

###################
images=[]
labels=[]

count = 0
img_size=50
img_type=3
###################

CATEGORIES = subprocess.check_output('ls').decode()
CATEGORIES = CATEGORIES.split('\n')
CATEGORIES.remove('')

nb_classes = len(CATEGORIES)

# Iterating through each folder in dataset
for category in CATEGORIES:
	# for each img in the folder
	for img in os.listdir(category):
		# make a dir path of img
		path=os.path.join(category,img)

		# creating an img array
		img_array=cv2.imread(path)

		# checking for errors if not resize the img
		if img_array is not None:
			format_img=cv2.resize(img_array, (img_size,img_size))

			#  creating a label
			label = CATEGORIES.index(category)

		else:
			print('[-] Image not loaded')

		images.append(format_img)
		labels.append(label)


# shuffling data
X,y = shuffle(images, labels)

# formatting y data for the nn
y = np_utils.to_categorical(y, nb_classes)

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# reshaping the data
X_train = np.array(X_train).reshape(len(X_train), img_size, img_size, img_type)
X_test = np.array(X_test).reshape(len(X_test), img_size, img_size, img_type)

y_train = y_train.reshape(len(y_train), nb_classes)
y_test = y_test.reshape(len(y_test), nb_classes)

print('[+] Files succesfully reformatted')

# Changing directory
os.chdir('../Files')

file=open('X_train', 'wb')
pickle.dump(X_train, file)
file.close()

file=open('X_test', 'wb')
pickle.dump(X_test, file)
file.close()

file=open('y_train', 'wb')
pickle.dump(y_train, file)
file.close()

file=open('y_test', 'wb')
pickle.dump(y_test, file)
file.close()
