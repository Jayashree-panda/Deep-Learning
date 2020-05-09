#Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step1 - convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step2 - pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step3 - Flattening
classifier.add(Flatten())

#step4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))