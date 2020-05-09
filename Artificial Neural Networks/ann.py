#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categories = [1])
#X = onehotencoder.fit_transform(X).toarray()

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)

X = ct.fit_transform(X)
X = X[:, 1:]

#splitting the data into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)



#importing keras
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN as sequence of layers
classifier = Sequential()

#Adding the input layer and first hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#Adding the second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#Adding the output layer
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#Compiling the classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting classifier to training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)
#create your classifier here

#fitting logistic regression to test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#predicting for a particular test case
new_prediction = classifier.predict(Sc_X.transform(np.array([[0.0, 0,600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
#Making the confusion matrix(returns correct and incorrect predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs =-1)