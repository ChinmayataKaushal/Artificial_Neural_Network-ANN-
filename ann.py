# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
lst=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = dataset[lst]
y = dataset['Exited']

#Encoding
temp = pd.get_dummies(X['Geography'], drop_first=True)
X.drop(['Geography'], axis=1, inplace=True)
X = pd.concat([X, temp], axis=1)
temp = pd.get_dummies(X['Gender'], drop_first=True)
X.drop(['Gender'], axis=1, inplace=True)
X = pd.concat([X, temp], axis=1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

"""# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropouts
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer with dropouts
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)"""

## Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer , loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

"""classifier=KerasClassifier(build_fn= build_classifier(), batch_size = 32, epochs = 100)
accuracies= cross_val_score(estimator= classifier, X= X_train, y= y_train, cv=10, n_jobs=-1)
mean= accuracies.mean()
variance= accuracies.std()"""

## Tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


classifier=KerasClassifier(build_fn= build_classifier)
parameters = {'batch_size': [25, 32], 'epochs': [35] , 'optimizer':['adam']}

clf = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs=-1)
clf = clf.fit(X_train, y_train)

best_parameters = clf.best_params_
best_accuracy = clf.best_score_
print(best_accuracy)

# Predicting the Test set results
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

clf.best_estimator_.model.save('firstANN.h5')
