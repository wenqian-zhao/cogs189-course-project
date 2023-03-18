# K-Nearest Neighbors (K-NN)

# Importing the libraries
from src import data_import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('EEG_Eye_State_Arff.csv')
X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)