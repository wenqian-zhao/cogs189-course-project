import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def logitsitc_regression(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression().fit(X_train, y_train)
    prediction=logreg.predict(X_test)
    cmatrix = confusion_matrix(y_test,prediction)
    return logreg, prediction, cmatrix

def random_forest():
    ...
