# %%
# import models 
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#%%
def logitsitc_regression(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression().fit(X_train, y_train)
    prediction=logreg.predict(X_test)
    cmatrix = confusion_matrix(y_test,prediction)
    return logreg, prediction, cmatrix


#%%
# place holder for Data loader 
#%%
# place holder for train test split 

#%%
regression_result = logitsitc_regression(X_train, y_train, X_test, y_test)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
print("Precision  {:.2f}".format(precision_score(y_test,prediction, average='binary')))
print("Recall {:.2f}".format(recall_score(y_test,prediction, labels=[-1,1], average='micro')))
plt.figure()
plot_confusion_matrix(logreg, X_test, y_test)
plt.show()