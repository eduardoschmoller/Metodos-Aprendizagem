# -*- coding: utf-8 -*-
"""
Created on Wed May 16 19:54:46 2018

@author: eduar
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score 

X_train = pd.read_csv("X_train.txt", sep = " ", header=None)
Y_train = pd.read_csv("y_train.txt", sep = " ", header=None)
X_test = pd.read_csv("X_test.txt", sep = " ", header=None)
Y_test = pd.read_csv("y_test.txt", sep = " ", header=None)


X = X_train.append(X_test)
Y = np.ravel(Y_train.append(Y_test))

# Splitting data into 70% training and 30% test data:
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3)

from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=100, eta0=0.1, verbose=2, n_jobs = 9)
ppn.fit(X_train, Y_train)

# Classify test samples
y_pred = ppn.predict(X_test)

C = confusion_matrix(Y_test,y_pred)

D = cohen_kappa_score(np.ravel(Y_test), y_pred)

#print(C)

# Measuring the accuracy in 3 different ways
print('Misclassified samples: %d' % (np.ravel(Y_test) != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(np.ravel(Y_test), y_pred))
print('Accuracy: %.2f' % ppn.score(X_test, np.ravel(Y_test)))