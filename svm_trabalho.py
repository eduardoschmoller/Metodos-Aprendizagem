#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 18:50:26 2018

@author: eduardo
"""

import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score 

#entrada de dados
X_train = pd.read_csv("X_train.txt", sep = " ", header=None)
Y_train = pd.read_csv("y_train.txt", sep = " ", header=None)
X_test = pd.read_csv("X_test.txt", sep = " ", header=None)
Y_test = pd.read_csv("y_test.txt", sep = " ", header=None)

X = X_train.append(X_test)#transforma treino e teste em um só conjunto para depois separar em dois grupos aleatórimente
Y = np.ravel(Y_train.append(Y_test))

# Splitting data into 80% training and 20% test data:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state = np.random.randint(100000))

#classificador
clf = SVC(kernel = 'rbf')#kernel pode ser 'rbf', 'linear', 'poly', 'signmoid'

clf.fit(X_train, np.ravel(Y_train))

pred = clf.predict(X_test)

print(accuracy_score(np.ravel(Y_test), pred))

C = confusion_matrix(np.ravel(Y_test), pred)

D = cohen_kappa_score(np.ravel(Y_test), pred)

print(D)