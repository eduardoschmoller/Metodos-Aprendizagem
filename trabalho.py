#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 18:50:26 2018

@author: eduardo
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score 

from sklearn import decomposition

#entrada de dados
X_train = pd.read_csv("X_train.txt", sep = " ", header=None)
Y_train = pd.read_csv("y_train.txt", sep = " ", header=None)
X_test = pd.read_csv("X_test.txt", sep = " ", header=None)
Y_test = pd.read_csv("y_test.txt", sep = " ", header=None)

X = X_train.append(X_test)#transforma treino e teste em um só conjunto para depois separar em dois grupos aleatórimente
Y = np.ravel(Y_train.append(Y_test))

# Splitting data into 80% training and 20% test data:
from sklearn.model_selection import train_test_split


#classificador
nbrs = knn(n_neighbors=5, algorithm= 'kd_tree')#kernel pode ser 'rbf', 'linear', 'poly', 'signmoid'

#nbrs.fit(X_train,Y_train)
#
#pred = nbrs.predict(X_test)
#
#print(accuracy_score(np.ravel(Y_test), pred))
#
#C = confusion_matrix(np.ravel(Y_test), pred)
#
#D = cohen_kappa_score(np.ravel(Y_test), pred)
a = list()
D1 = list()
b = range(10,X.shape[1],25)
for i in b:
    X_pca = X
    pca = decomposition.PCA(n_components=i)
    pca.fit(X_pca)
    X_pca = pca.transform(X_pca)
    X_train, X_test, Y_train, Y_test = train_test_split( X_pca, Y, test_size=0.2, random_state = np.random.randint(100000))
    
    # KNN
    nbrs.fit(X_train,Y_train)
    pred_pca = nbrs.predict(X_test)
    a.append(accuracy_score(np.ravel(Y_test), pred_pca))
    D1.append(cohen_kappa_score(np.ravel(Y_test), pred_pca, weights="quadratic"))

    # SVM
    
    
    #PERCETRON

#    print(a[-1])
#    print("\n")
#    print(D1[-1])
#    print("\n----")
#    
C1 = confusion_matrix(np.ravel(Y_test), pred_pca)
