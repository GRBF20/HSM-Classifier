# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

##SETUP

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

##


#LOADING DATASET

import scipy.io as sio

dataset=sio.loadmat('dataset')

##

##VISUALIZING DATA
x=dataset['obs']

y=dataset['grp']

y_n=np.zeros(shape=(len(y),1))

for i in range(0,len(y)):
    if y[i]=='Normal':
         y_n[i]=0
    else:
        y_n[i]=1

##DIMENSIONALITY REDUCTION PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.95)              #reduce to n PC's maintaining a 95% ratio of variance
X_red = pca.fit_transform(x)


##SPLIT TRAINING SET AND TEST SET

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_red, y_n, test_size=0.2, random_state=42)

##

##DIMENSIONALITY REDUCTION PCA

#from sklearn.decomposition import PCA

#pca = PCA(n_components = 0.95)              #reduce to n PC's maintaining a 95% ratio of variance
#X_train_red = pca.fit_transform(X_train)
#X_test_red=pca.fit_transform(X_test)
##


##Cross Validation

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'max_leaf_nodes': list(range(2,100)), 'min_samples_split':[2, 3, 4, 5]},
      

dec_tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dec_tree, param_grid, cv=5)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
##

##Training the best model on the full training set

dec_tree_best=grid_search.best_estimator_
dec_tree_best.fit(X_train,y_train)

##

##Accuracy for the training data

from sklearn.metrics import accuracy_score

y_pred = dec_tree_best.predict(X_train)
print(accuracy_score(y_train, y_pred))

##Evaluating the system on the Test Set

final_model=dec_tree_best
final_predictions=final_model.predict(X_test)
##

##Accuracy of the final model

print(accuracy_score(y_test, final_predictions))
