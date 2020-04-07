# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:46:11 2020

@author: guilh
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

#Imports
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from IPython.display import Image,display
import pydot
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
dataset=sio.loadmat('data3SS2009')
##

##PREPARING DATA
x_raw=dataset['dataset']
y_raw=dataset['labels']

x=np.reshape(x_raw,(850,40960))
scaler= MinMaxScaler()
x_scaled=scaler.fit_transform(x)

y=np.zeros(shape=(len(y_raw),1))
for i in range(0,len(y)):
    if y_raw[i]<=9:
         y[i]=1
    else:
        y[i]=-1
##
        
##DIMENSIONALITY REDUCTION PCA
pca = PCA(n_components = 0.95)              #reduce to n PC's maintaining a 95% ratio of variance
X_red = pca.fit_transform(x)
##

##SPLIT TRAINING SET AND TEST SET
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X_red, y):
    X_train, X_test = X_red[train_index], X_red[test_index]
    y_train, y_test = y[train_index], y[test_index]
##

##Cross Validation
param_grid = {'max_leaf_nodes': list(range(2,100)), 
              'min_samples_split':[3, 4, 5],
              'max_depth':[10,20,30]
              },
      
dec_tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dec_tree, param_grid, cv=5)

grid_search.fit(X_train, y_train)

display(grid_search.best_params_)
##

##Training the best model on the full training set
dec_tree_best=grid_search.best_estimator_
dec_tree_best.fit(X_train,y_train)
##

##Accuracy/f1 Score for the training data
y_pred = dec_tree_best.predict(X_train)
display(accuracy_score(y_train, y_pred))
display(f1_score(y_train,y_pred))
##

##Evaluating the system on the Test Set
final_model=dec_tree_best
final_predictions=final_model.predict(X_test)
##

##Accuracy/f1 Score of the final model
display(accuracy_score(y_test, final_predictions))
display(f1_score(y_test, final_predictions))
##

##Visualizing the Decision Tree
dot_data=StringIO()
export_graphviz(final_model,
                out_file=dot_data,
                class_names=['Undamaged','Damaged'],
                rounded=True,
                filled=True
                )

graph=pydot.graph_from_dot_data(dot_data.getvalue())
display(Image(graph[0].create_png()))
##

##Confusion Matrix
conf_matrix=confusion_matrix(y_test,final_predictions)
display(conf_matrix)
##










