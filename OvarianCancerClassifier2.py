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

#Imports
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
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
dataset=sio.loadmat('dataset')

##

##PREPARING DATA
x=dataset['obs']
y=dataset['grp']

y_n=np.zeros(shape=(len(y),1))

for i in range(0,len(y)):
    if y[i]=='Normal':
         y_n[i]=0
    else:
        y_n[i]=1

##

##DIMENSIONALITY REDUCTION PCA
pca = PCA(n_components = 0.95)              #reduce to n PC's maintaining a 95% ratio of variance
X_red = pca.fit_transform(x)

##

##SPLIT TRAINING SET AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X_red, y_n, test_size=0.2, random_state=42)

##

##Cross Validation
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
y_pred = dec_tree_best.predict(X_train)
print(accuracy_score(y_train, y_pred))

##Evaluating the system on the Test Set
final_model=dec_tree_best
final_predictions=final_model.predict(X_test)

##

##Accuracy of the final model
print(accuracy_score(y_test, final_predictions))

##

##Visualizing the Decision Tree
dot_data=StringIO()
export_graphviz(final_model,
                out_file=dot_data,
                class_names=['Normal','Cancer'],
                rounded=True,
                filled=True
                )

graph=pydot.graph_from_dot_data(dot_data.getvalue())
display(Image(graph[0].create_png()))

##

##Confusion Matrix

conf_matrix=confusion_matrix(y_test,final_predictions)
display(conf_matrix)
display(precision_score(y_test,final_predictions))
display(recall_score(y_test,final_predictions))
display(f1_score(y_test,final_predictions))