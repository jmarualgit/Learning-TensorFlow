# taken from python.py, took out some comments for cleanliness

from __future__ import absolute_import, division, print_function, unicode_literals\

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load data sets
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')   # training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')     # testing dataset

# removes 'survived' column and puts into y_train and y_eval
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# creating columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# for linear estimators
feature_columns = []    # blank list
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()     # .unique() gets a list of all unique values from given feature column
    
    # creates a row of feature_name and associated vocabulary
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    
    # create a column of feature_name and associated data type
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))

print(feature_columns)

#print(dftrain["sex"].unique())
    # prints out all of the values(2) of "sex"

# training process
# epoch = one stream of entire dataset