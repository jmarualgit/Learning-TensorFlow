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

# input function; the way we define how the data will be broken into batches and epochs

# takes data and encodes in a tf.data.Dataset object 
    # because need a Dataset object to see data and create a model
# taken straight from TensorFlow documentation (https://www.tensorflow.org/tutorials/estimator/linear)

# explaining the parameters of function make_input_fn
"""
    data_df: pandas dataframe
    label_df: the labels dataframe; the y_train, y_eval etc.
    num_epochs: how many epochs will be done
    shuffle: will shuffle/mix before passing to model or not
    batch_size: how many elements will be given to model while training
"""
# outer function makes an input_function and returns the function object to wherever called from
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    
    # inner function, this will be returned
    def input_function():  
        
        # create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  
        
        if shuffle:
            # shuffles the data set/randomize order of data
            ds = ds.shuffle(1000)
        
        # split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        
        # return a batch of the dataset
        return ds
    return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Linear Classifier object from estimator module from TensorFlow
# creates an estimator; a basic implementation
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# training the model
# give the input function
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears console output
#print(result['accuracy'])  # the result variable is simply a dict of stats about our model

# printed out '0.7613636'; will prob be different each time because shuffled each time

# result is a dictionary object and can reference any key needed / wanted
# print(result)

# making predictions
result = list(linear_est.predict(eval_input_fn))

# a dictionary of one prediction
print(result[0]['probabilities'][1])
    # prints out [0.8350972  0.16490284] (different for everytime run) without the [1]
    # want to find the change of surviving (surviving = 1; dead = 0) so index 1

# did they survive or not
print("did they survive?")
print(y_eval.loc[2])    

# look at person predicted for
print(dfeval.loc[0])

# the outcome
"""
0.042104978
sex                          male
age                          35.0
n_siblings_spouses              0
parch                           0
fare                         8.05
class                       Third
deck                      unknown
embark_town           Southampton
alone                           y
Name: 0, dtype: object
"""

# compare it to next person
print(result[2]['probabilities'][1])
print("did they survive?")
print(y_eval.loc[2])
print(dfeval.loc[2])

# the outcome
"""
0.86056805
sex                        female
age                          58.0
n_siblings_spouses              0
parch                           0
fare                        26.55
class                       First
deck                            C
embark_town           Southampton
alone                           y
Name: 2, dtype: object
"""
    # higher change of survival; probably because female?

# note: model isn't perfect
# needs more data