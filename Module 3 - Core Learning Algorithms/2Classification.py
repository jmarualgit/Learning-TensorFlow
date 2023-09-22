from __future__ import absolute_import, division, print_function, unicode_literals\

import pandas as pd
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset is of flowers
"""
    3 different specias: setosa, versicolor, virginica
    
    info abt each flower: sepal length & width + petal length & width
"""

# defining csv column names
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# loading our data sets and putting into train_path & test_path
# how get data is slightly different each time
# keras is a submodule of tensorflow
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# train and test are pandas dataframes
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header = 0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header = 0)

# pop to species column off and use it as label
# this will remove the Species column from train and test and 
# put it into variables train_y and test_y respectively
train_y = train.pop('Species')
test_y = test.pop('Species')

# input function
def input_fn(features, labels, training = true, batch_size = 256):
    # convert the inputs to a dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)