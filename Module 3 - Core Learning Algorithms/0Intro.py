from __future__ import absolute_import, division, print_function, unicode_literals\

# optimized version of arrays; lots of multidimensional calculations
# representing such data and easily manipulate and perform operations (cross & dot product, etc.)
import numpy as np

# data analytics tool
# if ModuleNotFoundError -> pip install pandas
import pandas as pd

# graphs and charts
# pip install matplotlib
import matplotlib.pyplot as plt

# clear output
# pip install IPython
from IPython.display import clear_output

# for python 3.0 stuff
from six.moves import urllib

# for linear regression algorithm model
import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# error came up, might be something to do with not using CUDA
# this fixed it
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# linear regression
""""""

# one of the most basic forms of machine learning; used to predict numeric values
# find linear correspondence

# loading practice dataset
    # titanic dataset; likelihood of someone surviving on titanic given info

# using pandas; pd = panda
# csv = comma separated values
# loading into data frame objects, can reference certain rows and columns
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')   # training dataset
    # looking at data, it has many values to look at for analysis; survived, sex, age, # of siblings, class (first, third, etc.), etc.

dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')     # testing dataset

# prints organized in terminal in chart form
# head shows the first 5 entries in data set
#print(dftrain.head())

# the outcome
"""
   survived     sex   age  n_siblings_spouses  parch     fare  class     deck  embark_town alone
0         0    male  22.0                   1      0   7.2500  Third  unknown  Southampton     n
1         1  female  38.0                   1      0  71.2833  First        C    Cherbourg     n
2         1  female  26.0                   0      0   7.9250  Third  unknown  Southampton     y
3         1  female  35.0                   1      0  53.1000  First        C  Southampton     n
4         0    male  28.0                   0      0   8.4583  Third  unknown   Queenstown     y
"""

# removes 'survived' column and puts them into y_train
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#print(dftrain.head())

# the outcome
"""
      sex   age  n_siblings_spouses  parch     fare  class     deck  embark_town alone
0    male  22.0                   1      0   7.2500  Third  unknown  Southampton     n
1  female  38.0                   1      0  71.2833  First        C    Cherbourg     n
2  female  26.0                   0      0   7.9250  Third  unknown  Southampton     y
3  female  35.0                   1      0  53.1000  First        C  Southampton     n
4    male  28.0                   0      0   8.4583  Third  unknown   Queenstown     y
"""

# prints y_train which has the 'survived' column as well as how long the data is
#print(y_train)

# outcome
"""
0      0
1      1
2      1
3      1
4      0
      ..
622    0
623    0
624    1
625    0
626    0
Name: survived, Length: 627, dtype: int64
"""

#print(dftrain["age"])   # can look at a whole column

# the outcome
"""
0      22.0
1      38.0
2      26.0
3      35.0
4      28.0
       ...
622    28.0
623    25.0
624    19.0
625    28.0
626    32.0
Name: age, Length: 627, dtype: float64
"""

# .loc[] finds one specific row in dataframe
#print(dftrain.loc[0], y_train.loc[0])   # row 0

# the outcome
"""
sex                          male
age                          22.0
n_siblings_spouses              1
parch                           0
fare                         7.25
class                       Third
deck                      unknown
embark_town           Southampton
alone                           n
Name: 0, dtype: object 0
"""
    # 0 refers to them not surviving
    
# overall info; stat analysis
#print(dftrain.describe())

# the outcome
"""
              age  n_siblings_spouses       parch        fare
count  627.000000          627.000000  627.000000  627.000000
mean    29.631308            0.545455    0.379585   34.385399
std     12.511818            1.151090    0.792999   54.597730
min      0.750000            0.000000    0.000000    0.000000
25%     23.000000            0.000000    0.000000    7.895800
50%     28.000000            0.000000    0.000000   15.045800
75%     35.000000            1.000000    0.000000   31.387500
max     80.000000            8.000000    5.000000  512.329200
"""

# dataframes have shapes too and can look at it
#print(dftrain.shape)

# the outcome
"""
(627, 9)
"""
    # 627 rows and 9 columns

# making histrograms and plots for analysis and intuition

# a histogram of the age
#dftrain.age.hist(bins = 20)

# horizontal bar graph of sexes
#dftrain.sex.value_counts().plot(kind="barh")

# horizontal bar of their classes (first, second, etc.)
#dftrain['class'].value_counts().plot(kind='barh')

# showing graph(s); don't use print()
plt.show()