import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import confusion_matrix,recall_score,classification_report,f1_score,roc_auc_score
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn import svm
import itertools
import math
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import json
import glob
import os
import time
import sys
from scipy.stats import reciprocal

# Note the mess above...

from utils import *


class_names = ["dns", "ldap", "mssql", "netbios", "ntp", "snmp", "ssdp", "syn", "tftp", "udp", "udplag", "benign", "portmap"]

# these two functions set up the conversions between str and int. since MLP classifier would require number as output.
def str2int(string):
    return class_names.index(string)

def int2str(integer):
    return class_names[integer]

def dataPreprocessing2(data, scaler):
    data = data.dropna()

    features = data[cic_cols_filtered].astype('float64')
    # note this extra line which is for converting into numbers
    labels = np.array(data['Label'].apply(str2int))
    features = features.replace(np.inf, 2e+10)

    if len(features) != 0:
        features = scaler.transform(features)

    return features, labels






benign = pd.read_csv('all_benign.csv')
benign['Label'] = 'benign'



my_scaler = get_me_scaler(benign)

# glob all files
train_csvs = glob.glob('train/*.csv')
test_csvs = glob.glob('test/*.csv')

train_li = []

for file in train_csvs:
    attack = get_base(file)
    df = pd.read_csv(file, index_col=None, header=0)
    df = filter_attacks(df)
    df['Label'] = attack
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', 1) # some files might have this unexpected col
    train_li.append(df)

all_train = pd.concat(train_li, axis=0, ignore_index=True)

X, y = dataPreprocessing2(all_train, my_scaler)



input_dim = len(X[0])


# load our MLP model. You can try other structures as well, but you must have the
# last layer to be softmax.
model = keras.models.Sequential([
    keras.layers.Dense(input_dim, activation="relu", input_dim=input_dim),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(12, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])

# As usual, you can remove the random_state.
X_train, X_opt, y_train, y_opt = train_test_split(X, y, test_size=0.2, random_state=66, shuffle=True)

history = model.fit(X_train, y_train, epochs=50,
                          validation_data=(X_opt, y_opt),
                          verbose=1).history

# ---- now testing
test_li = []

for file in test_csvs:
    attack = get_base(file)
    df = pd.read_csv(file, index_col=None, header=0)
    df = filter_attacks(df)
    df['Label'] = attack
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', 1) # some files might have this unexpected col
    test_li.append(df)

all_test = pd.concat(test_li, axis=0, ignore_index=True)

X_test, y_test = dataPreprocessing2(all_test, my_scaler)
y_test = np.array(class_names)[y_test] # this is to convert it back to str

y_pred = model.predict_classes(X_test)
y_pred = np.array(class_names)[y_pred]

acc_score = accuracy_score(y_test, y_pred)
my_print('acc_score is ' + str(acc_score))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
