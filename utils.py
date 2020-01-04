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

# this var contains a set of features after filtering out some bias info from csv files generated
# from CICFlowMeter directly. The version of CICFlowMeter, I believe, is 4.0.
cic_cols = ['Protocol',  'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

# here you can filter out several features by seeing the correlation matrix or all constant-valued features
# In our experiment, it turns out filtering didn't improve the performance. The reason might be that our sampled
# data have some bias.
cic_cols_filtered = cic_cols

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# print the *str* in customized style when debugging
def my_print(str, option=0):
    if option==0:
        print(bcolors.WARNING, str, bcolors.ENDC)
    elif option==1:
        print(bcolors.HEADER, str, bcolors.ENDC)
    elif option==2:
        print(bcolors.OKBLUE, str, bcolors.ENDC)
    elif option==3:
        print(bcolors.OKGREEN, str, bcolors.ENDC)
    elif option==4:
        print(bcolors.FAIL, str, bcolors.ENDC)
    else:
        print(bcolors.BOLD, str, bcolors.ENDC)


# return a scaler according to the fitted data. In our case, it's purely benign.
def get_me_scaler(data):
    data = data.dropna()                                    # drop na
    features = data[cic_cols_filtered].astype('float64')    # here you can pass you customized cols
    features = features.replace(np.inf, 2e+10)              # replace inf
    scaler = StandardScaler()                               # here we use this scaler to get unit variance.
    return scaler.fit(features)


# scale the data according to the scaler obtained above
def dataPreprocessing(data, scaler):
    data = data.dropna()

    features = data[cic_cols_filtered].astype('float64')
    labels = np.array(data['Label'])
    features = features.replace(np.inf, 2e+10)

    if len(features) != 0:
        features = scaler.transform(features)

    # find the number of benign/malicious
    print('--------')
    uniq, cts = np.unique(labels, return_counts=True)
    print(np.asarray((uniq, cts)).T)
    print('------')

    return features, labels


# get pr, rc, f1 (you should be familiar with these metrics in ML) from conf matrix
def get_metrics(confusion_matrix):
    pr = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    rc = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
    f1 = 2 * pr * rc / (pr + rc)
    return pr, rc, f1

# I named it as *ownae* since it is designed by my own...
# You can easily see the layout by the code.
# Some factors: dropout, learning rate (you need to invoke the ctor outside), optimizer, activation function.
def ownae(input_dim=11, dropout=False, dr_rate=0.5, optimizer='adamax', act="relu"):
    my_print('my own AE ---------------------------')
    autoencoder = keras.models.Sequential()
    autoencoder.add(keras.layers.Dense(input_dim, activation=act, input_dim=input_dim))
    autoencoder.add(keras.layers.Dense(int(0.8 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.7 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.6 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.5 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.6 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.7 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.8 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(input_dim))

    autoencoder.compile(loss='mean_squared_error', optimizer=optimizer)
    return autoencoder


# this *bios* autoencoder is from N-BaIoT paper (google this name).
# Our experiment setup is similar to theirs.
def bios(input_dim=17, dropout=False, dr_rate=0.5,  optimizer='adamax', act="relu"):
    my_print('bios AE ---------------------------')
    autoencoder = keras.models.Sequential()
    autoencoder.add(keras.layers.Dense(int(0.75 * input_dim), activation=act, input_dim=input_dim))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.5 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.33 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.25 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.33 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.5 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(int(0.75 * input_dim), activation=act))
    if dropout:
        autoencoder.add(keras.layers.Dropout(dr_rate))
    autoencoder.add(keras.layers.Dense(input_dim))

    autoencoder.compile(loss='mean_squared_error', optimizer=optimizer)
    return autoencoder


# ------
# the functions below are for classification

# filter attacks only
def filter_attacks(df):
    return df[(df['Src IP'] == '172.16.0.5') | (df['Dst IP'] == '172.16.0.5')]

# get the base of a file.
def get_base(file):
    return os.path.splitext(os.path.basename(file))[0]
