import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import confusion_matrix,recall_score,classification_report,f1_score,roc_auc_score
from sklearn.ensemble import IsolationForest,RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Note the mess above...

from utils import *

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

X_train, y_train = dataPreprocessing(all_train, my_scaler)

# here you have plenty of choices...
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
xgb_clf = XGBClassifier()


voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf)],
    voting='hard')
bag_clf = BaggingClassifier(
    RandomForestClassifier(), bootstrap=True, n_jobs=-1)

ada_clf = AdaBoostClassifier(
    RandomForestClassifier(), algorithm="SAMME.R")

# you can change the model here.
model = rnd_clf

# here is the actual fit.
model.fit(X_train, y_train)



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

X_test, y_test = dataPreprocessing(all_test, my_scaler)

y_pred = model.predict(X_test) 

acc_score = accuracy_score(y_test, y_pred)
my_print('acc_score is ' + str(acc_score))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
