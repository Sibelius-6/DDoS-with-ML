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

# this is the tda lib we are using.
import kmapper as km


benign = pd.read_csv('all_benign.csv')
benign['Label'] = 0

my_scaler = get_me_scaler(benign)
X_benign, y_benign = dataPreprocessing2(benign, my_scaler) # scale the testing files

# as always, you could change optimizer.
optzer = "adamax"
input_dim = len(X_benign[0])
autoencoder = bios(input_dim=input_dim, dropout=False, optimizer=optzer)

# note that this is the file you are "plotting", be sure it has benign and attacks.
# The better AE performs, the better separation you can see in the tda visualization.
testt = pd.read_csv("attack.csv")
# labeling
testt['Label'] = testt.apply(lambda x: int(x['Src IP'] == '172.16.0.5' or x['Dst IP'] == '172.16.0.5'), axis=1)

X_test, y_test = dataPreprocessing(testt, my_scaler)

X_test_predictions = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - X_test_predictions, 2), axis=1)

mapper = km.KeplerMapper(verbose=3)
# here you could choose other projection functions. like knn distance, min, max...
# read the kmapper documentations.
lens2 = mapper.fit_transform(X_test, projection="l2norm")

# Combine both lenses to create a 2-D lens
lens = np.c_[mse_test, lens2]

# Create the simplicial complex
graph = mapper.map(lens,
                   X_test,
                   nr_cubes=15,
                   overlap_perc=0.4,
                   clusterer=km.cluster.KMeans(n_clusters=2,
                                               random_state=1618033))
file = 'tda.html'
_ = mapper.visualize(graph,
                 path_html=file,
                 title=attack,
                 custom_tooltips=y_test,
                 color_function=y_test
                )
