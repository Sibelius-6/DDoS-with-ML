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

benign = pd.read_csv('all_benign.csv')
benign['Label'] = 0

my_scaler = get_me_scaler(benign)
B_X, B_y = dataPreprocessing(benign, my_scaler) # scale the benign

# here you can remove the random_state, then it's randomly shuffled
# Split the data for train and optimization
X_train, X_opt = train_test_split(B_X, test_size=0.2, random_state=66, shuffle=True)

input_dim = X_train.shape[1]

# the string one will invoke the default ctor with default parameters. If you want to
#  change some parameters, you can do sth like this:
#    optzer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Check the doc for more details
optzer = "adamax"

# here you can change the parameters as well as different structure of AE
autoencoder = bios(input_dim=input_dim, optimizer=optzer)

# print the structure
#print(autoencoder.summary())

history = autoencoder.fit(X_train, X_train, epochs=150,
                  validation_data=(X_opt, X_opt),
                  callbacks=[keras.callbacks.EarlyStopping(patience=15)],
                  verbose=1).history


# you can use save to save the model or save_weights. If you use save_weights, then
#   you would need to load the structure and load_weights.
autoencoder.save(filepath="model.h5")

# plot the model loss
plt.plot(history['loss'], linewidth=2, label="train")
plt.plot(history['val_loss'], linewidth=2, label="opt")
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig("model_loss.png")



# setting threshold according to the formula in Nabiot paper
x_opt_predictions = autoencoder.predict(X_opt)
mse = np.mean(np.power(X_opt - x_opt_predictions, 2), axis=1)

tr = mse.mean() + mse.std()




# ----------------------- separating line --------------------------------------
# load the attack csv files
testt = pd.read_csv('attack.csv')
# label the csv files by src/dst ip
testt['Label'] = testt.apply(lambda x: int(x['Src IP'] == '172.16.0.5' or x['Dst IP'] == '172.16.0.5'), axis=1)

X_test, y_test = dataPreprocessing(testt, my_scaler) # scale the testing files


# make the prediction
X_test_predictions = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - X_test_predictions, 2), axis=1)
y_pred = np.array(list(map(lambda x : 1 if x > tr else 0, mse_test)))


conf_matrix = confusion_matrix(y_test, y_pred)
pr, rc, f1 = get_metrics(conf_matrix)
auc = roc_auc_score(y_test, y_pred)
my_print('--------report------')
print('pr:', pr)
print('rc:', rc)
print('f1:', f1)
print('auc', auc)
my_print('--------end---------')
