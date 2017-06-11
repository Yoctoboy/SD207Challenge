# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:36:57 2017

@author: essid
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

FILEROOT = "audio/"

# ----------------------------------------------------------------------------

train_files = pd.read_table(os.path.join(FILEROOT, "train.txt"), names=['file', 'label'], sep='\s+')
test_files = pd.read_table(os.path.join(FILEROOT, "test_files.txt"), names=['file', 'label'], sep='\s+')

labels = {'beach': 0,
          'bus': 1,
          'cafe/restaurant': 2,
          'car': 3,
          'city_center': 4,
          'forest_path': 5,
          'grocery_store': 6,
          'home': 7,
          'library': 8,
          'metro_station': 9,
          'office': 10,
          'park': 11,
          'residential_area': 12,
          'train': 13,
          'tram' : 14
          }

for i, afile in train_files.iterrows():

    y, sr = librosa.load(str(afile.file), sr=None)
    mfcc = librosa.feature.mfcc(y=y, n_fft=256, hop_length=512, n_mfcc=20)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(np.abs(mfcc), ref=np.max),
    #                         y_axis='mel', x_axis='time')
    mfcc = np.ravel(mfcc)
    if i == 0:
        X_train = [mfcc]
        y_train = [labels[afile.label]]
    else:
        X_train += [mfcc]
        y_train += [labels[afile.label]]
    if (i%10==0 or i==train_files.shape[0]):
        print("Train : {0}/{1}".format(i, train_files.shape[0]))

for i, afile in test_files.iterrows():

    y, sr = librosa.load(str(afile.file), sr=None)
    mfcc = librosa.feature.mfcc(y=y, n_fft=256, hop_length=512, n_mfcc=20)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(np.abs(mfcc), ref=np.max),
    #                         y_axis='mel', x_axis='time')
    mfcc = np.ravel(mfcc)
    if i == 0:
        X_test = [mfcc]
    else:
        X_test += [mfcc]
    if (i+1 % 10 == 0 or i+1 == test_files.shape[0]):
        print("Test : {0}/{1}".format(i+1, test_files.shape[0]))

X_tot = np.concatenate([X_train, X_test])
preprocessing.scale(X_tot)
X_train = X_tot[:len(X_train)]
X_test = X_tot[len(X_train):]
clf = Lasso(max_iter=5000)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
np.savetxt("y_pred.txt", y_pred, fmt="%i")