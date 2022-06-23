# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 01:39:01 2022

@author: Shekhar Rajput
"""


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\Shekhar\Desktop\minor/Crop_recommendation.csv')
data.head()

X = data.drop('label' ,axis =1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['label'] = le.fit_transform(data['label'])

y = data['label']

model = []
accuracy = []

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train , y_train)
predict = DT.predict(X_test)
DT_accuracy = DT.score(X_test,y_test)
DT_accuracy
