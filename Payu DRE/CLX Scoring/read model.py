# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:11:16 2019

@author: aman.khatri91425
"""

import os
import cx_Oracle
import pandas as pd
import numpy as np
from datetime import date,datetime,time
import xgboost as xgb
import warnings
import pickle
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score 
from inspect import signature




os.chdir(r"C:\Users\aman.khatri91425\Desktop\HomeCredit\Bigdata\PayU CLX Scoring")
df=pd.read_pickle(os.getcwd()+"\\Data Files\\payu_tenure_predictors_fstpd30_20_Sep_13hrs.pkl")
df.dropna(subset = ["RISK_SCORE"], inplace = True)
df = df[((df["DECISION_DATE"].dt.month != 1) & (df["DECISION_DATE"].dt.month != 6))]
X=df.drop(['DECISION_DATE','RISK_FSTPD30','MOBILE' ] ,axis=1)
Y=df['RISK_FSTPD30']

seed = 10
test_size = 0.2
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size, 
                                                 random_state = seed, stratify= df['DECISION_DATE'].dt.month)



bst = xgb.XGBClassifier()  # init model

bst.load_model(r'C:\Users\aman.khatri91425\Desktop\HomeCredit\Bigdata\PayU CLX Scoring\xgb_fst_clx_wo_hcscore_4variable_25_Sep_19hrs.model')

bst.predict_proba(X_train[['DIGITAL_RISK_SCORE','NIGHT_CNT_SALE_365','NIGHT_SALE_365','MORNING_SALE_180']])
