# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:26:40 2019

@author: homecredit
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import time
import pickle
import os
from math import log
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid,StratifiedKFold, cross_val_score

from sklearn import metrics 
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, brier_score_loss, f1_score, log_loss,auc,roc_curve

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.chdir(r"E:\Data Processing\aman\dump")

df4=pd.read_pickle(r"E:\Data Processing\aman\dump\xgb_raw_data_1608.pkl")
#df=pd.read_pickle(r"E:\Data Processing\aman\dump\customer_pred_scores_2008.pkl")


df4=df4[(df4['RISK_AGRF120']==1) & (df4['PRODUCT_GROUP']=='CD')]

df4['TIME_DECISION_DATE'] = pd.to_datetime(df4['TIME_DECISION_DATE'])
df4.dropna(subset=['HCIMODELSCORE'],inplace=True)
df4['month']=df4['TIME_DECISION_DATE'].dt.month
df4=df4[df4['month']!=4]


X=df4.drop(['CLIENT_EXI_3M' ,'RISK_FSTPD30', 'RISK_AGRF120' , 'RISK_AGRF90' , 'RISK_AGRF60' , 'RISK_FSTQPD30' , 'RISK_FSPD30' , 'RISK_FPD30' , 'ACTIVITY_PAYTM_DEF' , 'ACTIVITY' , 'RESULT' , 'TIME_DECISION_DATE' , 'PRODUCT_GROUP' , 'SKP_CLIENT' , 'SKP_CREDIT_CASE' , 'decision_date' , 'TIME_CREATION_DATE_max' , 'bill_date_min' , 'bill_date_max' ] ,axis=1)
Y=df4[['RISK_FSTQPD30']]




y_hcc = X[hcc_score]
df_hcc = pd.DataFrame({'true': np.ravel(Y), 'predict': y_hcc})
df_hcc = df_hcc.loc[pd.notnull(df_hcc['predict']),:]
metric_auc_hcc = metrics.roc_auc_score(df_hcc['true'], df_hcc['predict'])
gini_hcc_only = 2 * metric_auc_hcc - 1

gini_hcc_only


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify =df4['month'])

X_train.drop(['month'],axis=1,inplace=True) 
X_test.drop(['month'],axis=1,inplace=True) 


bst1 =  xgb.Booster({'nthread':-1})
bst1.load_model(r"E:\Data Processing\Aman\dump\xgb_1908_fstq_pos.model")



ypred_train = bst1.predict(X_train, ntree_limit=40)[:, 1]
ypred_test = bst1.predict_proba(X_test, ntree_limit=bst1.best_ntree_limit)[:, 1]



cd=pd.DataFrame(test[['HCIMODELSCORE','RISK_FSTQPD30']]).sort_values(by=['HCIMODELSCORE'],ascending=False)
#cd.to_csv("cd_hcscore.csv",index=0)
cd.groupby(pd.qcut(cd['HCIMODELSCORE'],10,duplicates='drop')).mean()
cd.groupby(pd.qcut(cd['HCIMODELSCORE'],10,duplicates='drop')).sum()
cd['RISK_FSTQPD30'].sum()

cd=pd.DataFrame(outtest[['pred','act']]).sort_values(by=['pred'],ascending=False)
#cd.to_csv("cd_hcscore.csv",index=0)
cd.groupby(pd.qcut(cd['pred'],10,duplicates='drop')).mean()
cd.groupby(pd.qcut(cd['pred'],10,duplicates='drop')).sum()
y_test.iloc[:,0].sum()
