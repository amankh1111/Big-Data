# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:03:16 2019

@author: aman.khatri91425
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:04:21 2019

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


# Event rate
df["RISK_FSTPD30"].sum()/df["RISK_FSTPD30"].count()

# drop cases where risk score is not available
df.dropna(subset = ["RISK_SCORE"], inplace = True)


# check data availability across different months
df["DECISION_DATE"].dt.strftime('%B %Y').value_counts()
df["DECISION_DATE"].dt.month.plot.hist()

# remove 1st and 6th month due to low value counts
df = df[((df["DECISION_DATE"].dt.month != 6))]


# check distribution of HC score
df["RISK_SCORE"].plot.hist(bins=50, range=(0,0.04) ,color='#607c8e')


#Feature addition



#%ages
for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'AMT_RFND_180' in y]:
    df[x+"%"] = df[x]/df["TOTAL_AMT_RFND_180"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'AMT_RFND_365' in y]:
    df[x+"%"] = df[x]/df["TOTAL_AMT_RFND_365"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'AMT_RFND_90' in y]:
    df[x+"%"] = df[x]/df["TOTAL_AMT_RFND_90"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'AMT_SALE_180' in y]:
    df[x+"%"] = df[x]/df["TOTAL_AMT_SALE_180"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'AMT_SALE_365' in y]:
    df[x+"%"] = df[x]/df["TOTAL_AMT_SALE_365"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'AMT_SALE_90' in y]:
    df[x+"%"] = df[x]/df["TOTAL_AMT_SALE_90"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'CNT_RFND_180' in y]:
    df[x+"%"] = df[x]/df["TOTAL_CNT_RFND_180"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'CNT_RFND_365' in y]:
    df[x+"%"] = df[x]/df["TOTAL_CNT_RFND_365"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'CNT_RFND_90' in y]:
    df[x+"%"] = df[x]/df["TOTAL_CNT_RFND_90"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'CNT_SALE_180' in y]:
    df[x+"%"] = df[x]/df["TOTAL_CNT_SALE_180"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'CNT_SALE_365' in y]:
    df[x+"%"] = df[x]/df["TOTAL_CNT_SALE_365"]

for x in [y for y in df.drop(columns=[z for z in df.columns if 'TOTAL' in z]).columns if 'CNT_SALE_90' in y]:
    df[x+"%"] = df[x]/df["TOTAL_CNT_SALE_90"]



df["TOTAL_AMT_RFND_180_per"] = df["TOTAL_AMT_RFND_180"]/df["TOTAL_AMT_SALE_180"]
df["TOTAL_AMT_RFND_365_per"] = df["TOTAL_AMT_RFND_365"]/df["TOTAL_AMT_SALE_365"]
df["TOTAL_AMT_RFND_90_per"] = df["TOTAL_AMT_RFND_90"]/df["TOTAL_AMT_SALE_90"]

df["TOTAL_CNT_RFND_180_per"] = df["TOTAL_CNT_RFND_180"]/df["TOTAL_CNT_SALE_180"]
df["TOTAL_CNT_RFND_365_per"] = df["TOTAL_CNT_RFND_365"]/df["TOTAL_CNT_SALE_365"]
df["TOTAL_CNT_RFND_90_per"] = df["TOTAL_CNT_RFND_90"]/df["TOTAL_CNT_SALE_90"]


#convergence
for x in [y for y in df.drop(columns=[z for z in df.columns if 'AIRLINES' in z]).columns if '90' in y]:
    df[x.replace('90','90to365')]= df[x]/df[x.replace('90','365')]
    df[x.replace('90','180to365')]= df[x.replace('90','180')]/df[x.replace('90','365')]


    
df["AF_div_EN"]=df["AFFLUENCE_SCORE"]/df["ENGAGEMENT_SCORE"]
df["AF_div_DR"]=df["AFFLUENCE_SCORE"]/df["DIGITAL_RISK_SCORE"]
df["EN_div_AF"]=df["ENGAGEMENT_SCORE"]/df["AFFLUENCE_SCORE"]


df["EN_div_DR"]=df["ENGAGEMENT_SCORE"]/df["DIGITAL_RISK_SCORE"]
df["DR_div_AF"]=df["DIGITAL_RISK_SCORE"]/df["AFFLUENCE_SCORE"]
df["DR_div_EN"]=df["DIGITAL_RISK_SCORE"]/df["ENGAGEMENT_SCORE"]

df["AF_mul_EN"]=df["AFFLUENCE_SCORE"]*df["ENGAGEMENT_SCORE"]
df["EN_mul_DR"]=df["ENGAGEMENT_SCORE"]*df["DIGITAL_RISK_SCORE"]
df["DR_mul_AF"]=df["DIGITAL_RISK_SCORE"]*df["AFFLUENCE_SCORE"]




df[["WDAY_CNT_SALE_90to365%","WDAY_CNT_SALE_90to365"]].head()

# Description of variables
ds=df.describe().transpose()
ds.to_csv(os.getcwd()+"\eda_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")

# data preparation         
X=df.drop(['DECISION_DATE','RISK_FSTPD30','MOBILE' ] ,axis=1)
Y=df['RISK_FSTPD30']


# HCC gini score calculation
y_hcc = X["RISK_SCORE"]
df_hcc = pd.DataFrame({'true':Y,'predict':y_hcc})
metric_auc_hcc = metrics.roc_auc_score(df_hcc['true'], df_hcc['predict'])
gini_hcc_only = 2 * metric_auc_hcc - 1

# For generating logs
lg=open(os.getcwd()+"\log_xgb_clx_wo_hc.txt","a")   
print("Home Credit Only GINI:", '{:,.2%}'.format(gini_hcc_only))
lg.write("Home Credit Only GINI:"  + '{:,.2%}'.format(gini_hcc_only))


# define model parameters
missing_value=None; cpu=-1; test_size=0.3; early_stopping_round=40; split_random_state=None


#Split data into train and test set---Stratified sampling on month
seed = 3
test_size = 0.2
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size, 
                                                 random_state = seed, stratify= df['DECISION_DATE'].dt.month)

print("\nEvent rate in train set :", '{:,.2%}'.format(sum(Y_train) / len(Y_train)))
lg.write("\nEvent rate in train set :" + '{:,.2%}'.format(sum(Y_train) / len(Y_train)))

print("\nEvent rate in test set :", '{:,.2%}'.format(sum(Y_test) / len(Y_test)))
lg.write("\nEvent rate in test set :"+ '{:,.2%}'.format(sum(Y_test) / len(Y_test)))


metric_auc_hcc_train = metrics.roc_auc_score(Y_train, X_train['RISK_SCORE'])
gini_hcc_train = 2 * metric_auc_hcc_train - 1

metric_auc_hcc_test = metrics.roc_auc_score(Y_test, X_test['RISK_SCORE'])
gini_hcc_test = 2 * metric_auc_hcc_test - 1

print("\nGini in train set :", '{:,.2%}'.format(gini_hcc_train))
lg.write("\nGini in train set :" + '{:,.2%}'.format(gini_hcc_train))

print("\nGini in test set :", '{:,.2%}'.format(gini_hcc_test))
lg.write("\nGini in test set :"+ '{:,.2%}'.format(gini_hcc_test))



## Variable importance Function for xgboost....to present importance in a cleaner way
def fimportances(xc, predictors):
    importances = pd.DataFrame({'predictor': predictors, 'importance': xc.feature_importances_})
    importances = importances[importances['importance'] > 0]
    importances.sort_values(by='importance', ascending=False, inplace=True)
    importances.reset_index(inplace=True, drop=True)
    importances = importances[['predictor', 'importance']]
    return importances





#################################Variable Selection######################
    
params = {'max_depth': 3,
              'learning_rate': 0.01,
              'subsample': 0.5,
              'min_child_weight': 11,
              'colsample_bytree': 0.7,
              'scale_pos_weight' :25,
              'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200
            }
kfolds = StratifiedKFold(10, random_state=1)
xgb_model = xgb.XGBClassifier(**params)

bst = xgb_model.fit(X_train.drop(columns='RISK_SCORE'), np.ravel(Y_train), eval_metric="auc", eval_set=[(X_test.drop(columns='RISK_SCORE'), np.ravel(Y_test))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train.drop(columns = 'RISK_SCORE'))[:, 1]
    ypred_test = bst.predict_proba(X_test.drop(columns = 'RISK_SCORE'))[:, 1]
else:
    ypred_train = bst.predict_proba(X_train.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train.drop(columns = 'RISK_SCORE').columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_wo_hcscore_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)


print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))







####################################################################################


X_train_1 = X_train[list(importance_tbl["predictor"])+list(["RISK_SCORE"])]
X_test_1 = X_test[list(importance_tbl["predictor"])+list(["RISK_SCORE"])]

params = {'max_depth': 3,
              'learning_rate': 0.01,
              'subsample': 0.5,
              'min_child_weight': 11,
              'colsample_bytree': 0.7,
              'scale_pos_weight' :25,
              'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200
            }
kfolds = StratifiedKFold(10, random_state=1)
xgb_model = xgb.XGBClassifier(**params)


auc_cv = cross_val_score(xgb_model, X_train_1.drop(columns = 'RISK_SCORE'), Y_train, scoring='roc_auc', cv=kfolds.split(X_train_1.drop(columns='RISK_SCORE'), Y_train),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()



bst = xgb_model.fit(X_train_1.drop(columns='RISK_SCORE'), np.ravel(Y_train), eval_metric="auc", eval_set=[(X_test_1.drop(columns='RISK_SCORE'), np.ravel(Y_test))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train_1.drop(columns = 'RISK_SCORE'))[:, 1]
    ypred_test = bst.predict_proba(X_test_1.drop(columns = 'RISK_SCORE'))[:, 1]
else:
    ypred_train = bst.predict_proba(X_train_1.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test_1.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train_1.drop(columns = 'RISK_SCORE').columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_wo_hcscore_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)


print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))


    
pickle.dump(bst, open(os.getcwd()+"/xgb_fst_clx_169var_"+datetime.now().strftime("%d_%b_%H")+"hrs", 'wb'))
#bst1 = pickle.load(open(os.getcwd()+"/xgb_fst_clx_169var_"+datetime.now().strftime("%d_%b_%H")+"hrs", 'rb'))


##########################################################################








#########################################################################
X_train_2 = X_train_1.drop(columns = 'RISK_SCORE')
X_train_2 = X_train_2.replace({np.nan:-9999999,np.Inf:9999999,np.NINF:-9999999})

from sklearn.feature_selection import SelectFromModel

#bst1 = bst
#X_train_2 = X_train_1
#select_X_train = X_train_1
#
#while select_X_train.shape[1]>1:
#    # select features using threshold
#    selection = SelectFromModel(bst1, threshold=list(importance_tbl['importance'])[-2], prefit=True)   
#    select_X_train = pd.DataFrame(selection.transform(X_train_2))
#    feature_idx = selection.get_support() 
#    select_X_train.columns = X_train_2.columns[feature_idx]
#    # train model
#    selection_model = xgb.XGBClassifier(**params) 
#    auc_cv = cross_val_score(selection_model, select_X_train, Y_train, scoring='roc_auc', cv=kfolds.split(select_X_train, Y_train),
#                         n_jobs=cpu)
#    gini_cv = (2 * np.array(auc_cv) - 1).mean()
#    bst1 = selection_model.fit(select_X_train, Y_train)
#    importance_tbl = fimportances(bst1,select_X_train.columns)
#    print("\nThresh=%.5f, n=%d, gini_cv: %.2f%% variables: %s"% (list(importance_tbl['importance'])[-2], select_X_train.shape[1], gini_cv*100.0, ",".join(importance_tbl['predictor'])))
#    lg.write("\nThresh=%.5f, n=%d, gini_cv: %.2f%% variables: %s"% (list(importance_tbl['importance'])[-2], select_X_train.shape[1], gini_cv*100.0, ",".join(importance_tbl['predictor'])))
#    X_train_2 = select_X_train

test_size = 0.3
X_train_tmp,X_test_tmp,Y_train_tmp,Y_test_tmp = train_test_split(X_train_2,Y_train,test_size=test_size, 
                                                 random_state = seed)



xgb_model = xgb.XGBClassifier(**params)
auc_cv = cross_val_score(xgb_model, X_train_tmp, Y_train_tmp, scoring='roc_auc', cv=kfolds.split(X_train_tmp, Y_train_tmp),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()



bst = xgb_model.fit(X_train_tmp, np.ravel(Y_train_tmp), eval_metric="auc", eval_set=[(X_test_tmp, np.ravel(Y_test_tmp))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train_tmp)[:, 1]
    ypred_test = bst.predict_proba(X_test_tmp)[:, 1]
else:
    ypred_train = bst.predict_proba(X_train_tmp, ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test_tmp, ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train_tmp, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test_tmp, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train_tmp.columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_wo_hcscore_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)


print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))







for thresh in importance_tbl["importance"]:
	# select features using threshold
    selection = SelectFromModel(bst, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train_tmp)
    select_X_test = selection.transform(X_test_tmp)
	# train model
    selection_model = xgb.XGBClassifier(**params)
    selection_model.fit(select_X_train, Y_train_tmp)

#    bst = selection_model.fit(select_X_train, np.ravel(Y_train_tmp))

	# eval model
    y_pred_test = selection_model.predict_proba(select_X_test)[:, 1]
    y_pred_train = selection_model.predict_proba(select_X_train)[:, 1]    
#    if early_stopping_round is None:
#        y_pred_train = bst.predict_proba(select_X_train)[:, 1]
#        y_pred_test = bst.predict_proba(select_X_test)[:, 1]
#    else:
#        y_pred_train = bst.predict_proba(select_X_train, ntree_limit=bst.best_ntree_limit)[:, 1]
#        y_pred_test = bst.predict_proba(select_X_test, ntree_limit=bst.best_ntree_limit)[:, 1]

    metric_auc_train = metrics.roc_auc_score(Y_train_tmp, y_pred_train)
    metric_auc_test = metrics.roc_auc_score(Y_test_tmp, y_pred_test)
    gini_train = 2 * metric_auc_train - 1
    gini_test = 2 * metric_auc_test - 1
    print("\n Thresh=%.3f, n=%d, gini_test: %.2f%%, gini_train:%.2f%%" % (thresh, select_X_train.shape[1], gini_test*100.0, gini_train*100.0))
    lg.write("\n Thresh=%.3f, n=%d, gini_test: %.2f%%, gini_train:%.2f%%" % (thresh, select_X_train.shape[1], gini_test*100.0, gini_train*100.0))



#########################################################################################
    
    
thresh=0.0105852755

X_train_1 = X_train[list(importance_tbl[importance_tbl["importance"]>=0.0105852755]["predictor"])+list(["RISK_SCORE"])]
X_test_1 = X_test[list(importance_tbl[importance_tbl["importance"]>=0.0105852755]["predictor"])+list(["RISK_SCORE"])]

params = {'max_depth': 3,
              'learning_rate': 0.01,
              'subsample': 0.5,
              'min_child_weight': 11,
              'colsample_bytree': 0.7,
              'scale_pos_weight' :25,
              'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200
            }
kfolds = StratifiedKFold(10, random_state=1)
xgb_model = xgb.XGBClassifier(**params)


auc_cv = cross_val_score(xgb_model, X_train_1.drop(columns = 'RISK_SCORE'), Y_train, scoring='roc_auc', cv=kfolds.split(X_train_1.drop(columns='RISK_SCORE'), Y_train),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()



bst = xgb_model.fit(X_train_1.drop(columns='RISK_SCORE'), np.ravel(Y_train), eval_metric="auc", eval_set=[(X_test_1.drop(columns='RISK_SCORE'), np.ravel(Y_test))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train_1.drop(columns = 'RISK_SCORE'))[:, 1]
    ypred_test = bst.predict_proba(X_test_1.drop(columns = 'RISK_SCORE'))[:, 1]
else:
    ypred_train = bst.predict_proba(X_train_1.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test_1.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train_1.drop(columns = 'RISK_SCORE').columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_wo_hcscore_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)


print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))


    
pickle.dump(bst, open(os.getcwd()+"/xgb_fst_clx_30var_"+datetime.now().strftime("%d_%b_%H")+"hrs", 'wb'))

















########################################################################################
# Optimize hyperparameters using cross validation




X_train_3 = X_train[list(importance_tbl[importance_tbl["importance"]>=0.0105852755]["predictor"])+list(["RISK_SCORE"])]
X_test_3 = X_test[list(importance_tbl[importance_tbl["importance"]>=0.0105852755]["predictor"])+list(["RISK_SCORE"])]


param_grid = {'max_depth': np.arange(1,6),
              'learning_rate': np.arange(0.001, 0.02, 0.001),
              'subsample': np.arange(0.4, 0.7, 0.05),
              'min_child_weight': np.arange(1, 25, 3),
              'colsample_bytree': np.arange(0.5, 0.75, 0.05),
              'scale_pos_weight' :np.arange(5, 40, 2)
            }


if X_train_3.shape[1] == 1:
    param_grid['colsample_bytree'] = [1]

param_dist = {'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200}

xgb_model = xgb.XGBClassifier(**param_dist)

kfolds = StratifiedKFold(6, random_state=1)
clf = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, scoring=['roc_auc'], n_iter=100,
                         random_state=40, cv=kfolds.split(X_train_3.drop(columns='RISK_SCORE'), Y_train), refit='roc_auc',error_score=0, return_train_score =True)


print("Randomized search..")
lg.write("Randomized search..")

search_time_start = time.time()
clf.fit(X_train_3.drop(columns='RISK_SCORE'), np.ravel(Y_train))
print("Randomized search time:", time.time() - search_time_start)
lg.write("\nTime taken :"+str(round(time.time()-search_time_start,2)))

# to check importance of different hyperparameters in xgboost
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

params = clf.best_params_   #get best parameters
dic=clf.cv_results_    #provides scoring results for each iteration, 50 in this case, mean values can be used 
tmp =  pd.DataFrame(dic["params"])


print("Best score: {}".format(clf.best_score_))    #best auc score as we took that for refit 
lg.write("\n"+ "Best score: {}".format(clf.best_score_))


print("Best params: ")
lg.write("\nBest params: ")

for param_name in sorted(params.keys()):
    print('%s: %r' % (param_name, params[param_name]))
    lg.write('\n %s: %r' % (param_name, params[param_name]))


#update estimated hyperparameter values to new model
params.update(param_dist)

params= {
        "subsample": 0.5,
        "scale_pos_weight": 7,	
        "min_child_weight" : 7,
        "max_depth" :	2,
        "learning_rate": 0.011,
        "sample_bytree" : 0.65,
        'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200
        }

params= {
        "subsample": 0.65,
        "scale_pos_weight": 5,	
        "min_child_weight" : 22,
        "max_depth" :	3,
        "learning_rate": 0.007,
        "sample_bytree" : 0.5,
        'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200
        }


xgb_model = xgb.XGBClassifier(**params)



# cross validation gini
auc_cv = cross_val_score(xgb_model, X_train_3.drop(columns = 'RISK_SCORE'), Y_train, scoring='roc_auc', cv=kfolds.split(X_train_3.drop(columns='RISK_SCORE'), Y_train),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()


bst = xgb_model.fit(X_train_3.drop(columns='RISK_SCORE'), np.ravel(Y_train), eval_metric="auc", eval_set=[(X_test_3.drop(columns='RISK_SCORE'), np.ravel(Y_test))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train_3.drop(columns = 'RISK_SCORE'))[:, 1]
    ypred_test = bst.predict_proba(X_test_3.drop(columns = 'RISK_SCORE'))[:, 1]
else:
    ypred_train = bst.predict_proba(X_train_3.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test_3.drop(columns = 'RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train_3.drop(columns = 'RISK_SCORE').columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_wo_hcscore_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)


print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))




## save the model 
    
pickle.dump(bst, open(os.getcwd()+"/xgb_fst_clx_30var_bst_14%gini_"+datetime.now().strftime("%d_%b_%H")+"hrs", 'wb'))

outtest=X_test;  outtest['pred']=ypred_test ; outtest['act']=Y_test ; 
outtest.to_csv(os.getcwd()+"/xgb_fst_testout_clx_wo_hcscore_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")



#########metrics calculation

#this can be corrected, cutoff can be taken basis max f1 score, accuracy (at worst)
ypred=pd.Series(ypred_test).apply(lambda x : 1 if x >np.percentile(ypred_test,95) else 0 )

metrics.confusion_matrix(Y_test, ypred)
pd.Series(ypred_test).plot.hist(bins=10, rwidth=0.9 ,color='#607c8e')

print("\n Log Loss is :",metrics.log_loss(Y_test, ypred_test))
print('Average precision-recall score: {0:0.2f}'.format(metrics.average_precision_score(Y_test, ypred_test)))
lg.write("\n Log Loss is :"+str(metrics.log_loss(Y_test, ypred_test)))
lg.write('\n Average precision-recall score: {0:0.2f}'.format(metrics.average_precision_score(Y_test, ypred_test)))








## PRC curve
precision, recall, _ = metrics.precision_recall_curve(Y_test, ypred_test)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 0.2])
plt.xlim([0.0, 0.2])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(metrics.average_precision_score(Y_test, ypred_test)))


# retrieve performance metrics
results = bst.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)


# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')

# Compute micro-average ROC curve and ROC area
fpr, tpr, thresholds = metrics.roc_curve(Y_test.ravel(), ypred_test.ravel())
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 1
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.ix[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])

optimal_idx = np.argmax(tpr - fpr)  ## check it once 
optimal_threshold = thresholds[optimal_idx]
print("\n Optimal Cut off :",optimal_threshold)

## fecth original test data
test=X_test
test['RISK_FSTPD30']=Y_test

## for HC_SC distri
frq, edges = np.histogram(test['RISK_SCORE'])
test['bins']=pd.cut(test['RISK_SCORE'],bins=edges,labels=edges[:-1])
out2=pd.DataFrame(test.groupby('bins')['RISK_FSTPD30'].sum())
out2['count']=test['bins'].value_counts()
out2['event rate']=out2['RISK_FSTPD30']/out2['count']

## PayTM new model 
frq, edges = np.histogram(outtest['pred'],bins=20)
outtest['bins']=pd.cut(pd.Series(outtest['pred']),bins=edges,labels=edges[:-1])
out2=pd.DataFrame(outtest.groupby('bins')['act'].sum())
out2['count']=outtest['bins'].value_counts()
out2['event rate']=out2['act']/out2['count']

##lift calculations 
cd=pd.DataFrame(test[['RISK_SCORE','RISK_FSTPD30']]).sort_values(by=['RISK_SCORE'],ascending=False)
#cd.to_csv("cd_hcscore.csv",index=0)
cd.groupby(pd.qcut(cd['RISK_SCORE'],10,duplicates='drop')).mean()
cd.groupby(pd.qcut(cd['RISK_SCORE'],10,duplicates='drop')).sum()
cd['RISK_FSTPD30'].sum()

cd=pd.DataFrame(outtest[['pred','act']]).sort_values(by=['pred'],ascending=False)
#cd.to_csv("cd_hcscore.csv",index=0)
cd.groupby(pd.qcut(cd['pred'],10,duplicates='drop')).mean()
cd.groupby(pd.qcut(cd['pred'],10,duplicates='drop')).sum()
Y_test.sum()



############################################################################# combined model################


seed = 10
test_size = 0.2
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size, 
                                                 random_state = seed, stratify= df['DECISION_DATE'].dt.month)



bst = xgb.XGBClassifier()  # init model

bst.load_model(r'C:\Users\aman.khatri91425\Desktop\HomeCredit\Bigdata\PayU CLX Scoring\xgb_fst_clx_wo_hcscore_27_Sep_09hrs_final.model')

bst.predict_proba(X_train.drop(columns = 'RISK_SCORE'))
bst.feature_names

X_train["RISK_SCORE"]

len(bst.feature_importances_)
X_train.shape

#combined model
#XGBoost

X_train_1 = pd.DataFrame(X_train['RISK_SCORE'])
X_train_1['PAYU_SCORE'] = ypred_train

X_test_1 = pd.DataFrame(X_test['RISK_SCORE'])
X_test_1['PAYU_SCORE'] = ypred_test


param_grid = {'max_depth': np.arange(1,7),
              'learning_rate': np.arange(0.001, 0.02, 0.001),
              'subsample': [0.7,0.6,0.5],
              'min_child_weight': np.arange(1, 25, 3),
              'colsample_bytree': [0.6,0.65,0.7,0.75,0.8]
              ,'scale_pos_weight' :np.arange(7, 25, 2)
            }

if X_train.shape[1] == 1:
    param_grid['colsample_bytree'] = [1]

param_dist = {'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200}

xgb_model = xgb.XGBClassifier(**param_dist)

kfolds = StratifiedKFold(4, random_state=1)
clf = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, scoring=['roc_auc','neg_log_loss','average_precision','f1_macro'], n_iter=30,
                         random_state=40, cv=kfolds.split(X_train_1, Y_train), refit='roc_auc',error_score=0, return_train_score =True)


print("Randomized search..")

search_time_start = time.time()
clf.fit(X_train_1, np.ravel(Y_train))
print("Randomized search time:", time.time() - search_time_start)



params = clf.best_params_   #get best parameters
dic=clf.cv_results_    #provides scoring results for each iteration, 50 in this case, mean values can be used 


print("Best score: {}".format(clf.best_score_))    #best auc score as we took that for refit 


print("Best params: ")

for param_name in sorted(params.keys()):
    print('%s: %r' % (param_name, params[param_name]))


#update estimated hyperparameter values to new model
params.update(param_dist)
xgb_model = xgb.XGBClassifier(**params)


# cross validation gini
auc_cv = cross_val_score(xgb_model, X_train_1, Y_train, scoring='roc_auc', cv=kfolds.split(X_train_1, Y_train),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()


bst = xgb_model.fit(X_train_1, np.ravel(Y_train), eval_metric="auc", eval_set=[(X_test_1, np.ravel(Y_test))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train_1)[:, 1]
    ypred_test = bst.predict_proba(X_test_1)[:, 1]
else:
    ypred_train = bst.predict_proba(X_train_1, ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test_1, ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train_1.columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_wo_hcscore_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)


print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )



lg.write("\n GINI TRAIN(combined model):" + str(gini_train))
lg.write("\n GINI TEST(combined model):" + str(gini_test))
lg.write("\n GINI CV(combined model):" + str(gini_cv))




## save the model 
bst.save_model(os.getcwd()+"/xgb_fst_clx_combined_"+datetime.now().strftime("%d_%b_%H")+"hrs.model")
outtest=X_test;  outtest['pred']=ypred_test ; outtest['act']=Y_test ; 
outtest.to_csv(os.getcwd()+"/xgb_fst_testout_combined_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")



