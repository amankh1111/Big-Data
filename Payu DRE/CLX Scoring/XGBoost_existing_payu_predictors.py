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
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score 
from inspect import signature
import math
import json

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
df = df[((df["DECISION_DATE"].dt.month != 1) & (df["DECISION_DATE"].dt.month != 6))]


# check distribution of HC score
df["RISK_SCORE"].plot.hist(bins=50, range=(0,0.04) ,color='#607c8e')


# Description of variables
ds=df.describe().transpose()
ds.to_csv(os.getcwd()+"\eda_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")

  
# data preparation         
X=df[['AFFLUENCE_SCORE','ENGAGEMENT_SCORE','DIGITAL_RISK_SCORE','ACTIVE_MONTHS_365',
      'TOTAL_AMT_SALE_365','TOTAL_CNT_SALE_365','RISK_SCORE']]
Y=df['RISK_FSTPD30']


#make group of variables, in line with data received from rcm connections

X['ACTIVE_MONTHS_365_BUC'] = X['ACTIVE_MONTHS_365'].apply(lambda symbol: 
 '0' if math.isnan(symbol) or symbol == 0
 else '1-3' if symbol> 0 and symbol<=3
 else '4-6' if symbol> 3 and symbol<=6
 else '7-9' if symbol> 6 and symbol<=9 
 else '>9')

X['TOTAL_AMT_SALE_365_BUC'] = X['TOTAL_AMT_SALE_365'].apply(lambda symbol: 
 '0' if math.isnan(symbol) or symbol == 0
 else '1-1000' if symbol> 0 and symbol<=1000
 else '1001-5000' if symbol> 1000 and symbol<=5000
 else '5001-10000' if symbol> 5000 and symbol<=10000 
 else '>10000')

X['TOTAL_CNT_SALE_365_BUC'] = X['TOTAL_CNT_SALE_365'].apply(lambda symbol: 
 '0' if math.isnan(symbol) or symbol == 0
 else '1-3' if symbol> 0 and symbol<=3
 else '4-6' if symbol> 3 and symbol<=6
 else '7-10' if symbol> 6 and symbol<=10 
 else '>10')


X['ACTIVE_MONTHS_365_ORD'] = X['ACTIVE_MONTHS_365'].apply(lambda symbol: 
 0 if math.isnan(symbol) or symbol == 0
 else 2 if symbol> 0 and symbol<=3
 else 5 if symbol> 3 and symbol<=6
 else 8 if symbol> 6 and symbol<=9 
 else 11)

X['TOTAL_AMT_SALE_365_ORD'] = X['TOTAL_AMT_SALE_365'].apply(lambda symbol: 
 0 if math.isnan(symbol) or symbol == 0
 else 500 if symbol> 0 and symbol<=1000
 else 3000 if symbol> 1000 and symbol<=5000
 else 7500 if symbol> 5000 and symbol<=10000 
 else 35000)

X['TOTAL_CNT_SALE_365_ORD'] = X['TOTAL_CNT_SALE_365'].apply(lambda symbol: 
 0 if math.isnan(symbol) or symbol == 0
 else 2 if symbol> 0 and symbol<=3
 else 5 if symbol> 3 and symbol<=6
 else 8.5 if symbol> 6 and symbol<=10 
 else 20)

    
X["AF_div_EN"]=X["AFFLUENCE_SCORE"]/X["ENGAGEMENT_SCORE"]
X["AF_div_DR"]=X["AFFLUENCE_SCORE"]/X["DIGITAL_RISK_SCORE"]
X["EN_div_AF"]=X["ENGAGEMENT_SCORE"]/X["AFFLUENCE_SCORE"]


X["EN_div_DR"]=X["ENGAGEMENT_SCORE"]/X["DIGITAL_RISK_SCORE"]
X["DR_div_AF"]=X["DIGITAL_RISK_SCORE"]/X["AFFLUENCE_SCORE"]
X["DR_div_EN"]=X["DIGITAL_RISK_SCORE"]/X["ENGAGEMENT_SCORE"]

X["AF_mul_EN"]=X["AFFLUENCE_SCORE"]*X["ENGAGEMENT_SCORE"]
X["EN_mul_DR"]=X["ENGAGEMENT_SCORE"]*X["DIGITAL_RISK_SCORE"]
X["DR_mul_AF"]=X["DIGITAL_RISK_SCORE"]*X["AFFLUENCE_SCORE"]
    




X['AVG_TICKET'] = X["TOTAL_AMT_SALE_365_ORD"]/X["TOTAL_CNT_SALE_365_ORD"]
X['MONTHLY_AMT'] = X["TOTAL_AMT_SALE_365_ORD"]/X["ACTIVE_MONTHS_365_ORD"]
X['MONTHLY_CNT'] = X["TOTAL_CNT_SALE_365_ORD"]/X["ACTIVE_MONTHS_365_ORD"]


    

# HCC gini score calculation
y_hcc = X["RISK_SCORE"]
df_hcc = pd.DataFrame({'true':Y,'predict':y_hcc})
metric_auc_hcc = metrics.roc_auc_score(df_hcc['true'], df_hcc['predict'])
gini_hcc_only = 2 * metric_auc_hcc - 1

# For generating logs
lg=open(os.getcwd()+"\log_xgb_clx_current_pred.txt","a")   
print("Home Credit Only GINI:", '{:,.2%}'.format(gini_hcc_only))
lg.write("Home Credit Only GINI:"  + '{:,.2%}'.format(gini_hcc_only))


# define model parameters
missing_value=None; cpu=-1; test_size=0.3; early_stopping_round=40; split_random_state=None


#use ordinal values
X_final=X[['AFFLUENCE_SCORE','ENGAGEMENT_SCORE','DIGITAL_RISK_SCORE','ACTIVE_MONTHS_365_ORD',
      'TOTAL_AMT_SALE_365_ORD','TOTAL_CNT_SALE_365_ORD','RISK_SCORE','AVG_TICKET',
       'AF_mul_EN', 'EN_mul_DR', 'DR_mul_AF', 'MONTHLY_AMT', 'MONTHLY_CNT',
       'AF_div_EN', 'AF_div_DR', 'EN_div_AF', 'EN_div_DR', 'DR_div_AF',
       'DR_div_EN']]


#Split data into train and test set---Stratified sampling on month
seed = 15
test_size = 0.2
X_train,X_test,Y_train,Y_test = train_test_split(X_final,Y,test_size=test_size, 
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


# Optimize hyperparameters using cross validation

param_grid = {'max_depth': np.arange(1,6),
              'learning_rate': np.arange(0.001, 0.02, 0.001),
              'subsample': np.arange(0.4, 0.7, 0.05),
              'min_child_weight': np.arange(1, 25, 3),
              'colsample_bytree': np.arange(0.5, 0.75, 0.05),
              'scale_pos_weight' :np.arange(15, 40, 2)
            }



param_grid = {
              'learning_rate': np.arange(0.003, 0.01, 0.0005),
              'min_child_weight': np.arange(5, 20, 1),
              'scale_pos_weight' :np.arange(25, 40, 1)
            }



if X_train.shape[1] == 1:
    param_grid['colsample_bytree'] = [1]

param_dist = {'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200, "max_depth":1,
              "colsample_bytree": 0.65, "subsample": 0.55

}

xgb_model = xgb.XGBClassifier(**param_dist)

kfolds = StratifiedKFold(10, random_state=1)
clf = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, scoring=['roc_auc'], n_iter=100,
                         random_state=40, cv=kfolds.split(X_train.drop(columns ='RISK_SCORE'), Y_train), refit='roc_auc',error_score=0, return_train_score =True)


print("Randomized search..")
lg.write("Randomized search..")

search_time_start = time.time()
clf.fit(X_train.drop(columns ='RISK_SCORE'), np.ravel(Y_train))
print("Randomized search time:", time.time() - search_time_start)
lg.write("\nTime taken :"+str(round(time.time()-search_time_start,2)))

# to check importance of different hyperparameters in xgboost
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

params = clf.best_params_   #get best parameters
dic=clf.cv_results_    #provides scoring results for each iteration, 50 in this case, mean values can be used 
tmp= pd.DataFrame(dic['params'])

print("Best score: {}".format(clf.best_score_))    #best auc score as we took that for refit 
lg.write("\n"+ "Best score: {}".format(clf.best_score_))


print("Best params: ")
lg.write("\nBest params: ")

for param_name in sorted(params.keys()):
    print('%s: %r' % (param_name, params[param_name]))
    lg.write('\n %s: %r' % (param_name, params[param_name]))


#update estimated hyperparameter values to new model

params = {
        "colsample_bytree": 0.65,
        "learning_rate":	0.006,
        "max_depth": 1,
        "min_child_weight":18,	
        "scale_pos_weight":	37,
        "subsample":0.55,
              'objective    ': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200
            }


xgb_model = xgb.XGBClassifier(**params)


# cross validation gini
auc_cv = cross_val_score(xgb_model, X_train.drop(columns ='RISK_SCORE'), Y_train, scoring='roc_auc', cv=kfolds.split(X_train, Y_train),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()



bst = xgb_model.fit(X_train.drop(columns ='RISK_SCORE'), np.ravel(Y_train), eval_metric="auc", eval_set=[(X_test.drop(columns ='RISK_SCORE'), np.ravel(Y_test))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


bst = xgb_model.fit(X_train.drop(columns ='RISK_SCORE'), np.ravel(Y_train), verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train.drop(columns ='RISK_SCORE'))[:, 1]
    ypred_test = bst.predict_proba(X_test.drop(columns ='RISK_SCORE'))[:, 1]
else:
    ypred_train = bst.predict_proba(X_train.drop(columns ='RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test.drop(columns ='RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train.drop(columns ='RISK_SCORE').columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_6pred_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)

print(params)
print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write(json.dumps(params))
lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))



## save the model 
pickle.dump(bst, open(os.getcwd()+"/xgb_fst_clx_6pred_derived_bst1_"+datetime.now().strftime("%d_%b_%H")+"hrs.sav","wb"))
outtest=X_test;  outtest['pred']=ypred_test ; outtest['act']=Y_test ; 
outtest.to_csv(os.getcwd()+"/xgb_fst_testout_clx_6pred_derived_bst1"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")




##########metrics calculation

#this can be corrected, cutoff can be taken basis max f1 score, accuracy (at worst)
ypred=pd.Series(ypred_test).apply(lambda x : 1 if x >np.percentile(ypred_test,95) else 0 )

metrics.confusion_matrix(Y_test, ypred)
pd.Series(ypred_test).plot.hist(bins=10, rwidth=0.9 ,color='#607c8e')

print("\n Log Loss is :",metrics.log_loss(Y_test, ypred_test))
print('Average precision-recall score: {0:0.2f}'.format(metrics.average_precision_score(Y_test, ypred_test)))
lg.write("\n Log Loss is :"+str(metrics.log_loss(Y_test, ypred_test)))
lg.write('\n Average precision-recall score: {0:0.2f}'.format(metrics.average_precision_score(Y_test, ypred_test)))



pd.Series(ypred_train).value_counts()


X_test.drop(columns = ['act','pred'],inplace = True)

#Variable Selection
X_train_1 = X_train.drop(columns = 'RISK_SCORE')
X_test_1 = X_test.drop(columns = 'RISK_SCORE')


from sklearn.feature_selection import SelectFromModel


X_train_1 = X_train_1.replace({np.nan:0})
X_test_1 = X_test_1.replace({np.nan:0})


bst1 = bst
X_train_2 = X_train_1
X_test_2 = X_test_1
select_X_train = X_train_1
select_X_test = X_test_1

while select_X_train.shape[1]>1:
    # select features using threshold
    selection = SelectFromModel(bst1, threshold=list(importance_tbl['importance'])[-2], prefit=True)   
    select_X_train = pd.DataFrame(selection.transform(X_train_2))
    feature_idx = selection.get_support() 
    select_X_train.columns = X_train_2.columns[feature_idx]

    select_X_test = pd.DataFrame(selection.transform(X_test_2))
    select_X_test.columns = X_test_2.columns[feature_idx]
    # train model
    selection_model = xgb.XGBClassifier(**params) 
    auc_cv = cross_val_score(selection_model, select_X_train, Y_train, scoring='roc_auc', cv=kfolds.split(select_X_train, Y_train),
                         n_jobs=cpu)
    gini_cv = (2 * np.array(auc_cv) - 1).mean()
    bst1 = selection_model.fit(select_X_train, Y_train)
    importance_tbl = fimportances(bst1,select_X_train.columns)

    ypred_train = bst1.predict_proba(select_X_train)[:, 1]
    ypred_test = bst1.predict_proba(select_X_test)[:, 1]

    metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
    metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

    gini_train = 2 * metric_auc_train - 1
    gini_test = 2 * metric_auc_test - 1

    print("Thresh=%.5f, n=%d, gini_cv: %.2f%% variables: %s"% (list(importance_tbl['importance'])[-2], select_X_train.shape[1], gini_cv*100.0, ",".join(importance_tbl['predictor'])))
    lg.write("Thresh=%.5f, n=%d, gini_cv: %.2f%% variables: %s"% (list(importance_tbl['importance'])[-2], select_X_train.shape[1], gini_cv*100.0, ",".join(importance_tbl['predictor'])))
    print("\nGINI TRAIN:",gini_train )
    print("\nGINI TEST:",gini_test )

    lg.write("\n GINI TRAIN:" + str(gini_train))
    lg.write("\n GINI TEST:" + str(gini_test))

    X_train_2 = select_X_train
    X_test_2 = select_X_test






##########################Fit 2 Variable model######################



X_train_2 = X_train[["DIGITAL_RISK_SCORE","DR_div_AF","RISK_SCORE"]]
X_test_2 = X_test[["DIGITAL_RISK_SCORE","DR_div_AF","RISK_SCORE"]]


# Optimize hyperparameters using cross validation

param_grid = {'max_depth': np.arange(1,7),
              'learning_rate': np.arange(0.001, 0.02, 0.001),
              'subsample': [0.7,0.6,0.5],
              'min_child_weight': np.arange(1, 25, 3),
              'colsample_bytree': [0.6,0.65,0.7,0.75,0.8]
              ,'scale_pos_weight' :np.arange(7, 25, 2)
            }

if X_train_2.shape[1] == 1:
    param_grid['colsample_bytree'] = [1]

param_dist = {'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200}

xgb_model = xgb.XGBClassifier(**param_dist)

kfolds = StratifiedKFold(4, random_state=1)
clf = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, scoring=['roc_auc','neg_log_loss','average_precision','f1_macro'], n_iter=30,
                         random_state=40, cv=kfolds.split(X_train_2.drop(columns ='RISK_SCORE'), Y_train), refit='roc_auc',error_score=0, return_train_score =True)


print("Randomized search..")
lg.write("Randomized search..")

search_time_start = time.time()
clf.fit(X_train_2.drop(columns ='RISK_SCORE'), np.ravel(Y_train))
print("Randomized search time:", time.time() - search_time_start)
lg.write("\nTime taken :"+str(round(time.time()-search_time_start,2)))

# to check importance of different hyperparameters in xgboost
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

params = clf.best_params_   #get best parameters
dic=clf.cv_results_    #provides scoring results for each iteration, 50 in this case, mean values can be used 


print("Best score: {}".format(clf.best_score_))    #best auc score as we took that for refit 
lg.write("\n"+ "Best score: {}".format(clf.best_score_))


print("Best params: ")
lg.write("\nBest params: ")

for param_name in sorted(params.keys()):
    print('%s: %r' % (param_name, params[param_name]))
    lg.write('\n %s: %r' % (param_name, params[param_name]))


#update estimated hyperparameter values to new model

#params = {'max_depth': 3,
#              'learning_rate': 0.01,
#              'subsample': 0.5,
#              'min_child_weight': 11,
#              'colsample_bytree': 0.7,
#              'scale_pos_weight' :25,
#              'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200
#            }
#

params.update(param_dist)
xgb_model = xgb.XGBClassifier(**params)


# cross validation gini
auc_cv = cross_val_score(xgb_model, X_train_2.drop(columns ='RISK_SCORE'), Y_train, scoring='roc_auc', cv=kfolds.split(X_train_2.drop(columns='RISK_SCORE'), Y_train),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()



bst = xgb_model.fit(X_train_2.drop(columns ='RISK_SCORE'), np.ravel(Y_train), eval_metric="auc", eval_set=[(X_test_2.drop(columns ='RISK_SCORE'), np.ravel(Y_test))],
                    early_stopping_rounds=early_stopping_round,  verbose=True)


if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train_2.drop(columns ='RISK_SCORE'))[:, 1]
    ypred_test = bst.predict_proba(X_test_2.drop(columns ='RISK_SCORE'))[:, 1]
else:
    ypred_train = bst.predict_proba(X_train_2.drop(columns ='RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test_2.drop(columns ='RISK_SCORE'), ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(Y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(Y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1


importance_tbl = fimportances(bst, X_train_2.drop(columns ='RISK_SCORE').columns)
importance_tbl.to_csv(os.getcwd()+"/varimpt_xgb_fstpd_clx_6pred_2pred_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv",index=0)


print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))


## save the model 
bst.save_model(os.getcwd()+"/xgb_fst_clx_6pred_derived_2pred_"+datetime.now().strftime("%d_%b_%H")+"hrs.model")
outtest=X_test;  outtest['pred']=ypred_test ; outtest['act']=Y_test ; 
outtest.to_csv(os.getcwd()+"/xgb_fst_testout_clx_6pred_derived_2pred_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")




















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
bst.save_model(os.getcwd()+"/xgb_fst_clx_6pred_derived_2pred_combined_"+datetime.now().strftime("%d_%b_%H")+"hrs.model")
outtest=X_test;  outtest['pred']=ypred_test ; outtest['act']=Y_test ; 
outtest.to_csv(os.getcwd()+"/xgb_fst_testout_clx_6pred_derived_2pred_combined_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")


lg.close()






