# -*- coding: utf-8 -*-

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

##read datasets
df=pd.read_pickle(r"E:\Data Processing\aman\dump\xgb_raw_data_1608.pkl")
#df=pd.read_pickle(r"E:\Data Processing\aman\dump\customer_pred_scores_1908.pkl")

#df.groupby('PRODUCT_GROUP')['mobile'].count()
df['PRODUCT_GROUP'].value_counts()

df=df[(df['RISK_AGRF120']==1) & (df['PRODUCT_GROUP']=='CD')]

#Event Rate
#df['RISK_FSTQPD30'].sum()/df['PRODUCT_GROUP'].count()

#df['RISK_FSTPD30'].sum()/df['PRODUCT_GROUP'].count()


#[['RISK_FSTPD30']].sum()

df['TIME_DECISION_DATE'] = pd.to_datetime(df['TIME_DECISION_DATE'])
df.dropna(subset=['HCIMODELSCORE'],inplace=True)
df['month']=df['TIME_DECISION_DATE'].dt.month
df=df[df['month']!=4]

# Check distribution of scores
#pd.Series(df['HC_SC']).plot.hist(bins=10, range=(0,0.08) ,color='#607c8e')
#pd.Series(df['HCIMODELSCORE']).plot.hist(bins=10, rwidth=0.9 ,color='#607c8e')

#pred=pd.read_csv('E:/Data Processing/Aman/dump/varimpt_xgb_1908_fstq.csv')['predictor']

## data preparation         
X=df.drop(['CLIENT_EXI_3M' ,'RISK_FSTPD30', 'RISK_AGRF120' , 'RISK_AGRF90' , 'RISK_AGRF60' , 'RISK_FSTQPD30' , 'RISK_FSPD30' , 'RISK_FPD30' , 'ACTIVITY_PAYTM_DEF' , 'ACTIVITY' , 'RESULT' , 'TIME_DECISION_DATE' , 'PRODUCT_GROUP' , 'SKP_CLIENT' , 'SKP_CREDIT_CASE' , 'decision_date' , 'TIME_CREATION_DATE_max' , 'bill_date_min' , 'bill_date_max' ] ,axis=1)
Y=df[['RISK_FSTQPD30']]

# X=X[['tot_bill_amt_180', 'avg_bill_amt_180', 'med_bill_amt_180','max_bill_amt_180','std_bill_amt_180','spread_bill_amt_180','HCIMODELSCORE']]      
hcc_score='HCIMODELSCORE'

# HCC score Gini         
y_hcc = X[hcc_score]
df_hcc = pd.DataFrame({'true': np.ravel(Y), 'predict': y_hcc})
df_hcc = df_hcc.loc[pd.notnull(df_hcc['predict']),:]
metric_auc_hcc = metrics.roc_auc_score(df_hcc['true'], df_hcc['predict'])
gini_hcc_only = 2 * metric_auc_hcc - 1

lg=open(r"E:\Data Processing\aman\log_xgb190819_FST_POS.txt","a")   
print("HCIN SCORE GINI:" ,gini_hcc_only)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify =df['month'])

X_train.drop(['month'],axis=1,inplace=True) 
X_test.drop(['month'],axis=1,inplace=True) 

#X_train, X_test, y_train, y_test = train_test_split(X[pred], Y, test_size=test_size, random_state=seed)

##event rate
print("\nEvent rate in train set :", '{:,.2%}'.format(sum(y_train.iloc[:,0]) / len(y_train.iloc[:,0])))
lg.write("\nEvent rate in train set :" + '{:,.2%}'.format(sum(y_train.iloc[:,0]) / len(y_train.iloc[:,0])))

print("\nEvent rate in test set :", '{:,.2%}'.format(sum(y_test.iloc[:,0]) / len(y_test.iloc[:,0])))
lg.write("\nEvent rate in test set :"+ '{:,.2%}'.format(sum(y_test.iloc[:,0]) / len(y_test.iloc[:,0])))


# set model parameters 
missing_value=None; cpu=-1; test_size=0.3; early_stopping_round=40; split_random_state=None

## Variable importance
def fimportances(xc, predictors):
    importances = pd.DataFrame({'predictor': predictors, 'importance': xc.feature_importances_})
    importances = importances[importances['importance'] > 0]
    importances.sort_values(by='importance', ascending=False, inplace=True)
    importances.reset_index(inplace=True, drop=True)
    importances = importances[['predictor', 'importance']]
    return importances

## XGB model##
#def model_result(X_train, X_test, y_train, y_test):
param_grid = {'max_depth': np.arange(3, 6),
              'learning_rate': np.arange(0.01, 0.11, 0.01),
              'subsample': [0.5, 0.7],
              'min_child_weight': np.arange(1, 50, 10),
              'colsample_bytree': [0.5, 0.6, 0.7]
              ,'scale_pos_weight' :np.arange(5, 50, 5)
            }

if X_train.shape[1] == 1:
    param_grid['colsample_bytree'] = [1]
    
param_dist = {'objective': 'binary:logistic', 'nthread': cpu, 'n_estimators': 200}
xgb_model = xgb.XGBClassifier(**param_dist)

##random search XGB    
kfolds = StratifiedKFold(4, random_state=1)
clf = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, scoring=['roc_auc','neg_log_loss','average_precision','f1_macro'], n_iter=50,
                         random_state=40, cv=kfolds.split(X_train, y_train), refit='roc_auc',error_score=0, return_train_score =True)
print("Randomized search..")
lg.write("Randomized search..")

search_time_start = time.time()
clf.fit(X_train, np.ravel(y_train))
print("Randomized search time:", time.time() - search_time_start)
lg.write("\nTime taken :"+str(round(time.time()-search_time_start,2)))

params = clf.best_params_
dic=clf.cv_results_

params_bst = params
params_tbl = pd.DataFrame(columns=list(params_bst.keys()))  
params_tbl.loc['0'] = [params_bst[k] for k in params_tbl.columns]

print("Best score: {}".format(clf.best_score_))
lg.write("\n"+ "Best score: {}".format(clf.best_score_))

print("Best params: ")
lg.write("\nBest params: ")
for param_name in sorted(params_bst.keys()):
    print('%s: %r' % (param_name, params_bst[param_name]))
    lg.write('\n %s: %r' % (param_name, params_bst[param_name]))

params.update(param_dist)
xgb_model = xgb.XGBClassifier(**params)

# cross validation gini
auc_cv = cross_val_score(xgb_model, X_train, y_train, scoring='roc_auc', cv=kfolds.split(X_train, y_train),
                         n_jobs=cpu)
gini_cv = (2 * np.array(auc_cv) - 1).mean()

bst = xgb_model.fit(X_train, np.ravel(y_train), eval_metric="auc", eval_set=[(X_test, np.ravel(y_test))],
                    early_stopping_rounds=early_stopping_round   ,  verbose=True)
bst

if early_stopping_round is None:
    ypred_train = bst.predict_proba(X_train)[:, 1]
    ypred_test = bst.predict_proba(X_test)[:, 1]
else:
    ypred_train = bst.predict_proba(X_train, ntree_limit=bst.best_ntree_limit)[:, 1]
    ypred_test = bst.predict_proba(X_test, ntree_limit=bst.best_ntree_limit)[:, 1]

metric_auc_train = metrics.roc_auc_score(y_train, ypred_train)
metric_auc_test = metrics.roc_auc_score(y_test, ypred_test)

gini_train = 2 * metric_auc_train - 1
gini_test = 2 * metric_auc_test - 1

importance_tbl = fimportances(bst, X_train.columns)
importance_tbl.to_csv('E:/Data Processing/Aman/dump/varimpt_xgb_1908_fst_pos.csv',index=0)

print("GINI TRAIN:",gini_train )
print("\nGINI TEST:",gini_test )
print("\nGINI CV:",gini_cv )

lg.write("\n GINI TRAIN:" + str(gini_train))
lg.write("\n GINI TEST:" + str(gini_test))
lg.write("\n GINI CV:" + str(gini_cv))

## save the model 
bst.save_model('E:/Data Processing/Aman/dump/xgb_1908_fst_pos.model')
outtest=X_test;  outtest['pred']=ypred_test ; outtest['act']=y_test ; 
outtest.to_csv('E:/Data Processing/Aman/dump/xgb_1908_fst_testout_pos.csv')

#########
#metrics calculation
ypred=pd.Series(ypred_test).apply(lambda x : 1 if x >np.percentile(ypred_test,95) else 0 )

confusion_matrix(y_test, ypred)
pd.Series(ypred_test).plot.hist(bins=10, rwidth=0.9 ,color='#607c8e')

print("\n Log Loss is :",log_loss(y_test, ypred_test))
print('Average precision-recall score: {0:0.2f}'.format(average_precision_score(y_test, ypred_test)))
lg.write("\n Log Loss is :"+str(log_loss(y_test, ypred_test)))
lg.write('\n Average precision-recall score: {0:0.2f}'.format(average_precision_score(y_test, ypred_test)))

lg.close()


## PRC curve
precision, recall, _ = precision_recall_curve(y_test, ypred_test)

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
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision_score(y_test, ypred_test)))


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
fpr, tpr, thresholds = roc_curve(y_test.iloc[:, 0].ravel(), ypred_test.ravel())
roc_auc = auc(fpr, tpr)

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
test['RISK_FSTQPD30']=y_test.iloc[:,0]

## for HC_SC distri
frq, edges = np.histogram(test['HCIMODELSCORE'])
test['bins']=pd.cut(test['HCIMODELSCORE'],bins=edges,labels=edges[:-1])
out2=pd.DataFrame(test.groupby('bins')['RISK_FSTQPD30'].sum())
out2['count']=test['bins'].value_counts()
out2['event rate']=out2['RISK_FSTQPD30']/out2['count']

## for HCI_MODEL old distri
frq, edges = np.histogram(test['HCIMODELSCORE'])
test['bins']=pd.cut(test['HCIMODELSCORE'],bins=edges,labels=edges[:-1])
out2=pd.DataFrame(test.groupby('bins')['N4PD30'].sum())
out2['count']=test['bins'].value_counts()
out2['event rate']=out2['N4PD30']/out2['count']

## PayTM new model 
frq, edges = np.histogram(outtest['pred'],bins=20)
outtest['bins']=pd.cut(pd.Series(outtest['pred']),bins=edges,labels=edges[:-1])
out2=pd.DataFrame(outtest.groupby('bins')['act'].sum())
out2['count']=outtest['bins'].value_counts()
out2['event rate']=out2['act']/out2['count']

##lift calculations 
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

ds=X_train.describe().transpose()


df1 = pd.read_pickle(r"E:\Data Processing\aman\dump\xgb_raw_data_1608.pkl")
df1.count().to_csv(r"E:/Data Processing/Aman/dump/fill_rate_1908.csv")
