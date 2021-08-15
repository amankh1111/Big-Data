# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:21:21 2019

@author: homecredit
"""

import os 
import pandas as pd 
import glob 
import time
import pickle
import datetime 
import numpy as np


os.chdir(r"E:\Data Processing\aman\dump")
df=pd.read_pickle("customer_summary_160819.pkl") 

hc = pd.read_pickle(r"E:\Data Processing\nikhil\dump\hcin_export_0808_40L_dedup_with_score.pkl")

hc.columns = ['SKP_CREDIT_CASE', 'SKP_CLIENT', 'PRODUCT_GROUP', 'mobile',
       'TIME_DECISION_DATE', 'RESULT', 'ACTIVITY', 'ACTIVITY_PAYTM_DEF',
       'RISK_FPD30', 'RISK_FSPD30', 'RISK_FSTPD30', 'RISK_FSTQPD30',
       'RISK_AGRF60', 'RISK_AGRF90', 'RISK_AGRF120', 'CLIENT_EXI_3M','HCIMODELSCORE']

df = df.merge(hc, on = 'mobile',how = 'inner')
df.to_pickle(r"E:\Data Processing\aman\dump\customer_pred_scores_1608.pkl")
df.to_csv(r"E:\Data Processing\aman\dump\customer_pred_scores_1608.csv")



#df.groupby('PRODUCT_GROUP')['mobile'].count()

#for x in df.columns:
#    try:
#        y=df.loc[~np.isfinite(df[x]),x].count()
#    except: 
#        y = 0
#    
#    if y>0:
#        print(x,y)
        
        
