# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:29:59 2019

@author: aman.khatri91425
"""
import os
import cx_Oracle
import pandas as pd
import numpy as np
from datetime import date,datetime
connection = cx_Oracle.connect('HCIN_GUPTAN1[AP_UWI]/aGqQ4jnX92NG@INCL02.IN.PROD/HWIN_USR_DEV.HOMECREDIT.IN')
cursor = connection.cursor()

os.chdir(r"C:\Users\aman.khatri91425\Desktop\HomeCredit\Bigdata\PayU CLX Scoring")
qu1= """
select * from tbd_payu_clx_model_base_1"""

df=pd.read_sql(qu1,connection)

df.to_csv(os.getcwd()+"\\Data Files\\sql_data.csv")
df.to_pickle(os.getcwd()+"\\Data Files\\sql_data.pkl")


#df.to_csv(os.getcwd()+"\\Data Files\\test"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")
#df.to_pickle(os.getcwd()+"\\Data Files\\test"+datetime.now().strftime("%d_%b_%H")+"hrs.pkl")

df.drop(columns = ['COHORT', 'PRODUCT_GROUP', 'RESULT', 'ACTIVITY',
       'ACTIVITY_PAYTM_DEF', 'CLIENT_EXI_3M', 'NTC_FLAG'], inplace = True)


for y in [x for x in df.columns if "date" in x.lower()]:
    df[y] = df[y].replace({'01-JAN-01':np.nan})
    df[y] = pd.to_datetime(df[y])

df = df.replace({'-9999':np.nan,-9999:np.nan})

#convert date variables to tenure variables 
for z in [x for x in df.columns if "date" in x.lower()]:
    df[z.replace("DATE","TENURE")]=df['DECISION_DATE'].sub(df[z], axis=0).dt.days


#drop date variables
date_columns = [x for x in df.columns if "date" in x.lower()]
date_columns.remove("DECISION_DATE")
df = df.drop(columns=date_columns)
df = df.drop(columns = "DECISION_TENURE")


#check columns with data type other than int
df.select_dtypes(exclude = ['int64','float64']).columns


#drop additional columns....keep fstqpd30 as risk flag
df= df[df["RISK_AGRF90"]==1]

df = df.drop(columns = ['NETBANK_AMT_RFND_90', 'AIRLINES_CNT_SALE_180', 'SKP_CREDIT_CASE',
       'SKP_CLIENT', 'RISK_FPD30', 'RISK_FSPD30', 'RISK_FSTQPD30',
       'RISK_AGRF30', 'RISK_AGRF60', 'RISK_AGRF90','RISK_AGRF120'])

df.to_csv(os.getcwd()+"\\Data Files\\payu_tenure_predictors_fstpd30_"+datetime.now().strftime("%d_%b_%H")+"hrs.csv")
df.to_pickle(os.getcwd()+"\\Data Files\\payu_tenure_predictors_fstpd30_"+datetime.now().strftime("%d_%b_%H")+"hrs.pkl")





