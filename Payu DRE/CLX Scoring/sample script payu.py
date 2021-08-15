import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import math

os.chdir(r"C:\Users\aman.khatri91425\Desktop\HomeCredit\Bigdata\PayU CLX Scoring")
df=pd.read_csv(os.getcwd()+"\\testing_file.csv")

loaded_model = pickle.load(open("xgb_payu_31pred", 'rb'))

ypred = loaded_model.predict_proba(df.drop(columns ='PRED_SCORE'), ntree_limit=loaded_model.best_ntree_limit)[:, 1]

