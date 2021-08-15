# -*- coding: utf-8 -*-

import os 
import pandas as pd 
import glob 
import time
import pickle

#hc=pd.read_pickle(r"E:\Data Processing\nikhil\dump\hcin_export_0808_45L.pkl")
#hc['TEXT_MOBILE']=hc['TEXT_MOBILE'].apply( lambda x: str(x).strip()[-10:])
#hc = hc.sort_values('TIME_DECISION_DATE').drop_duplicates('TEXT_MOBILE',keep= 'last')
#hc.to_pickle(r"E:\Data Processing\nikhil\dump\hcin_export_0808_40L_dedup.pkl")

def main():
    os.chdir(r"E:\VMM data -2016-19\0908\All Data")
    lg=open(r"E:\Data Processing\aman\log_1308.txt","a")
    hc=pd.read_pickle(r"E:\Data Processing\nikhil\dump\hcin_export_0808_40L_dedup_with_score.pkl")[['TEXT_MOBILE','SKP_CREDIT_CASE','TIME_DECISION_DATE']]
    hc['TEXT_MOBILE']=hc['TEXT_MOBILE'].apply( lambda x: str(x).strip()[-10:])      
    hc.columns = ['mobile', 'SKP_CREDIT_CASE', 'TIME_CREATION_DATE']

    for fl in ('transactions_1308_2016','transactions_1308_2017','transactions_1308_2018','transactions_1308_2019','transactions_2016','transactions_2017','transactions_2018','transactions_2019','transactions_2017_1','transactions_2018_1','transactions_2019_1'):
        start=time.time()
        df=pd.DataFrame() ;        des=pd.DataFrame()
        print("\n File read: ",fl)
        file=glob.glob(fl+"\*.csv")
        df=pd.read_csv(file[0],low_memory=False)
        df['mobile']=df['mobile'].apply(lambda x: str(x).strip()[-10:])

        df = df.merge(hc, on = 'mobile',how = 'inner')

        df['bill_date']=pd.to_datetime(df['bill_date'])
        df['TIME_CREATION_DATE']=pd.to_datetime(df['TIME_CREATION_DATE'])

        df =  df[(df['bill_date']<= df['TIME_CREATION_DATE'])]
        
        lg.write("\nFile merge :"+fl+"\nTime taken :"+str(round(time.time()-start,2))+"\n Dimensions : "+str(df.shape))
        des=df.describe(include='all').transpose()

        des.to_csv(r'E:\Data Processing\aman\dump\\'+fl+"_(des)_1308.csv")        
        df.to_pickle(r'E:\Data Processing\aman\dump\\'+fl+"_(NI+OLP)_1308.pkl")
       
    del df
    dx=pd.DataFrame()
    files=glob.glob(r'E:\Data Processing\aman\dump\\*_(NI+OLP)_1308.pkl')
    for f in files:
          tmp=pd.read_pickle(f)
          dx=pd.concat([dx,tmp])
          
    dx.to_pickle(r'E:\Data Processing\aman\dump\ALL_NI_OLP_1308.pkl')
    lg.write("\nPocess complete"+"\t Dimensions : "+str(dx.shape))
    dsc=dx.describe(include='all').transpose()
    dsc.to_csv(r'E:\Data Processing\aman\dump\ALL_NI_OLP_desc_1308.csv')
    lg.close()            
    return    

main()

###add flags 