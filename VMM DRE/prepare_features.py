# -*- coding: utf-8 -*-

import os 
import pandas as pd 
import glob 
import time
import pickle
import datetime 
import numpy as np

os.chdir(r"E:\Data Processing\aman\dump")
lg=open(r"E:\Data Processing\aman\log_prepare_features.txt","a")
df=pd.read_pickle('ALL_NI_OLP_1308.pkl')
df.drop_duplicates(inplace=True)
df.reset_index(inplace= True, drop = True)
df['zero_unit_price_flag'] = np.where(df['bill_am']==0,1,0)
df= df[df['mobile']!='9999999999']
#df['weekend']=df['bill_date'].apply(lambda x: 1 if x.weekday()==5 or x.weekday()==6  else 0)
#df.columns
###bill details
bill=df.groupby(['bill_number','mobile','billing_store_code']).agg({
        'bill_am': ['sum'],
        'item_code':['nunique'],
        'quantity': ['sum'],
        'bill_date' :['max'],
        'TIME_CREATION_DATE' : ['max'],
        'zero_unit_price_flag' :['max']
        }) 


bill.columns=["_".join(x) for x in bill.columns.ravel()]
bill = bill.reset_index()
bill['weekday']=bill['bill_date_max'].dt.weekday 
bill['week']=bill['bill_date_max'].dt.week
bill['year']=bill['bill_date_max'].dt.year ;
bill['month']=bill['bill_date_max'].dt.month; 
bill['day'] = bill['bill_date_max'].dt.day
bill['weekend_flag']=np.where(bill['weekday'].isin([5,6]),1,0)
bill['month_1_10day_flag'] = np.where(bill['day'] <= 10,1,0)
bill['month_11_20day_flag'] = np.where((bill['day'] > 10) & (bill['day'] <= 20),1,0)
bill['month_21_31day_flag'] = np.where(bill['day'] > 20,1,0)


try:
    bill['avg_item_unit_price'] = bill['bill_am_sum']/bill['quantity_sum']
except:
    print("unable to divide for avg_item_unit_price")
    lg.write("unable to divide for avg_item_unit_price")

##customer details
cust=df.groupby('mobile').agg({'bill_number': ['nunique'],
                                 'bill_date' : ['max','min','nunique'],
                                 'TIME_CREATION_DATE' : ['max'],
                                 'bill_am' : ['sum']})

cust.columns=["_".join(x) for x in cust.columns.ravel() ]

#cust['decision_date']=datetime.datetime.today() ; df['decision_date']=datetime.datetime.today() ; bill['decision_date']=datetime.datetime.today() 

##to calculate predictors as on decision date
#cust['decision_date']=np.NaN 

cust['decision_date']=cust['TIME_CREATION_DATE_max'] ;
bill['decision_date']=bill['TIME_CREATION_DATE_max']

cust['days_since_fsttxn']=(cust['decision_date']-cust['bill_date_min']).dt.days
cust['days_since_lsttxn']=(cust['decision_date']-cust['bill_date_max']).dt.days
cust['1txn_only_ever']= (cust['days_since_fsttxn']==cust['days_since_lsttxn'])

vntg=[30,60,90,180,365]
for v in vntg:
    cust['tot_bill_amt_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['bill_am_sum'].sum()
    cust['avg_bill_amt_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['bill_am_sum'].mean()
    cust['med_bill_amt_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['bill_am_sum'].median()
    cust['max_bill_amt_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['bill_am_sum'].max()
    cust['min_bill_amt_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['bill_am_sum'].min()
    cust['std_bill_amt_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['bill_am_sum'].std()
    try:
        cust['spread_bill_amt_'+str(v)] = cust['std_bill_amt_'+str(v)]/cust['avg_bill_amt_'+str(v)]
    except:
        print("unable to divide for spread_bill_amt")
        lg.write("unable to divide for spread_bill_amt")
        cust['spread_bill_amt_'+str(v)]=np.NaN

    cust['tot_bill_items_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['item_code_nunique'].sum()
    cust['avg_bill_items_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['item_code_nunique'].mean()
    cust['med_bill_items_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['item_code_nunique'].median()
    cust['max_bill_items_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['item_code_nunique'].max()
    cust['min_bill_items_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['item_code_nunique'].min()
    cust['std_bill_items_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['item_code_nunique'].std()
    try:
        cust['spread_bill_items_'+str(v)] = cust['std_bill_items_'+str(v)]/cust['avg_bill_items_'+str(v)]
    except:
        print("unable to divide for spread_bill_items")
        lg.write("unable to divide for spread_bill_items")
        cust['spread_bill_items_'+str(v)]=np.NaN
    
    cust['tot_bill_qty_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['quantity_sum'].sum()
    cust['avg_bill_qty_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['quantity_sum'].mean()
    cust['med_bill_qty_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['quantity_sum'].median()
    cust['max_bill_qty_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['quantity_sum'].max()
    cust['min_bill_qty_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['quantity_sum'].min()
    cust['std_bill_qty_'+str(v)]=bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['quantity_sum'].std()
    try:
        cust['spread_bill_qty_'+str(v)] = cust['std_bill_qty_'+str(v)]/cust['avg_bill_qty_'+str(v)]
    except:
        print("unable to divide for spread_bill_qty")
        lg.write("unable to divide for spread_bill_qty")
        cust['spread_bill_qty_'+str(v)]=np.NaN

    cust['avg_bill_item_unit_price_'+str(v)] = cust['tot_bill_amt_'+str(v)]/cust['tot_bill_qty_'+str(v)]
    cust['max_bill_item_unit_price_'+str(v)] = bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['avg_item_unit_price'].max()
    cust['med_bill_item_unit_price_'+str(v)] = bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['avg_item_unit_price'].median()
    cust['min_bill_item_unit_price_'+str(v)] = bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['avg_item_unit_price'].min()
    cust['std_bill_item_unit_price_'+str(v)] = bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['avg_item_unit_price'].std()
    try:
        cust['spread_bill_item_unit_price_'+str(v)] = cust['std_bill_item_unit_price_'+str(v)]/cust['avg_bill_item_unit_price_'+str(v)]
    except:
        print("unable to divide for spread_bill_item_unit_price")
        lg.write("unable to divide for spread_bill_item_unit_price")
        cust['spread_bill_item_unit_price_'+str(v)]=np.NaN

    cust['tot_bill_amt_month_1_10day_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['month_1_10day_flag']==1)].groupby('mobile')['bill_am_sum'].sum()
    cust['tot_bill_amt_month_11_20day_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['month_11_20day_flag']==1)].groupby('mobile')['bill_am_sum'].sum()
    cust['tot_bill_amt_month_21_31day_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['month_21_31day_flag']==1)].groupby('mobile')['bill_am_sum'].sum()
    
    cust['tot_visit_month_1_10day_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['month_1_10day_flag']==1)].groupby('mobile')['bill_am_sum'].count()
    cust['tot_visit_month_11_20day_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['month_11_20day_flag']==1)].groupby('mobile')['bill_am_sum'].count()
    cust['tot_visit_month_21_31day_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['month_21_31day_flag']==1)].groupby('mobile')['bill_am_sum'].count()
    
    cust['tot_bill_amt_weekend_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['weekend_flag']==1)].groupby('mobile')['bill_am_sum'].sum()
    cust['tot_bill_amt_weekday_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['weekend_flag']==0)].groupby('mobile')['bill_am_sum'].sum()
    
    cust['tot_visit_weekend_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['weekend_flag']==1)].groupby('mobile')['bill_am_sum'].count()
    cust['tot_visit_weekday_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v) & (bill['weekend_flag']==0)].groupby('mobile')['bill_am_sum'].count()
    cust['tot_visit_'+str(v)]=bill[((bill['decision_date']-bill['bill_date_max']).dt.days < v)].groupby('mobile')['bill_am_sum'].count()
    
    try:
        cust['tot_bill_amt_month_0_10day_pect_'+str(v)] = cust['tot_bill_amt_month_1_10day_'+str(v)]/cust['tot_bill_amt_'+str(v)]
    except:
        print("unable to divide for tot_bill_amt_month_0_10day_pect_")
        lg.write("unable to divide for tot_bill_amt_month_0_10day_pect_")
        cust['tot_bill_amt_month_0_10day_pect_'+str(v)]=np.NaN

    try:
        cust['tot_visit_month_0_10day_pect_'+str(v)] = cust['tot_visit_month_1_10day_'+str(v)]/cust['tot_visit_'+str(v)]
    except:
        print("unable to divide for tot_visit_month_0_10day_pect_")
        lg.write("unable to divide for tot_visit_month_0_10day_pect_")
        cust['tot_visit_month_0_10day_pect_'+str(v)]=np.NaN
   
    try:
        cust['tot_bill_amt_weekend_pect_'+str(v)] = cust['tot_bill_amt_weekend_'+str(v)]/cust['tot_bill_amt_'+str(v)]
    except:
        print("unable to divide for tot_bill_amt_weekend_pect_")
        lg.write("unable to divide for tot_bill_amt_weekend_pect_")
        cust['tot_bill_amt_weekend_pect_'+str(v)]=np.NaN

    try:
        cust['tot_visit_weekend_pect_'+str(v)] = cust['tot_visit_weekend_'+str(v)]/cust['tot_visit_'+str(v)]
    except:
        print("unable to divide for tot_visit_weekend_pect_")
        lg.write("unable to divide for tot_visit_weekend_pect_")
        cust['tot_visit_weekend_pect_'+str(v)]=np.NaN
    
    cust['months_active_'+str(v)] = bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['month'].nunique()
    cust['unique_store_'+str(v)] = bill[(bill['decision_date']-bill['bill_date_max']).dt.days < v].groupby('mobile')['billing_store_code'].nunique()
    
    try:
        cust['ratio_unique_store_to_month_'+str(v)] = cust['unique_store_'+str(v)]/cust['months_active_'+str(v)]
    except:
        print("unable to divide for ratio_unique_store_to_month_")
        lg.write("unable to divide for ratio_unique_store_to_month_")
        cust['ratio_unique_store_to_month_'+str(v)]=np.NaN
    
    try:
        cust['store_visits_per_month_'+str(v)] = cust['tot_visit_'+str(v)]/cust['months_active_'+str(v)]
    except:
        print("unable to divide for store_visits_per_month_")
        lg.write("unable to divide for store_visits_per_month_")
        cust['store_visits_per_month_'+str(v)]=np.NaN

    for col in [str('avg_bill_item_unit_price_' + str(v)),str('max_bill_item_unit_price_' + str(v)), str('min_bill_item_unit_price_'+ str(v)), str('med_bill_item_unit_price_'+ str(v)),
                str('std_bill_item_unit_price_' + str(v)), str('spread_bill_item_unit_price_'+str(v)), str('spread_bill_amt_'+str(v)),str('spread_bill_items_'+str(v)),str('spread_bill_qty_'+str(v)),
                str('spread_bill_item_unit_price_'+str(v)),
                str('tot_bill_amt_month_0_10day_pect_'+str(v)),str('tot_visit_month_0_10day_pect_'+str(v)),str('tot_bill_amt_weekend_pect_'+str(v)),
                str('tot_visit_weekend_pect_'+str(v)),str('ratio_unique_store_to_month_'+str(v)), str('store_visits_per_month_'+str(v))
                ]:
        cust.loc[~np.isfinite(cust[col]),col]=np.nan
        cust[col] = cust[col].round(3)
        
        
cust['sales_gt_365']=bill[(bill['decision_date']-bill['bill_date_max']).dt.days > 365].groupby('mobile')['bill_am_sum'].sum()
cust['txncnt_gt_365']=bill[(bill['decision_date']-bill['bill_date_max']).dt.days > 365].groupby('mobile')['bill_number'].nunique()

cust['1txn_only_ever'] = cust['1txn_only_ever'].astype(int)

cust['zero_unit_price_item']=bill.groupby('mobile')['zero_unit_price_item_flag_max'].sum()

try:
    cust['bill_amt_90_to_total'] = cust['tot_bill_amt_90']/cust['bill_am_sum']
except:
    print("unable to divide for bill_amt_90_to_total")
    lg.write("unable to divide for bill_amt_90_to_total")
    cust['bill_amt_90_to_total']=np.NaN

try:
    cust['bill_amt_180_to_total'] = cust['tot_bill_amt_180']/cust['bill_am_sum']
except:
    print("unable to divide for bill_amt_180_to_total")
    lg.write("unable to divide for bill_amt_180_to_total")
    cust['bill_amt_180_to_total']=np.NaN

try:
    cust['bill_amt_365_to_total'] = cust['tot_bill_amt_365']/cust['bill_am_sum']
except:
    print("unable to divide for bill_amt_365_to_total")
    lg.write("unable to divide for bill_amt_365_to_total")
    cust['bill_amt_365_to_total']=np.NaN

try:
    cust['bill_amt_90_to_180'] = cust['tot_bill_amt_90']/cust['tot_bill_amt_180']
except:
    print("unable to divide for bill_amt_90_to_180")
    lg.write("unable to divide for bill_amt_90_to_180")
    cust['bill_amt_90_to_180']=np.NaN

try:
    cust['bill_amt_180_to_365'] = cust['tot_bill_amt_180']/cust['tot_bill_amt_365']
except:
    print("unable to divide for bill_amt_180_to_365")
    lg.write("unable to divide for bill_amt_180_to_365")
    cust['bill_amt_180_to_365']=np.NaN

try:    
    cust['avg_item_price_90_to_180'] = cust['avg_bill_item_unit_price_90']/cust['avg_bill_item_unit_price_180']
except:
    print("unable to divide for avg_item_price_90_to_180")
    lg.write("unable to divide for avg_item_price_90_to_180")
    cust['avg_item_price_90_to_180']=np.NaN

try:
    cust['avg_item_price_180_to_365'] = cust['avg_bill_item_unit_price_180']/cust['avg_bill_item_unit_price_365']
except:
    print("unable to divide for avg_item_price_180_to_365")
    lg.write("unable to divide for avg_item_price_180_to_365")
    cust['avg_item_price_180_to_365']=np.NaN

try:
    cust['avg_qty_90_to_180'] = cust['avg_bill_qty_90']/cust['avg_bill_qty_180']
except:
    print("unable to divide for avg_qty_90_to_180")
    lg.write("unable to divide for avg_qty_90_to_180")
    cust['avg_qty_90_to_180']=np.NaN

try:
    cust['avg_qty_180_to_365'] = cust['avg_bill_qty_180']/cust['avg_bill_qty_365']
except:
    print("unable to divide for avg_qty_180_to_365")
    lg.write("unable to divide for avg_qty_180_to_365")
    cust['avg_qty_180_to_365']=np.NaN

try:
    cust['avg_items_90_to_180'] = cust['avg_bill_items_90']/cust['avg_bill_items_180']
except:
    print("unable to divide for avg_items_90_to_180")
    lg.write("unable to divide for avg_items_90_to_180")
    cust['avg_items_90_to_180']=np.NaN
    
try:    
    cust['avg_items_180_to_365'] = cust['avg_bill_items_180']/cust['avg_bill_amt_365']
except:
    print("unable to divide for avg_items_180_to_365")
    lg.write("unable to divide for avg_items_180_to_365")
    cust['avg_items_180_to_365']=np.NaN

try:
    cust['med_item_price_90_to_180'] = cust['med_bill_item_unit_price_90']/cust['med_bill_item_unit_price_180']
except:
    print("unable to divide for med_item_price_90_to_180")
    lg.write("unable to divide for med_item_price_90_to_180")
    cust['med_item_price_90_to_180']=np.NaN

try:    
    cust['med_item_price_180_to_365'] = cust['med_bill_item_unit_price_180']/cust['med_bill_item_unit_price_365']
except:
    print("unable to divide for med_item_price_180_to_365")
    lg.write("unable to divide for med_item_price_180_to_365")
    cust['med_item_price_180_to_365']=np.NaN

try:    
    cust['med_qty_90_to_180'] = cust['med_bill_qty_90']/cust['med_bill_qty_180']
except:
    print("unable to divide for med_qty_90_to_180")
    lg.write("unable to divide for med_qty_90_to_180")
    cust['med_qty_90_to_180']=np.NaN

try:    
    cust['med_qty_180_to_365'] = cust['med_bill_qty_180']/cust['med_bill_qty_365']
except:
    print("unable to divide for med_qty_180_to_365")
    lg.write("unable to divide for med_qty_180_to_365")
    cust['med_qty_180_to_365']=np.NaN

try:    
    cust['med_items_90_to_180'] = cust['med_bill_items_90']/cust['med_bill_items_180']
except:
    print("unable to divide for med_items_90_to_180")
    lg.write("unable to divide for med_items_90_to_180")
    cust['med_items_90_to_180']=np.NaN

try:
    cust['med_items_180_to_365'] = cust['med_bill_items_180']/cust['med_bill_amt_365']
except:
    print("unable to divide for med_items_180_to_365")
    lg.write("unable to divide for med_items_180_to_365")
    cust['med_items_180_to_365']=np.NaN


for col in ['med_items_180_to_365' , 'med_items_90_to_180' , 'med_qty_180_to_365' , 'med_qty_90_to_180' , 
            'med_item_price_180_to_365' , 'med_item_price_90_to_180' , 'avg_items_180_to_365' , 
            'avg_items_90_to_180' , 'avg_qty_180_to_365' , 'avg_qty_90_to_180' , 'avg_item_price_180_to_365' , 
            'avg_item_price_90_to_180' , 'bill_amt_180_to_365' , 'bill_amt_90_to_180' , 'bill_amt_365_to_total' , 
            'bill_amt_180_to_total' , 'bill_amt_90_to_total']:
    cust.loc[~np.isfinite(cust[col]),col]=np.nan
    cust[col] = cust[col].round(3)


cust.to_csv("customer_summary_160819.csv") 
cust.to_pickle("customer_summary_160819.pkl") 

desc=cust.describe(include= 'all').transpose()
desc.to_csv("customer_desc_160819.csv")


#test=cust[list([ c for c in cust.columns if "spread" in c ])]
#
#cust.hist([x for x in cust.columns if "spread" in x],bins = 30)
#
#[x for x in df.columns if "spread" in x]
#
#cust[['spread_bill_amt_90']].hist(bins =30)
#
