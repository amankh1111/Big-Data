# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:54:02 2019

@author: homecredit
"""


import dask.dataframe as dd
df = dd.read_csv(file[0], parse_dates=['transaction_date'])   