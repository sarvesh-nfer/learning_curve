import pandas as pd
import numpy as np
import os
import glob
import sqlite3

db_list = glob.glob('/home/adminspin/Downloads/additional/*/*.db')

lst1 = []
lst2 = []
lst3 = []
lst4 = []
lst5 = []
for i in db_list:
    try:
        db_path = i
        conn = sqlite3.connect(db_path)
        cur = conn.cursor() 
        cur.execute("select * from aoi limit 5;")
        results = cur.fetchall()
        df = pd.read_sql_query("select * from aoi ;", conn)
        df_blob = pd.read_sql_query("select * from acquisition_blob_info ;", conn)
        df_val = pd.read_sql_query("select * from focus_sampling_info ;", conn)
        a= os.path.split(db_path)[-1].split('.')[0]
        df2_acq = df[(df['bg_state_acq'] == 0) & (df['focus_metric'] >= 6) & ((df['color_metric'] >= 40) | (df['hue_metric'] >= 1000))]
        for i in df['grid_id'].unique():
            lst1.append(len(df[df['grid_id'] == i]))
            lst2.append(len(df[(df['grid_id'] == i) & (df['bg_state_acq'] == 1)]))
            lst3.append(len(df[(df['grid_id'] == i) & (df['bg_state_acq'] == 0)]))
            lst4.append(a)
            lst5.append(i)
        df_first = pd.DataFrame(list(zip(lst4,lst5,lst1, lst2, lst3)),columns =['slide_name','grid_id','total_aoi','foreground_aoi','background_aoi'])
        df_first.to_csv('/home/adminspin/Desktop/df_first.csv',index=False)
        print('saved for slide : ',a)
        print("--"*50)
    except Exception as msg:
        print("**"*50)
        print("db error for Slide : ",a)
    