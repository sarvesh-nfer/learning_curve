import pandas as pd
import numpy as np
import os
import glob
import sqlite3

db_list = glob.glob('/home/adminspin/Desktop/plane/sheet3/*/*.db')

lst1 = []
lst2 = []
lst3 = []
lst4 = []
lst5 = []
lst6 = []
lst7 = []
lst8 = []
lst9 = []
lst10 = []
lst22 = []
lst33 = []
lst44 = []
lst55 = []
lst66 = []
for i in db_list:
    try:
        db_path = i
        conn = sqlite3.connect(db_path)
        cur = conn.cursor() 
        cur.execute("select * from aoi limit 5;")
        results = cur.fetchall()
        df = pd.read_sql_query("select * from aoi ;", conn)
        df_blob = pd.read_sql_query("select * from acquisition_blob_info ;", conn)
        df_blob = df_blob[df_blob['color_metric'] != 0]
        a= os.path.split(db_path)[-1].split('.')[0]
        df2_acq = df[(df['bg_state_acq'] == 0) & (df['focus_metric'] >= 6) & ((df['color_metric'] >= 40) | (df['hue_metric'] >= 1000))]
        for j in df_blob['grid_id'].unique():
            df_new = df_blob[df_blob['grid_id'] == j]
            lst1.append(len(df_new))
            lst2.append(np.round((len(df_new[df_new['focus_metric']>=6])/len(df_new))*100,2))
            lst22.append(len(df_new[df_new['focus_metric']>=6]))
            lst3.append(np.round((len(df_new[df_new['color_metric']>=40])/len(df_new))*100,2))
            lst33.append(len(df_new[df_new['color_metric']>=40]))
            lst4.append(np.round((len(df_new[df_new['hue_metric']>=1000])/len(df_new))*100,2))
            lst44.append(len(df_new[df_new['hue_metric']>=1000]))
            lst5.append(np.round((len(df_new[(df_new['focus_metric']>=6)&(df_new['color_metric']>=40)])/len(df_new))*100,2))
            lst55.append(len(df_new[(df_new['focus_metric']>=6)&(df_new['color_metric']>=40)]))
            lst6.append(np.round((len(df_new[(df_new['focus_metric']>=6)&(df_new['hue_metric']>=1000)])/len(df_new))*100,2))
            lst66.append(len(df_new[(df_new['focus_metric']>=6)&(df_new['hue_metric']>=1000)]))
            lst7.append(len(df_blob[(df_blob['grid_id'] == j) & ((df_blob['focus_metric'] >= 6) & ((df_blob['color_metric'] >= 40) | (df_blob['hue_metric'] >= 1000)))]))
            lst8.append(np.round((len(df_blob[(df_blob['grid_id'] == j) & ((df_blob['focus_metric'] >= 6) & ((df_blob['color_metric'] >= 40) | (df_blob['hue_metric'] >= 1000)))])/len(df_blob[df_blob['grid_id'] == j]))*100,2))
            lst9.append(a)
            lst10.append(j)
        df_second = pd.DataFrame(list(zip(lst9,lst10,lst1, lst2,lst22,lst3,lst33,lst4,lst44,lst5,lst55,lst6,lst66,lst7,lst8)),columns =['slide_name','grid_id','total_blobs','fm_blobs%','fm_blobs','cm_blobs%','cm_blobs','hm_blobs%','hm_blobs','sat_blobs%','sat_blobs','hue_blobs%','hue_blobs','valid_blob','valid_blob%'])
        df_second.to_csv('/home/adminspin/Desktop/df_second.csv',index=False)
        print('saved for slide : ',a)
        print("--"*50)
    except Exception as msg:
        print("**"*50)
        print('Error occured for slide : ', a)
        print("**"*50)