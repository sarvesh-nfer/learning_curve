import pandas as pd
import sqlite3
import os,glob
import numpy as np

sar = {}
lst = []
for slide in glob.glob("/home/adminspin/wsi_app/acquired_data/*/*.db"):
    conn = sqlite3.connect(slide)
    df = pd.read_sql_query("select * from grid_info;",conn)
    df2 = df[df['grid_name']=="merged_grid"]
    df = df[df['grid_name']!="merged_grid"]
    try:
        lst.append(df2['row_count'].iloc[0])
    except:
        lst.append(0)
    total = 0
    df['re_scanned_rows_trm'] = df['re_scanned_rows_trm'].replace("", 0)
    for i in df["re_scanned_rows_trm"]:

        try:
            print(len(i.split(',')))
            total = total + len(i.split(','))
        except Exception as msg:
            print("msg : ",msg)
    sar[slide.split('/')[-2]] = total
    

df = pd.DataFrame(sar.items(),columns=['slide_name','trm_rows'])
df['lst'] = lst
df.to_csv("/home/adminspin/Music/scripts/trm.csv",index=False)

print("Sum of Snap miss : ",df['trm_rows'].sum())
print("total no. of rows : ",df['lst'].sum())
