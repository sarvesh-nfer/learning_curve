import sqlite3
import pandas as pd
import os,glob
import shutil
import sys

def transfer_raw_best_idx(slide_path):
    db_path = glob.glob(slide_path+"/*.db")[0]
    conn = sqlite3.connect(db_path)
    df1 = pd.read_sql_query("select * from aoi;",conn)
    df2 = pd.read_sql_query("select * from acquisition_blob_info;",conn)

    df1 = df1[['aoi_name','grid_id','best_idx']]
    df3 = pd.merge(df1,df2,on=['aoi_name','grid_id'])

    df3 = df3[df3['stack_index'] == df3['best_idx']]
    # print(df3)

    count = 1
    for i in df3['grid_id'].unique():
        try:

            print(i)
            lst = df3[(df3['grid_id'] == i) & (df3['focus_metric'] < 6)]['aoi_name'].unique()
            print(lst)
            for j in lst:
                try:
                    src = os.path.join(slide_path,"grid_"+str(i),"20x_fm_cm_hm",j+".jpeg")
                    dst = os.path.join(slide_path,"grid_"+str(i),"BI_bg")
                    print(dst)
                    if not os.path.exists(dst):
                        os.mkdir(dst)
                    shutil.copy2(src,dst)
                    print("**"*50)
                    print("Copied image for grid_:",i,"aoi:",j," \t COUNT :",count)
                    count += 1
                    print("**"*50)
                except Exception as msg1:
                    print("error in msg1 :",msg1)
        except Exception as msg2:
            print("error in msg2 :",msg2)


path = "/home/adminspin/wsi_app/acquired_data"
slide_name = ['H01FBA08R_2469', 'H01FBA08R_2470', 'H01FBA08R_2467', 'H01FBA08R_2463', 'H01FBA08R_2465', 'H01FBA08R_2472', 'H01FBA08R_2466', 'H01FBA08R_2471', 'H01FBA08R_2473', 'H01FBA08R_2468']


for k in slide_name:
    slide_path = path + '/' + k
    transfer_raw_best_idx(slide_path)