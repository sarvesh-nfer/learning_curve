import os,glob
import shutil
import sqlite3
import pandas as pd
import plotly.express as px



def transferFromDB(slide_path):
    db_path = glob.glob(slide_path+'/*db')[0]
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    df1 = pd.read_sql_query("select * from acquisition_blob_info ;", conn)
    df2 = pd.read_sql_query("select * from aoi ;",conn)

    df2 = df2[['aoi_name','grid_id','best_idx']]

    df3 = pd.merge(df2,df1,on=['aoi_name','grid_id'])

    df = df3[df3['stack_index'] == df3['best_idx']]

    df['blob_index'] = df['blob_index'].astype(str)

    df['actual'] = df['aoi_name'] + "_blob_" + df['blob_index']+".jpeg"

    path = glob.glob(slide_path+"/grid_*/cropped/*/*.jpeg")
    output_path = slide_path
    lst2 = []
    for i in path:
        try:
                a = os.path.split(i)[-1]
                g = i.split('/')[-4].split('_')[-1]
                s = i.split('/')[-5]
                if df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] < 6:
                    dst = output_path +"/grid_"+g+"/FM_l6"
                    print("grid : ",g,"\t a : ",a)
                    print(dst)
                    if not os.path.exists(dst):
                        os.mkdir(dst)
                    shutil.copy2(i,dst)
                
                elif df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] > 6 and \
                df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] < 7:
                    
                    dst = output_path +"/grid_"+g+"/FM_6_7"
                    print("grid : ",g,"\t a : ",a)
                    print(dst)
                    if not os.path.exists(dst):
                        os.mkdir(dst)
                    shutil.copy2(i,dst)
                
                elif df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] > 7 and \
                df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] < 8:
                    
                    dst = output_path +"/grid_"+g+"/FM_7_8"
                    print("grid : ",g,"\t a : ",a)
                    print(dst)
                    if not os.path.exists(dst):
                        os.mkdir(dst)
                    shutil.copy2(i,dst)

                elif df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] > 8 and \
                df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] <= 9:
                    
                    dst = output_path +"/grid_"+g+"/FM_8_9"
                    print("grid : ",g,"\t a : ",a)
                    print(dst)
                    if not os.path.exists(dst):
                        os.mkdir(dst)
                    shutil.copy2(i,dst)
                else:
                    (" More than 9 : ",i )
        except Exception as msg:
            print(a," has a problem","\t :",msg)

path = "/home/adminspin/Music/sarvesh/validation"
slide_names = ['H01FBA08R_2469', 'H01FBA08R_2470', 'H01FBA08R_2467', 'H01FBA08R_2463', 'H01FBA08R_2465', 'H01FBA08R_2472', 'H01FBA08R_2466', 'H01FBA08R_2471', 'H01FBA08R_2473', 'H01FBA08R_2468']

# slide_path = "/home/adminspin/Desktop/validation/H01FBA08R_2435"
for sar in slide_names:
    slide_path = path +"/"+sar
    transferFromDB(slide_path)

