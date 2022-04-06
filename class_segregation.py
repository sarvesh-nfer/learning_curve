import pandas as pd 
import sqlite3
import os,glob
import shutil
import sys

def db_conn(slide_path):
    conn = sqlite3.connect(glob.glob(slide_path+"/*.db")[0])
    df = pd.read_sql_query("select * from aoi;",conn)
    # df['grid'] = "grid_"+df['grid_id'] 
    raw_images = glob.glob(slide_path+"/grid_*/raw_images/*.bmp")

    for i in raw_images:
        aoi = os.path.split(i)[-1].split('.')[0]
        grid = i.split('/')[-3].split('_')[-1]
        aoi_class = str(df[(df['grid_id'] == int(grid)) & (df['aoi_name'] == aoi)]['aoi_class'].unique())
        # print("aoi",aoi,"grid",grid,"aoi_class",aoi_class)
        output_path = os.path.join(slide_path,"grid_"+grid,"class",aoi_class)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.copy2(i,output_path)
        print("COPIED : ",i)

if __name__ == '__main__':
    path  = "/home/adminspin/wsi_app/acquired_data/"
    slides = ['H01FBA08R_3247']
    for k in slides:
        db_conn(os.path.join(path,k))