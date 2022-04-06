import pandas as pd
import sqlite3
import glob,os
import sys

def json_csv(path):
    slides = os.listdir(path)
    for i in slides:
        slide_path = os.path.join(path,i)
        for j,db in zip(glob.glob(slide_path+"/*/stitching_error_aois_coordinates.json"),glob.glob(slide_path+"/*db")):
            #reading json to df
            json = pd.read_json(j)
            aoi1=[]
            aoi2=[]
            for jlen in range(len(json)):
                aoi1.append(json['error_aois_coord_list'][jlen]['aoi_1'])
                aoi2.append(json['error_aois_coord_list'][jlen]['aoi_2'])
            aoi = aoi1+aoi2
            
            conn = sqlite3.connect(db)
            df = pd.read_sql_query("select * from aoi;",conn)
            df = df[df['aoi_name'].isin(aoi)]
            df['background_ratio'] = 100-(df['biopsy_ratio'] - df['debris_ratio'])
            df = df[['aoi_name','biopsy_ratio','debris_ratio','background_ratio','focus_metric','color_metric','hue_metric','grid_id']]
            df.to_csv(slide_path+"/brdr_"+i+".csv",index=False)
            print("Data saved for Slide : ",i)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("Pass correct arguments !!!")
    json_csv(file_path)