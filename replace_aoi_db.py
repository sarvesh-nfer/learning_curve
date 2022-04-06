import pandas as pd
import sqlite3
import sys,os
import glob

def replace_shift(path):
    if os.path.exists(glob.glob(path+"/*.db")[0]):
        conn = sqlite3.connect(glob.glob(path+"/*.db")[0])
        db = pd.read_sql_query("select * from registration_info;",conn)
        txt = glob.glob(path+"/*/AcquisitionData.xlsx")
        for i in txt:
            print(i)
            excel = pd.read_excel(i)
            grid = i.split('/')[-2].split('_')[-1]
            df = db[db['grid_id'] == int(grid)]
            # uncomment this if you want only < 12, shift values
            # dfx = df[df['x_shift'] >= 12]
            # dfy = df[df['y_shift'] >= 12]
            # x_shift = excel.replace(to_replace =dfx['aoi_name'].to_list(), 
            #                 value =dfx['x_shift'].to_list())
            # y_shift = excel.replace(to_replace =dfy['aoi_name'].to_list(), 
            #                 value =dfy['y_shift'].to_list())

            x_shift = excel.replace(to_replace =df['aoi_name'].to_list(), 
                            value =df['x_shift'].to_list())
            y_shift = excel.replace(to_replace =df['aoi_name'].to_list(), 
                            value =df['y_shift'].to_list())
            x_shift.to_excel(os.path.split(i)[0]+"/x_shift.xlsx",index=False)
            y_shift.to_excel(os.path.split(i)[0]+"/y_shift.xlsx",index=False)
    else:
        print("error No db file Present")
            
if __name__ == "__main__":
    if len(sys.argv) == 2:
        slide_path = sys.argv[1]
    replace_shift(slide_path)