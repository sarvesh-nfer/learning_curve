import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import pandas as pd
from pandas import DataFrame
import sys
import os,glob
import time

def acq_plot(path):
    start = time.time()
    conn = sqlite3.connect(path)
    save_path = os.path.split(path)[0]
    df = pd.read_sql_query("select * from acquisition_blob_info ;", conn)
    ##plot


    ###################
    j=0
    focus=[]
    focus1=[]
    focus2=[]
    focus3=[]
    focus4=[]
    for k in df['aoi_name'].unique():
        for i in range(0,175):
                if df['blob_index'][i]==df['blob_index'][j]:
                    focus.append(df['focus_metric'][j])
                    focus1.append(df['stack_index'][j])
                    focus2.append(df['blob_index'][j])
                    focus3.append(df['aoi_name'][j])
                    focus4.append(df['color_metric'][j])
                    i =i+1
                    j= j+1           
                    if len(focus)>=175:
                        df4 = pd.DataFrame(list(zip(focus1, focus, focus2,focus3,focus4)),columns =['stack_index', 'focus_metric','blob_index','aoi_name','color_metric']) 
                        df3=df4.iloc[[0,1,2,3,4,25,26,27,28,29,50,51,52,53,54,75,76,77,78,79,100,101,102,103,104,125,126,127,128,129,
                                    150,151,152,153,154,
                                    5,6,7,8,9,30,31,32,33,34,55,56,57,58,59,80,81,82,83,84,105,106,107,108,109,130,131,132,133,134,
                                    155,156,157,158,159,
                                    10,11,12,13,14,35,36,37,38,39,60,61,62,63,64,85,86,87,88,89,110,111,112,113,114,135,136,137,138,139,
                                    160,161,162,163,164,
                                    15,16,17,18,19,40,41,42,43,44,65,66,67,68,69,90,91,92,93,94,115,116,117,118,119,140,141,142,143,144,
                                    165,166,167,168,169,
                                    20,21,22,23,24,45,46,47,48,49,70,71,72,73,74,95,96,97,98,99,120,121,122,123,124,145,146,147,148,149,
                                    170,171,172,173,174]]
                        df3.reset_index()
                        fig = px.line(df3, x="stack_index", y="focus_metric",facet_col="blob_index",facet_col_wrap=7,facet_col_spacing=0.04,height=800, width=1000,title="Stack_index VS Focus_metric for"+" " +df3['aoi_name'][1])
                        fig.add_hline(y=6.0, line_width=2, line_dash="dashdot", line_color="red")
                        focus.clear()
                        focus1.clear()
                        focus2.clear()
                        focus3.clear()
        if j>=len(df['aoi_name'])-1:
            print("All PLots Generated")
            break
        if not os.path.exists(save_path+"/grid_"+str(df['grid_id'][j])+"/acq_plots"):
            os.makedirs(save_path+"/grid_"+str(df['grid_id'][j])+"/acq_plots")                            
        fig.write_image(save_path+"/grid_"+str(df['grid_id'][j])+"/acq_plots"+"/"+df3['aoi_name'][1]+".png")
    end = time.time()

    print("**"*50)
    print(f"TIME TAKEN FOR EXECUTION : {end - start}")
    print("**"*50)
        

if __name__ == "__main__":

    # if len(sys.argv) < 1:
    #     print("Inavlid input arguments\n\n<python3 3_bubble_mapping.py>\n\t"\
    #             "1.DB Path")

    # slide_path = sys.argv[1]
    lst = glob.glob("/home/adminspin/Music/sarvesh/validation/*/*.db")
    for i in lst:
        acq_plot(i)
