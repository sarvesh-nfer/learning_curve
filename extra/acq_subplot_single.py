import sqlite3
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import numpy as np

aoi_name = "aoi1544"
grid_id = 1

db_path = "/home/adminspin/Music/db_data_shipped/H01BBB16P-3603.db"

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("select * from acquisition_blob_info where aoi_name == '{}' and grid_id == {}".format(aoi_name,grid_id),conn)

# df

row = 1
col = 1



order = [0,5,10,15,20,25,30,1,6,11,16,21,26,31,2,7,12,17,22,27,32
            ,3,8,13,18,23,28,33,4,9,14,19,24,29,34]

for j in df['aoi_name'].unique():
    try:
        aoi = df[df['aoi_name']==j]
        
        lst = []
        for i in range(0,len(aoi['focus_metric'])+len(aoi['focus_metric'])//35,5):
            try:
                lst.append(np.argmax(aoi['focus_metric'][i-5:i]))
            except Exception as msg:
                print(msg)
        
        fig = make_subplots(rows=5, cols=7,shared_yaxes=True,
                            subplot_titles = [f"Blob idx : {sar}<br>Best idx : {sar2}" for sar,sar2 in zip(order,lst)])
        start_time = time.time()
        count = 0
        for i in aoi['blob_index'].unique():

            

            fig.add_trace(go.Scatter(x=aoi[aoi['blob_index']==i]["stack_index"],
                                    y=aoi[aoi['blob_index']==i]['focus_metric'],mode="lines+markers",
                                    showlegend=False,marker=dict(color="blue")),row=row,col=col)

            col+=1

            if col > 7:
                col = 1
                row+=1

        fig.update_yaxes(range=[0,(max(aoi['focus_metric'])+10)])

        fig.update_xaxes(dtick=1)
        fig.add_hline(y=7.0, line_width=2, line_dash="dashdot", line_color="red")
        fig.update_layout(height=1000,width=1200,title="AOI: <b>"+j)
        fig.update_yaxes(title_text="Focus Metric", title_font=dict(
                family="Courier New, monospace",
                size=28,
                color="RebeccaPurple"),row=3, col=1)
        fig.update_xaxes(title_text="Stack index", title_font=dict(
                family="Courier New, monospace",
                size=28,
                color="RebeccaPurple"),row=5, col=4)
        fig.show()
#         fig.write_image("/home/adminspin/Downloads/JR-22-1322-A3-1_H01BBB18P-2283/FM_plots"+"/"+j+".png")
        print("saved for : ",j)
        print("--- %s seconds ---" % (time.time() - start_time))

        
    except Exception as msg:
        print("msg :",msg)
        
#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
