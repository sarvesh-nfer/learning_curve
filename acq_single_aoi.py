import sqlite3
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time,sys,os

db_path = "/home/adminspin/Music/H-129-21-A9_H01BBB27P-6362/H-129-21-A9_H01BBB27P-6362.db"

aoi_name = "aoi4591"
grid_id = 1

conn = sqlite3.connect(db_path)

df = pd.read_sql_query("select * from acquisition_blob_info",conn)

df = df[(df['aoi_name']==aoi_name)&(df['grid_id']==grid_id)]

df = df.drop_duplicates(subset=["stack_index","blob_index"],keep="last")
row = 1
col = 1

lst = []

for j in df['aoi_name'].unique():
    try:
        
        aoi = df[df['aoi_name']==j]
        fig = make_subplots(rows=5, cols=7,subplot_titles = ["Blob idx :"+str(sar) for sar in aoi['blob_index'].unique()])
        start_time = time.time()

        for i in aoi['blob_index'].unique():
            
            fig.add_trace(go.Scatter(x=aoi[aoi['blob_index']==i]["stack_index"],
                                    y=aoi[aoi['blob_index']==i]['focus_metric'],mode="lines+markers",
                                    showlegend=False,marker=dict(color="blue")),row=row,col=col)
            row+=1
            col+=1

            if row > 5:
                row = 1
            if col > 7:
                col = 1

        # fig.update_yaxes(dtick=1)
        if max(aoi['focus_metric'] > 7):
            fig.update_yaxes(range=[min(aoi['focus_metric']),(max(aoi['focus_metric'])+10)])

        fig.update_xaxes(dtick=1)
        fig.add_hline(y=7.0, line_width=2, line_dash="dashdot", line_color="red")
        fig.update_layout(height=800,width=1200,title="AOI: <b>"+j)
        fig.update_yaxes(title_text="Focus Metric", title_font=dict(
                family="Courier New, monospace",
                size=28,
                color="RebeccaPurple"),row=3, col=1)
        fig.update_xaxes(title_text="Stack index", title_font=dict(
                family="Courier New, monospace",
                size=28,
                color="RebeccaPurple"),row=5, col=4)
    #     fig.show()
        fig.write_image(os.path.split(db_path)[0]+"/"+j+".png")
        print("saved for : ",j)
        print("--- %s seconds ---" % (time.time() - start_time))
        lst.append((time.time() - start_time))
        
    except Exception as msg:
        print("msg :",msg)
