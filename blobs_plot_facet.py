import sqlite3
import pandas as pd
import os
import plotly.express as px
import time

aoi_name = "aoi1544"
grid_id = 1

db_path = "/home/adminspin/Music/db_data_shipped/H01BBB16P-3603.db"

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("select * from acquisition_blob_info;",conn)

df = df[(df['aoi_name']==aoi_name)&(df['grid_id']==grid_id)]

df = df.drop_duplicates(subset=["stack_index","blob_index"],keep="last")

row = 1
col = 1


lst = []
for j in df['aoi_name'].unique():
    try:
        
        aoi = df[df['aoi_name']==j]
        start_time = time.time()
        fig = px.scatter(x=aoi['stack_index'],y=aoi['focus_metric'],facet_col=aoi['blob_index'],facet_col_wrap=7,
        category_orders={"facet_col": [0,5,10,15,20,25,30,1,6,11,16,21,26,31,2,7,12,17,22,27,32
            ,3,8,13,18,23,28,33,4,9,14,19,24,29,34]}).update_traces(mode="lines+markers")
        fig.add_hline(y=7,row="all",col="all",line_width=3, line_dash="dash", line_color="red")
        fig.update_yaxes(title='',title_font=dict(size=20),showticklabels=True)
        fig.update_xaxes(title='',title_font=dict(size=20),showticklabels=True)
        fig.update_layout(title="AOI : <b>"+j,width=1200,height=1000,yaxis15=dict(title="Focus Metric"),xaxis4=dict(title="Stack Index"))
        fig.for_each_annotation(lambda a: a.update(text="<b>Blob Index :"+a.text.split("=")[-1]))
        # fig.show()
        fig.write_image(os.path.split(db_path)[0]+"/"+j+".png")
    except:
        print("yes")
