import sqlite3
import pandas as pd

# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import plotly.express as px
import time

conn = sqlite3.connect("/home/adminspin/Downloads/JR-22-1322-A3-1_H01BBB18P-2283/JR-22-1322-A3-1_H01BBB18P-2283.db")
df = pd.read_sql_query("select * from acquisition_blob_info;",conn)

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
        # fig = make_subplots(rows=5, cols=7,subplot_titles = ["Blob idx :"+str(sar) for sar in aoi['blob_index'].unique()])
        # 

        # for i in aoi['blob_index'].unique():
            
        #     fig.add_trace(go.Scatter(x=aoi[aoi['blob_index']==i]["stack_index"],
        #                             y=aoi[aoi['blob_index']==i]['focus_metric'],mode="lines+markers",
        #                             showlegend=False,marker=dict(color="blue")),row=row,col=col)
        #     row+=1
        #     col+=1

        #     if row > 5:
        #         row = 1
        #     if col > 7:
        #         col = 1

        # # fig.update_yaxes(dtick=1)
        # if max(aoi['focus_metric'] > 7):
        #     fig.update_yaxes(range=[min(aoi['focus_metric']),(max(aoi['focus_metric'])+10)])

        # fig.update_xaxes(dtick=1)
        # fig.add_hline(y=7.0, line_width=2, line_dash="dashdot", line_color="red")
        # fig.update_layout(height=800,width=1200,title="AOI: <b>"+j)
        # fig.update_yaxes(title_text="Focus Metric", title_font=dict(
        #         family="Courier New, monospace",
        #         size=28,
        #         color="RebeccaPurple"),row=3, col=1)
        # fig.update_xaxes(title_text="Stack index", title_font=dict(
        #         family="Courier New, monospace",
        #         size=28,
        #         color="RebeccaPurple"),row=5, col=4)
    #     fig.show()
        fig.write_image("/home/adminspin/Downloads/JR-22-1322-A3-1_H01BBB18P-2283/FM_plots"+"/"+j+".png")
        print("saved for : ",j)
        print("--- %s seconds ---" % (time.time() - start_time))
        lst.append((time.time() - start_time))
        
    except Exception as msg:
        print("msg :",msg)

df = pd.DataFrame(list(zip(lst)),
               columns =['plot_timings'])
df.to_csv("/home/adminspin/Desktop/plot_timings.csv",index=False)
