import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys,os,glob

def direct_plot(slide_path):
    print(slide_path)
    print(glob.glob(slide_path+"/*.db"))
    conn = sqlite3.connect(glob.glob(slide_path+"/*.db")[0])
    df = pd.read_sql_query("select * from registration_info;",conn)
    #sorting the df by stack index 1
    df = df.sort_values(by=['stack_index_1'])
    df['axes'] = ["img_"+str(y)+" & img_" + str(x) if x > y else "img_"+str(x)+" & img_" + str(y) for x,y in zip(df['stack_index_1'],df['stack_index_2'])]
    df
    fig3 = make_subplots(rows=1, cols=2,subplot_titles=("<b>Frequency of X-Shift","<b>Frequency of Y-Shift"))
    fig3.add_trace(
        go.Bar(x=df['x_shift'].value_counts().index,y=df['x_shift'].value_counts().values,
            text=df['x_shift'].value_counts().values,
            name="X-shift",marker=dict(color="blue")),
        row=1, col=1
    )
    fig3.add_trace(
        go.Bar(x=df['y_shift'].value_counts().index,y=df['y_shift'].value_counts().values,
            text=df['y_shift'].value_counts().values,name="Y-Shift",marker=dict(color="purple")),
        row=1, col=2
    )
    fig3.update_xaxes(title="Shift Values",dtick=1,tickangle=0,range=[-16.9,16.9])
    fig3.update_yaxes(title="Count")
    fig3.update_layout(width=1800,height=800)
    fig3.update_traces(textfont_size=20, textposition="outside",textangle=0, cliponaxis=True)
    fig3.add_annotation(text="<b>Stack Shift Data for <b>"+str(glob.glob(path+"/*.db")[0].split("/")[-1].split("-")[0]),xref="paper", yref="paper",showarrow=False,x=0, y=1.11,font=dict(family="Courier New, monospace",
            size=24,color="RebeccaPurple"))
    fig3.add_vline(x=-11.5, line_width=3, line_dash="dash", line_color="red")
    fig3.add_vline(x=11.5, line_width=3, line_dash="dash", line_color="red")


    if round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2) > 95:

        fig3.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
                text="Percentage of distribution <br>of X-Shift : "+str(round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2)),
                showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=1)
    else:
        fig3.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
            text="Percentage of distribution of X-Shift : "+str(round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2)),
            showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
            borderwidth=2,borderpad=4,bgcolor="crimson",opacity=0.8,row=1,col=1)

    if round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2) > 95:
        fig3.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
                text="Percentage of distribution <br>of Y-Shift : "+str(round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2)),
                showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=2)
    else:
        fig3.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
            text="Percentage of distribution <br>of Y-Shift : "+str(round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2)),
            showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
            borderwidth=2,borderpad=4,bgcolor="crimson",opacity=0.8,row=1,col=2)


    fig4 = px.scatter(x=df['x_shift'],y=df['y_shift'],facet_col=df['axes'],facet_col_wrap=2,
                    facet_col_spacing=0.025,)
    fig4.for_each_annotation(lambda a: a.update(text="<b> Stack Wise Shift for : "+a.text.split("=")[-1]))
    fig4.update_yaxes(title="Y-Shift Values",showticklabels=True)
    fig4.update_xaxes(title="X-shift Values",showticklabels=True,tickangle=45)


    # fig.update_yaxes(matches=None)
    fig4.update_layout(width=1800,height=1400)
    fig4.update_traces(marker_color='green')
    fig4.add_shape(type="rect",
        xref="x", yref="y",
        x0=-12, y0=12, x1=12, y1=-12,row="all",col='all',
        line_color="red")
    # fig4.show()
    print("SHIFTS DONE")

    with open(path+'/'+str(glob.glob(path+"/*.db")[0].split("/")[-1].split("-")[0])+'_report.html', 'a') as f:
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]

    direct_plot(path)
    # for i in os.listdir(path):
    #     slide_path = path+"/"+i
    #     direct_plot(slide_path)