import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import sqlite3

txt = glob.glob("/home/adminspin/Music/db_data_shipped/*.db")


df = pd.DataFrame()

for i in txt:
    try:
        conn = sqlite3.connect(i)



        df2 = pd.read_sql_query("select * from registration_info;",conn)

        df2['slide_name'] = i.split("/")[-2]

        df = df.append(df2)

    except:
        print("error in DB : ",i)

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
fig3.add_annotation(text="<b>Stack Shift Data across scanner <b>"+str(len(txt)),xref="paper", yref="paper",showarrow=False,x=0, y=1.11,font=dict(family="Courier New, monospace",
        size=24,color="RebeccaPurple"))

for idx,i in enumerate(["yellow","blue","green"]):
    fig3.add_vline(x=(df['x_shift'].mean() - (idx+1) * df['x_shift'].std()), line_width=3, line_dash="dash", line_color=i,row=1,col=1)
    fig3.add_vline(x=(df['x_shift'].mean() + (idx+1) * df['x_shift'].std()), line_width=3, line_dash="dash", line_color=i,row=1,col=1)

fig3.add_vline(x=-11.5, line_width=3, line_dash="dash", line_color="red")
fig3.add_vline(x=11.5, line_width=3, line_dash="dash", line_color="red")


if round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2) > 95:

    fig3.add_annotation(x=16,y=(df['x_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
            text="Percentage of distribution <br>of X-Shift : "+str(round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2)),
            showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
            borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=1)
else:
    fig3.add_annotation(x=16,y=(df['x_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
        text="Percentage of distribution <br>of X-Shift : "+str(round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2)),
        showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
        borderwidth=2,borderpad=4,bgcolor="crimson",opacity=0.8,row=1,col=1)

if round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2) > 95:
    fig3.add_annotation(x=16,y=(df['y_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
            text="Percentage of distribution <br>of Y-Shift : "+str(round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2)),
            showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
            borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=2)
else:
    fig3.add_annotation(x=16,y=(df['y_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
        text="Percentage of distribution <br>of Y-Shift : "+str(round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2)),
        showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
        borderwidth=2,borderpad=4,bgcolor="crimson",opacity=0.8,row=1,col=2)

fig3.show()
