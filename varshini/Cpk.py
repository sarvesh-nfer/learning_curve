import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import pathlib
import statistics
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc 
import glob, os
import sys

filepath = '/home/nference/Music/Y connector.xlsx'
name = filepath.split('/')[-1].split('.')[0]
df = pd.read_excel(filepath)

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("CPK Analysis", className = 'text-center text-primary, mb-4'), width = 12)   
    ]),
    dbc.Row([
        dbc.Col([
            html.H5('Select Reference Location: ' + str(name), style = {'textDecoration':'underline'}, className = 'text-primary, mb-2'),
            dcc.RadioItems(id = 'part', value = df['Reference Location'].iloc[0],
                        options = [{'label':x, 'value':x}
                                for x in sorted(df['Reference Location'].unique())], labelClassName = 'mr-5')
        ], width = {'size':12})
    ]),
    html.Br(),
    html.Br(),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id = 'plot',figure = {})
        ], width = {'size':12})
    ])
],fluid = True)

@app.callback(Output(component_id='plot', component_property='figure'),
            Input(component_id='part', component_property='value'),
            prevent_initial_call=False)

def update_graph(part_chosen):
    dff = df[df['Reference Location'] == part_chosen]
    df1 = dff.drop(dff.columns[[0,1,2,3,4,5]], axis = 1)
    df1

    for i in df.index:
        obs = df1.iloc[i][2:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=obs,mode="lines+markers",showlegend = False))
        fig.add_hline(y=df['LSL'].loc[dff.index[i]], line_dash="dot", line_color="red", line_width=2)
        fig.add_hline(y=df['USL'].loc[dff.index[i]], line_dash="dot", line_color="red", line_width=2)
        ucl = statistics.mean(df1.iloc[i,:]) + 3*statistics.stdev(df1.iloc[i,:])
        lcl = statistics.mean(df1.iloc[i,:]) - 3*statistics.stdev(df1.iloc[i,:])
        fig.add_hline(y=ucl, line_dash="dot", line_color="green", line_width=2)
        fig.add_hline(y=lcl, line_dash="dot", line_color="green", line_width=2)
        fig.update_layout(title= str(name) + " : " +str(dff['Char. No/ Balloon Number'].loc[dff.index[i]])+ ' : ' + 
                        dff['Measuring Instrument'].loc[dff.index[i]] + " Specification Limits" ,height=800,width=1875,
                        font=dict(family="Courier New, monospace",size=15,color="RebeccaPurple"))
        fig.update_xaxes(title_text='Observations')
        fig.update_yaxes(title_text='Units of Measurement')   
        mn = min(df1.iloc[i,:].min(),dff['LSL'].loc[dff.index[i]], lcl)
        mx = max(df1.iloc[i,:].max(),dff['USL'].loc[dff.index[i]], ucl)
        fig.add_trace(go.Scatter(y=[df['LSL'].loc[dff.index[i]]],name="Specification Limits",
        line = dict(color="red",width=4,dash="dash")))
        fig.add_trace(go.Scatter(y=[lcl],name="Control Limits",
        line = dict(color="green",width=4,dash="dash")))
        fig.update_yaxes(range=[mn - 0.01*mn, mx + 0.01*mx])
        fig.update_xaxes(range=[-1,len(df1.transpose())])
        test_list = (df1.iloc[i,:]).tolist()
        test_list.pop(1)
        test_list.pop(1)
        # Greater than
        k = dff['USL'].loc[dff.index[i]]
        count = 0
        for p in test_list :
            if p > k :
                count = count + 1
        # Lesser than
        k1 = dff['LSL'].loc[dff.index[i]]
        count1 = 0
        for q in test_list :
            if q < k1 :
                count1 = count1 + 1 
        total = count + count1
        ######## Control
        # Greater than
        k2 = ucl
        count2 = 0
        for r in test_list :
            if r > k2 :
                count2 = count2 + 1
        # Lesser than
        k3 = lcl
        count3 = 0
        for s in test_list :
            if s < k3 :
                count3 = count3 + 1 
        total1 = count2 + count3
        fig.add_annotation(text="<b>Character No: "+str(dff['Char. No/ Balloon Number'].loc[dff.index[i]])+\
            "<br>Characteristic Designator: "+dff['Characteristic Designator'].loc[dff.index[i]]+\
                "<br>Measuring Instrument: "+dff['Measuring Instrument'].loc[dff.index[i]]+\
                    "<br>No of observation outside our specification limits: "+str(total) +\
                        "<br>No of observation outside our control limits: "+str(total1), showarrow=False, 
                x = len(df1.transpose()) - 15, y = mx + 0.005*mx)
        break

    return fig

if __name__ == '__main__':
    app.run_server(port = 4560)
    