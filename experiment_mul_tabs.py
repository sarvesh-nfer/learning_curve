import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np




df=pd.read_csv("/home/adminspin/Desktop/df2.csv")
app = dash.Dash(__name__)


app.layout = html.Div([
html.Div(
    
    html.Div(className='row', children=[
        html.Div(children=[
            html.Label(['Load Identifier'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(id="load",options=[{"label": x, "value": x}
                for x in df['loadIdentifier'].unique()],value=df['loadIdentifier'].unique()[0],clearable=False)],style=dict(width='50%')),

        
        html.Div(children=[
                html.Label(['Model Type'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(id="property",options=[{"label": x, "value": x}
                for x in ["ALL","slide_placement",'barcode_decoding','localization_time','micro_camera_conn','validation_time','FS_time','total_acq_time', "slide_drop","post_preocessing","transfer"]],
                value="ALL",clearable=False)],style=dict(width='50%'))],style=dict(display='flex'))),
    
    dcc.Graph(id="slide_placement")
    
])
@app.callback(
    Output("slide_placement", "figure"),
    [Input("load", "value"),Input("property", "value")])
def slide_placement(load,property):

    first = df[df['loadIdentifier'] == str(load)]
    if property == "ALL":
        fig = px.bar(first,x="slide_name", y=['slide_placement','barcode_decoding','localization_time','micro_camera_conn','validation_time','FS_time','total_acq_time', 'slide_drop','post_preocessing','transfer'],
        color_discrete_sequence=px.colors.qualitative.Bold, title="All Modules")
        fig.update_layout(width=1500,height=1000,hovermode="x")
        fig.update_xaxes(title='Slide Names')
        fig.update_yaxes(title = 'Time(s)')
        return fig
    else:
        fig = px.line(y = first[property],height= 800,title="Plot for "+str(property))
        fig.update_xaxes(title = "No. of Slides")
        fig.update_yaxes(title = "Time (s)",range=[first[property].min() - 10,first[property].max()+10])
        fig.add_hline(y=np.mean(first[property]),line_color="green")
        fig.add_hline(y=first[property].min(),line_dash='dash',line_color="red")
        fig.add_hline(y=first[property].max(),line_dash='dash',line_color="red")
        return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8081)



import dash
import dash_html_components as html
import dash_core_components as dcc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

tab1 = html.Div([
html.Div(
    
    html.Div(className='row', children=[
        html.Div(children=[
            html.Label(['Load Identifier'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(id="load",options=[{"label": x, "value": x}
                for x in df['loadIdentifier'].unique()],value=df['loadIdentifier'].unique()[0],clearable=False)],style=dict(width='50%')),

        
        html.Div(children=[
                html.Label(['Model Type'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(id="property",options=[{"label": x, "value": x}
                for x in ["ALL","slide_placement",'barcode_decoding','localization_time','micro_camera_conn','validation_time','FS_time','total_acq_time', "slide_drop","post_preocessing","transfer"]],
                value="ALL",clearable=False)],style=dict(width='50%'))],style=dict(display='flex'))),
    
    dcc.Graph(id="slide_placement")
    
])

tab2 = html.Div([
html.Div(
    
    html.Div(className='row', children=[
        html.Div(children=[
            html.Label(['Slide Name'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(id="slide_name",options=[{"label": x, "value": x}
                for x in df['loadIdentifier'].unique()],value=df['loadIdentifier'].unique()[0],clearable=False)],style=dict(width='100%'))])),
    
    dcc.Graph(id="slide_placement")
    
])

@app.callback(dash.dependencies.Output('slide_placement', 'children'),
             [dash.dependencies.Input('slide_name', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return tab1
    elif tab == 'tab-2-example':
        return tab2

@app.callback(
    [dash.dependencies.Output('second-dropdown', 'options'),
     dash.dependencies.Output('second-dropdown', 'value')],
    [dash.dependencies.Input('first-dropdown', 'value')])
def update_dropdown(value):
    return [[ {'label': i, 'value': i} for i in myDict[value] ], myDict[value][default_index]]

if __name__ == '__main__':
    app.run_server(debug=True)