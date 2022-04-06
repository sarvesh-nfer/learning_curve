import dash  # Dash 1.16 or higher
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
# need to pip install statsmodels for trendline='ols' in scatter plot
#for extracting the data from elastic search
import os
import sys
import elasticsearch
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from elasticsearch import helpers

es= Elasticsearch([{'host': '10.10.6.90','port': 9200}])

es.ping()



res = es.search(index="inline_corrections", doc_type="", body={"query": {"match_all": {}},"sort": [{"data.time_stamp": {"order": "desc"}}]}, size=1000000, )
print(type(res))
print(len((res['hits']['hits'])))
df = pd.json_normalize(res['hits']['hits'])
df['date'] = pd.to_datetime(df['_source.data.time_stamp']).dt.date
df['date2'] = df['date']
df['date'] = pd.to_datetime(df['date'])
df

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.Label("Date:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(
        id='states-dpdn',
        options=[{'label': s, 'value': s} for s in sorted(df['date2'].unique())],
        value=df['date'].iloc[0],
        clearable=False
    ),

    html.Label("Scanner:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.RadioItems(id='counties-dpdn', options=[]),
    html.Label("Cycle:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(id='load', options=[], multi=True),
    html.Label("Modules:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Graph(id='display-map', figure={})
])


# Populate the options of counties dropdown based on states dropdown
@app.callback(
    Output('counties-dpdn', 'options'),
    Input('states-dpdn', 'value')
)
def set_cities_options(chosen_state):
    dff = df[df["date"]==chosen_state]
    # print(dff)
    return [{'label': c, 'value': c} for c in sorted(dff["_source.data.scanner_name"].unique())]


# populate initial values of counties dropdown
@app.callback(
    Output('counties-dpdn', 'value'),
    Input('counties-dpdn', 'options')
)
def set_cities_value(available_options):
    print("available_options : ",available_options)
    return [x['value'] for x in available_options]

@app.callback(
    Output('load', 'options'),
    Input('states-dpdn', 'value'),
    Input('counties-dpdn', 'value')
)
def set_cities_options(chosen_state,chosen_country):
    print("chosen_state : ",chosen_state)
    print("chosen_state : ",chosen_state)
    dff = df[(df["date"]==chosen_state) & (df['_source.data.scanner_name']==chosen_country)]
    # print(dff)
    return [{'label': c, 'value': c} for c in sorted(dff["_source.data.load_identifier"].unique())]

@app.callback(
    Output('display-map', 'figure'),
    Input('counties-dpdn', 'value'),
    Input('states-dpdn', 'value'),
    Input('load', 'value')
)
def update_grpah(selected_counties, selected_state,load):
    if len(selected_counties) == 0:
        return dash.no_update
    else:
        print("selected_counties : ",selected_counties)
        print("selected_state : ",selected_state)
        dff = df[(df["_source.data.scanner_name"].isin(selected_counties))& (df['date'] ==  selected_state)&(df["_source.data.load_identifier"] == load)]
        print("\n\n\n")
        print(dff)

        fig = px.scatter(dff, x='_source.data.slide_id',y='_source.data.computed_angle')
        return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
