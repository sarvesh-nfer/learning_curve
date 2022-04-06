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
from pandas.io.json import json_normalize
df = pd.json_normalize(res['hits']['hits'])
df['date'] = pd.to_datetime(df['_source.data.time_stamp']).dt.date
df['date'] = pd.to_datetime(df['date'])
df


from datetime import date
import dash
from dash.dependencies import Input, Output
from dash import html
from dash import dcc



app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        clearable=True,
        max_date_allowed=date.today(),
        initial_visible_month=date.today(),
        date=date.today()
    ),
    html.Div(id='output-container-date-picker-single'),
    dcc.Dropdown(id='counties-dpdn', options=[], multi=True),
])



@app.callback(
    Output('output-container-date-picker-single', 'children'),
    Output('counties-dpdn', 'options'),
    Input('my-date-picker-single', 'date'))
def update_output(date_value):
    print(date_value)
    if date_value is not None:
        lst = df[df['date'] == date_value]['_source.data.scanner_name'].unique()
        print(lst)
        return [x['value'] for x in lst]

# @app.callback(
#     dash.dependencies.Output('cities-dropdown', 'options'),
#     [dash.dependencies.Input('output-container-date-picker-single', 'date2')])
# def set_cities_options(date_value):
#     print(date_value)
#     lst = df[df['date'] == date_value]['_source.data.load_identifier'].unique()
#     print(lst)
#     return [{'label': i, 'value': i} for i in lst]

if __name__ == '__main__':
    app.run_server(port=8060,debug=True)