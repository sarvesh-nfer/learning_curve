import time

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.express as px
df=pd.read_csv("/home/adminspin/Desktop/df2.csv")


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.H1("Module wise Timing Data"),
        html.Hr(),
        dbc.Button(
            "Reload Data",
            color="primary",
            id="button",
            className="mb-3",
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Module Wise", tab_id="module"),
                dbc.Tab(label="Slide Wise", tab_id="slide"),
            ],
            id="tabs",
            active_tab="module",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)

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
                for x in df['slide_name'].unique()],value=df['slide_name'].unique()[0],clearable=False),
                html.Br(),
                dcc.RadioItems(id = 'radio_button',
                                      options = [dict(label = 'Grouped', value = 'A'),
                                                 dict(label = 'Ungrouped', value = 'B')],
                                      value = 'A'),],style=dict(width='50%'))])),
    
    dcc.Graph(id="slide_wise")
    
])

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "module":
            return tab1
        elif active_tab == "slide":
            return tab2
    return "No tab selected"

@app.callback(
    Output("slide_wise", "figure"),
    [Input("slide_name", "value"),Input("radio_button","value")])
def slide_placement(slide_name,radio_button):
    slide_wise = df[df['slide_name'] == str(slide_name)]

    if radio_button == 'A':
        
        fig = px.bar(slide_wise, x="slide_name", y=["slide_placement",'barcode_decoding','localization_time','micro_camera_conn','validation_time','FS_time','total_acq_time', "slide_drop","post_preocessing","transfer"],
                color_discrete_sequence=px.colors.qualitative.Bold,barmode='group')
        # fig.update_layout(width=1500,height=1000,hovermode="x")
        fig.update_layout(legend_title_text='<b> Module')
        fig.update_xaxes(title='Slide Name')
        fig.update_yaxes(title = 'Time(s)')
        return fig
    else:
        fig = px.bar(slide_wise, x="slide_name", y=['pick_slide_from_basket','gripper_status_time','move_to_scanner_to_place_slide','load_slide','place_slide','barcode_decoding','localization_time','micro_camera_conn','FS_time','total_acq_time','move_into_scanner_to_pick','pick_slide_from_scanner','unlock_slide','move_to_drop_basket','pick_slide_from_scanner', 'move_into_scanner_to_pick','move_to_home_from_scanner', 'move_to_home_from_drop_basket','drop_slide','tar_time','move_to_cluster_time','move_to_nas_time','move_dcm_time','move_img_time','untar_time','pyr_gen', 'grid_merging', 'blending_tiling', 'dicom_generation','white_generation'],
                color_discrete_sequence=px.colors.qualitative.Bold,barmode='group')
        # fig.update_layout(width=1500,height=1000,hovermode="x")
        fig.update_layout(legend_title_text='<b> Module')
        fig.update_xaxes(title='Slide Name')
        fig.update_yaxes(title = 'Time(s)')
        return fig


@app.callback(
    Output("slide_placement", "figure"),
    [Input("load", "value"),Input("property", "value")])
def slide_placement(load,property):

    first = df[df['loadIdentifier'] == str(load)]
    if property == "ALL":
        fig = px.bar(first,x="slide_name", y=['slide_placement','barcode_decoding','localization_time','micro_camera_conn','validation_time','FS_time','total_acq_time', 'slide_drop','post_preocessing','transfer'],
        color_discrete_sequence=px.colors.qualitative.Bold, title="All Modules")
        fig.update_layout(width=1500,height=1000,hovermode="x")
        fig.update_layout(legend_title_text='<b> Module')
        fig.update_xaxes(title='Slide Names')
        fig.update_yaxes(title = 'Time(s)')
        return fig
    else:
        fig = px.line(y = first[property],height= 800,title="Plot for "+str(property),hover_name=first["slide_name"])
        fig.update_xaxes(title = "No. of Slides")
        fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        fig.update_yaxes(title = "Time (s)",range=[first[property].min() - 10,first[property].max()+10])
        fig.add_hline(y=np.mean(first[property]),line_color="green")
        fig.update_traces(mode="markers+lines")
        fig.update_layout(legend_title_text='<b> Module')
        fig.add_hline(y=first[property].min(),line_dash='dash',line_color="red")
        fig.add_hline(y=first[property].max(),line_dash='dash',line_color="red")
        return fig


@app.callback(Output("store", "data"), [Input("button", "n_clicks")])
def generate_graphs(n):
    """
    This callback generates three simple graphs from random data.
    """
    if not n:
        # generate empty graphs when app loads
        return {k: go.Figure(data=[]) for k in ["scatter", "hist_1", "hist_2"]}

    # simulate expensive graph generation process
    time.sleep(2)

    # generate 100 multivariate normal samples
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)

    scatter = go.Figure(
        data=[go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers")]
    )
    hist_1 = go.Figure(data=[go.Histogram(x=data[:, 0])])
    hist_2 = go.Figure(data=[go.Histogram(x=data[:, 1])])

    # save figures in a dictionary for sending to the dcc.Store
    return {"scatter": scatter, "hist_1": hist_1, "hist_2": hist_2}


if __name__ == "__main__":
    app.run_server(debug=True)