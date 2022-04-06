import time
import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.express as px
import pymongo
import json
import os
from pymongo import MongoClient
# df=pd.read_csv("/home/adminspin/Desktop/df2.csv")
client = MongoClient()
db1 = client.viewerDB

collection1 = db1.slide_timings


datapoints1 = list(db1.slide_timings.find({}))
datapoints1

df = pd.DataFrame()
count = 1
for i in range(len(datapoints1)):
    try:
        robo = pd.json_normalize(datapoints1[i]['robot_time_log'])

#         local = pd.json_normalize(datapoints1[i]['localization_time_log'])
        cluster = pd.json_normalize(datapoints1[i]['cluster_transfer_time_log'])
        scanner = pd.json_normalize(datapoints1[i]['scanner_transfer_time_log'])
        post = pd.json_normalize(datapoints1[i]['post_process_time_log'])
#         restore = pd.json_normalize(datapoints1[i]['restoration_time_log'])
#         acq = pd.json_normalize(datapoints1[i]['acquisition_time_log']['grid_info'])
        post['total_acq_time'] = pd.json_normalize(datapoints1[i]['acquisition_time_log']['grid_info'])['image_capture_time'].sum()
        post['validation_time'] = pd.json_normalize(datapoints1[i]['localization_time_log'])['validation_time']
        post['localization_time'] = datapoints1[i]['localization_time_log']['time_to_detect_bounding_boxes']
        post['barcode_decoding'] = datapoints1[i]['localization_time_log']['time_taken_for_barcode_decoding']
#         post['onex_conn_time'] = pd.json_normalize(datapoints1[i]['localization_time_log'])['total_time_for_entire_localization_process'].values - (pd.json_normalize(datapoints1[i]['localization_time_log'])['validation_time'].values + pd.json_normalize(datapoints1[i]['localization_time_log'])['localization_time'].values)
        post['micro_camera_conn'] = datapoints1[i]['localization_time_log']['connecting_to_the_micro_imaging_camera']
        post['slide_name'] = datapoints1[i]['slideName']
        post['loadIdentifier'] = datapoints1[i]['loadIdentifier']
        post['scannerId'] = datapoints1[i]['scannerId']
        post['image_capture_time'] = pd.json_normalize(datapoints1[i]['acquisition_time_log']['grid_info'])['image_capture_time'].sum()
        post['FS_time'] = pd.json_normalize(datapoints1[i]['acquisition_time_log']['grid_info'])['focus_sampling_time'].sum()
        
        post['total_fg'] = pd.json_normalize(datapoints1[i]['acquisition_time_log']['grid_info'])['fg_count'].sum()
        post['total_bg'] = pd.json_normalize(datapoints1[i]['acquisition_time_log']['grid_info'])['bg_count'].sum()

        df1 = pd.concat([post,scanner,cluster,robo],axis=1)
        df= df.append(df1)
    except Exception as msg:
        print("error in : ",i,"\t",count)
        print("msg : ",msg)
        count+=1


df['slide_placement'] = df['pick_slide_from_basket'] + df['gripper_status_time'] + df['move_to_scanner_to_place_slide'] + df['load_slide']+df['place_slide']
df['slide_drop'] = df['drop_slide'] + df['unlock_slide'] + df['move_to_drop_basket'] + df['pick_slide_from_scanner']+ df['move_into_scanner_to_pick']+ df['move_to_home_from_scanner'] + df['move_to_home_from_drop_basket']
df['post_preocessing'] = df['pyr_gen'] + df['grid_merging'] + df['blending_tiling'] + df['dicom_generation'] + df['white_generation'] + df['tar_time']
df['transfer'] = df['move_to_cluster_time'] + df['move_to_nas_time'] + df['move_dcm_time'] + df['move_img_time'] +df['untar_time']

print("!!DF saved!!")
if not os.path.exists("/home/adminspin/Music/scripts/dftimings"):
    os.makedirs("/home/adminspin/Music/scripts/dftimings")
df.to_csv("/home/adminspin/Music/scripts/dftimings"+str(time.strftime("%d-%m-%Y"))+".csv",index=False)

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
                for x in sorted(df['loadIdentifier'].unique())],value=sorted(df['loadIdentifier'].unique())[-1],clearable=False)],style=dict(width='50%')),

        
        html.Div(children=[
                html.Label(['Model Type'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(id="property",options=[{"label": x, "value": x}
                for x in ['ALL','pick_slide_from_basket','gripper_status_time','move_to_scanner_to_place_slide','load_slide','place_slide','barcode_decoding','localization_time','validation_time',
        'micro_camera_conn','FS_time','total_acq_time','move_into_scanner_to_pick','pick_slide_from_scanner','unlock_slide','move_to_drop_basket','pick_slide_from_scanner', 'move_into_scanner_to_pick','move_to_home_from_scanner',
         'move_to_home_from_drop_basket','drop_slide','tar_time','move_to_cluster_time','move_to_nas_time','move_dcm_time','move_img_time','untar_time','pyr_gen', 'grid_merging', 'blending_tiling', 'dicom_generation','white_generation']],
                value="ALL",clearable=False)],style=dict(width='50%'))],style=dict(display='flex'))),
    
    dcc.Graph(id="slide_placement")
    
])

tab2 = html.Div([
html.Div(
    
    html.Div(className='row', children=[
        html.Div(children=[
            html.Label(['Slide Name'], style={'font-weight': 'bold', "text-align": "center"}),
                dcc.Dropdown(id="slide_name",options=[{"label": x, "value": x}
                for x in df['slide_name'].unique()],value=df['slide_name'].unique()[-1],clearable=False),
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
        
        fig = px.bar(slide_wise, x="slide_name", y=["slide_placement",'barcode_decoding','localization_time','micro_camera_conn','validation_time','FS_time','total_acq_time', "slide_drop","post_preocessing","transfer"],template='simple_white',
                color_discrete_sequence=px.colors.qualitative.Bold,barmode='group')
        # fig.update_layout(width=1500,height=1000,hovermode="x")
        fig.update_layout(legend_title_text='<b> Module')
        fig.update_xaxes(title='Slide Name')
        fig.update_yaxes(title = 'Time(s)')
        if len(slide_wise) > 1:
            fig.add_annotation(text="Total foreground AOIs : <b>"+str(int(slide_wise['total_fg'].iloc[0]))+"</b><br>Total Background AOIs : <b>"+str(int(slide_wise['total_bg'].iloc[0])),showarrow=False,
            xref="paper", yref="paper",x=1.22, y=1.15,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
        else:
            fig.add_annotation(text="Total foreground AOIs : <b>"+str(int(slide_wise['total_fg']))+"</b><br>Total Background AOIs : <b>"+str(int(slide_wise['total_bg'])),showarrow=False,
            xref="paper", yref="paper",x=1.22, y=1.15,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
        return fig
    else:
        fig = px.bar(slide_wise, x="slide_name", y=['pick_slide_from_basket','gripper_status_time','move_to_scanner_to_place_slide','load_slide','place_slide','barcode_decoding','localization_time','validation_time',
        'micro_camera_conn','FS_time','total_acq_time','move_into_scanner_to_pick','pick_slide_from_scanner','unlock_slide','move_to_drop_basket','pick_slide_from_scanner', 'move_into_scanner_to_pick','move_to_home_from_scanner',
         'move_to_home_from_drop_basket','drop_slide','tar_time','move_to_cluster_time','move_to_nas_time','move_dcm_time','move_img_time','untar_time','pyr_gen', 'grid_merging', 'blending_tiling', 'dicom_generation','white_generation'],
         template='simple_white',color_discrete_sequence=px.colors.qualitative.Bold,barmode='group')
        # fig.update_layout(width=1500,height=1000,hovermode="x")
        fig.update_layout(legend_title_text='<b> Module')
        fig.update_xaxes(title='Slide Name')
        fig.update_yaxes(title = 'Time(s)')
        if len(slide_wise) > 1:
            fig.add_annotation(text="Total foreground AOIs : <b>"+str(int(slide_wise['total_fg'].iloc[0]))+"</b><br>Total Background AOIs : <b>"+str(int(slide_wise['total_bg'].iloc[0])),showarrow=False,
            xref="paper", yref="paper",x=1.28, y=1.15,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
        else:
            fig.add_annotation(text="Total foreground AOIs : <b>"+str(int(slide_wise['total_fg']))+"</b><br>Total Background AOIs : <b>"+str(int(slide_wise['total_bg'])),showarrow=False,
            xref="paper", yref="paper",x=1.28, y=1.15,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
        return fig


@app.callback(
    Output("slide_placement", "figure"),
    [Input("load", "value"),Input("property", "value")])
def slide_placement(load,property):

    first = df[df['loadIdentifier'] == str(load)]
    if property == "ALL":
        fig = px.bar(first,x="slide_name", y=['slide_placement','barcode_decoding','localization_time','micro_camera_conn','validation_time','FS_time','total_acq_time', 'slide_drop','post_preocessing','transfer'],template='simple_white',
        color_discrete_sequence=px.colors.qualitative.Bold, title="All Modules")
        fig.update_layout(width=1500,height=1000,hovermode="x")
        fig.update_layout(legend_title_text='<b> Module')
        fig.update_xaxes(title='Slide Names')
        fig.update_yaxes(title = 'Time(s)')
        return fig
    else:
        fig = px.line(y = first[property],height= 800,title="Plot for "+str(property),hover_name=first["slide_name"])
        fig.update_xaxes(title = "No. of Slides")
        fig.update_yaxes(title = "Time (s)",range=[first[property].min() - 10,first[property].max()+10])
        fig.add_hline(y=np.mean(first[property]),line_color="green")
        fig.update_traces(mode="markers+lines")
        fig.update_layout(legend_title_text='<b> Module')
        fig.update_layout(hoverlabel=dict(bgcolor="lightyellow",font_size=16,font_family="Rockwell"))
        fig.add_hline(y=first[property].min(),line_dash='dash',line_color="red")
        fig.add_hline(y=first[property].max(),line_dash='dash',line_color="red")
        fig.add_annotation(text="<b>Total Time taken :"+str(round(sum(first[property]),2))+"<br>Average :<b>"+str(round(np.mean(first[property]),2))+"</b><br>Max value : <b>"+str(round(max(first[property]),2))+"</b><br>Min value : <b>"+str(round(min(first[property]),2)),showarrow=False,
            font=dict(family="Courier New, monospace",size=18,color="black"),xref="paper", yref="paper",x=0.9, y=1.15,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
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
