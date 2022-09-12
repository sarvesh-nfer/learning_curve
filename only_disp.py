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
    df = pd.read_sql_query("select * from grid_info where grid_status = 10;",conn)

    lst = []
    for i,j in zip(df['grid_id'],df['best_row_to_start']):
        aoi = pd.read_sql_query("select * from aoi where grid_id = {} and aoi_row_idx = {};".format(i,(j-1)),conn)
        lst.extend(aoi['aoi_name'].to_list())


    primary = pd.read_csv(slide_path+"/disp_est_output/primary_dir.txt", sep="\t")
    secondary = pd.read_csv(slide_path+"/disp_est_output/secondary_dir.txt", sep="\t")

    primary.columns=['aoi_1','aoi_2','x','y','cost','foreground_ratio','mse','corr','aoi_status']
    secondary.columns=['aoi_1','aoi_2','x','y','cost','foreground_ratio','mse','corr','aoi_status']

    for i,j,k in zip(primary['x'],primary['y'],primary['aoi_status']):
        if k == 0:
            ref_x = i
            ref_y = j
            break
    primary['actual_x'] =primary['x']- ref_x
    primary['actual_y'] =primary['y']- ref_y

    for i,j,k in zip(secondary['x'],secondary['y'],secondary['aoi_status']):
        if k == 0:
            ref_x = i
            ref_y = j
            break
    secondary['actual_x'] =secondary['x']- ref_x
    secondary['actual_y'] =secondary['y']- ref_y


    fig5 = make_subplots(rows=1, cols=2,subplot_titles=("<b>Displacement in Primary Direction","<b>Displacement in Secondary Direction"))
    fig5.add_trace(go.Scatter(x = primary['actual_x'],y = primary['actual_y'],mode='markers',marker=dict(color='blue')),row=1,col=1)
    fig5.add_trace(go.Scatter(x = secondary['actual_x'],y = secondary['actual_y'],mode='markers',marker=dict(color='blue')),row=1,col=2)

    fig5.add_trace(go.Scatter(x = primary[(primary['actual_x']>=20)|(primary['actual_x']<=-20)]['actual_x'],
                            y = primary[(primary['actual_x']>=20)|(primary['actual_x']<=-20)]['actual_y'],
                            mode='markers',marker=dict(color='red',size=8)),row=1,col=1)

    fig5.add_trace(go.Scatter(x = primary[(primary['actual_y']>=20)|(primary['actual_y']<=-20)]['actual_x'],
                            y = primary[(primary['actual_y']>=20)|(primary['actual_y']<=-20)]['actual_y'],
                            mode='markers',marker=dict(color='red',size=8)),row=1,col=1)

    fig5.add_trace(go.Scatter(x = secondary[(secondary['actual_x']>=20)|(secondary['actual_x']<=-20)]['actual_x'],
                            y = secondary[(secondary['actual_x']>=20)|(secondary['actual_x']<=-20)]['actual_y'],
                            mode='markers',marker=dict(color='red',size=8)),row=1,col=2)

    fig5.add_trace(go.Scatter(x = secondary[(secondary['actual_y']>=20)|(secondary['actual_y']<=-20)]['actual_x'],
                            y = secondary[(secondary['actual_y']>=20)|(secondary['actual_y']<=-20)]['actual_y'],
                            mode='markers',marker=dict(color='red',size=8)),row=1,col=2)


    fig5.add_trace(go.Scatter(x = secondary[(secondary['aoi_1'].isin(lst))|(secondary['aoi_2'].isin(lst))]['actual_x'],
                            y = secondary[(secondary['aoi_1'].isin(lst))|(secondary['aoi_2'].isin(lst))]['actual_y'],
                            mode='markers',marker=dict(color='Black',size=8)),row=1,col=2)


    fig5.add_shape(type="rect",
        xref="x", yref="y",
        x0=-20, y0=20, x1=20, y1=-20,row="all",col='all',
        line_color="yellow")
    fig5.add_shape(type="rect",
        xref="x", yref="y",
        x0=-32, y0=32, x1=32, y1=-32,row="all",col='all',
        line_color="red")
    fig5.update_yaxes(title="Shift(px) in Y Direction",range=[-65,65])
    fig5.update_xaxes(title="Shift(px) in X Direction",range=[-65,65])
    fig5.add_vline(x=0,line=dict(color="red"),opacity=0.3)
    fig5.add_hline(y=0,line=dict(color="red"),opacity=0.3)

    fig5.add_annotation(text="<br><b>Average X-Shift : "+str(round(np.mean(primary['actual_x']),2))+"\t Average Y-Shift : <b>"+\
        str(round(np.mean(primary['actual_y']),2))+"</b><br>Max X-Shift : <b>"+str(round(max(primary['actual_x']),2))+\
                "</b>\t Max Y-Shift : <b>"+str(round(max(primary['actual_y']),2))+\
                "</b><br>Min X-Shift : <b>"+str(round(min(primary['actual_x']),2))+\
                "</b>\t Min Y-Shift : <b>"+str(round(min(primary['actual_y']),2))
                ,showarrow=False,font=dict(family="Courier New, monospace",size=12,color="black"),row=1,col=1,
            xref="paper", yref="paper",x=50, y=60,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)

    fig5.add_annotation(text="<br><b>Average X-Shift : "+str(round(np.mean(secondary['actual_x']),2))+"\t Average Y-Shift : <b>"+\
        str(round(np.mean(secondary['actual_y']),2))+"</b><br>Max X-Shift : <b>"+str(round(max(secondary['actual_x']),2))+\
                "</b>\t Max Y-Shift : <b>"+str(round(max(secondary['actual_y']),2))+\
                "</b><br>Min X-Shift : <b>"+str(round(min(secondary['actual_x']),2))+\
                "</b>\t Min Y-Shift : <b>"+str(round(min(secondary['actual_y']),2))
                ,showarrow=False,font=dict(family="Courier New, monospace",size=12,color="black"),row=1,col=2,
            xref="paper", yref="paper",x=45, y=60,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)

    fig5.update_layout(showlegend=False,font=dict(family="Courier New, monospace",size=16,color="Black"),width=1700,height=800)
    fig5.update_xaxes(showspikes=True)
    fig5.update_yaxes(showspikes=True)
    fig5.add_annotation(text="<b>Displacement Data For : "+slide_path.split('/')[-1],xref="paper", yref="paper",showarrow=False,x=0, y=1.11,font=dict(family="Courier New, monospace",
            size=24,color="RebeccaPurple"))
    # fig5.show()
    if (len(primary[(abs(primary['actual_x']) < 20)&(abs(primary['actual_y']) < 20)])/len(primary) *100) > 95:

        fig5.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
                text="Population Distribution <br>within ±20PX : "+str(round(len(primary[(abs(primary['actual_x']) < 20)&(abs(primary['actual_y']) < 20)])/len(primary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=1)
    else:
        fig5.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
                text="Population Distribution <br>within ±20PX : "+str(round(len(primary[(abs(primary['actual_x']) < 20)&(abs(primary['actual_y']) < 20)])/len(primary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=1)

    if (len(primary[(abs(primary['actual_x']) < 32)&(abs(primary['actual_y']) < 32)])/len(primary) *100) > 99:

        fig5.add_annotation(x=0,y=0.9,xref="x domain",yref="y domain",
                text="Population Distribution <br>within ±32PX : "+str(round(len(primary[(abs(primary['actual_x']) < 32)&(abs(primary['actual_y']) < 32)])/len(primary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=1)
    else:
        fig5.add_annotation(x=0,y=0.9,xref="paper",yref="paper",
                text="Population Distribution <br>within ±32PX : "+str(round(len(primary[(abs(primary['actual_x']) < 32)&(abs(primary['actual_y']) < 32)])/len(primary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=1)



    if (len(secondary[(abs(secondary['actual_x']) < 20)&(abs(secondary['actual_y']) < 20)])/len(secondary) *100) > 95:

        fig5.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
                text="Population Distribution <br>within ±20PX : "+str(round(len(secondary[(abs(secondary['actual_x']) < 20)&(abs(secondary['actual_y']) < 20)])/len(secondary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=2)
    else:
        fig5.add_annotation(x=0,y=1,xref="x domain",yref="y domain",
                text="Population Distribution <br>within ±20PX : "+str(round(len(secondary[(abs(secondary['actual_x']) < 20)&(abs(secondary['actual_y']) < 20)])/len(secondary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="red",opacity=0.8,row=1,col=2)

    if (len(secondary[(abs(secondary['actual_x']) < 32)&(abs(secondary['actual_y']) < 32)])/len(secondary) *100) > 99:

        fig5.add_annotation(x=0,y=0.9,xref="x domain",yref="y domain",
                text="Population Distribution <br>within ±32PX : "+str(round(len(secondary[(abs(secondary['actual_x']) < 32)&(abs(secondary['actual_y']) < 32)])/len(secondary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=2)
    else:
        fig5.add_annotation(x=0,y=0.9,xref="paper",yref="paper",
                text="Population Distribution <br>within ±32PX : "+str(round(len(secondary[(abs(secondary['actual_x']) < 32)&(abs(secondary['actual_y']) < 32)])/len(secondary) *100,2))+"%",
                showarrow=False,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="red",opacity=0.8,row=1,col=2)

    fig5.write_image(slide_path+"/Displacement_plot_"+slide_path.split('/')[-1]+".png")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    # print(slide_path)
    # direct_plot(slide_path)
    # lst = ["H01BBB27P-2_i11"]
    # direct_plot(path)

    for i in os.listdir(path):
        try:
            slide_path = path+"/"+i
            direct_plot(slide_path)
        except Exception as msg:
            print("yes ",msg)