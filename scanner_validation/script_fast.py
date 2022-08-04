import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from datetime import date
import sqlite3
import sys


def plot_mutiple(path):

    try:

        post = pd.read_csv(glob.glob(path+"/*centering_check.csv")[0])


        fig2 = make_subplots(rows=1, cols=2,subplot_titles=("<b>Centring","<b>Illuminaion"))
        fig2.add_trace(go.Scatter(x=post['CENTRING_COORDINATE_X'],y=post['CENTRING_COORDINATE_Y'],
                                mode='markers',
                                name='Dispersion',
                                ),row=1,col=1)
        fig2.update_yaxes(title="Stage Y",range=[1216,0],row=1,col=1,ticks="outside", tickwidth=2, tickcolor='crimson')
        fig2.update_xaxes(title="Stage X",range=[0,1936],row=1,col=1,ticks="outside", tickwidth=2, tickcolor='crimson')
        fig2.add_shape(type="rect",
            xref="x", yref="y",
            x0=768, y0=408, x1=1168, y1=808,
            line_color="red",row=1,col=1
        )
        fig2.add_vline(x=968,line=dict(color="red"),opacity=0.3,row=1,col=1)
        fig2.add_hline(y=608,line=dict(color="red"),opacity=0.3,row=1,col=1)
        fig2.add_annotation(text="<b>Number of Steps : <b>"+str(len(post))+"<br><b>Average X-Shift : "+\
                        str(round(np.mean(post['CENTRING_X_DIFFERENCE']),2))+"\t Average Y-Shift : <b>"+\
            str(round(np.mean(post['CENTRING_Y_DIFFERENCE']),2))+"</b><br>Max X-Shift : <b>"+str(round(max(post['CENTRING_X_DIFFERENCE']),2))+\
                    "</b>\t Max Y-Shift : <b>"+str(round(max(post['CENTRING_Y_DIFFERENCE']),2))+\
                    "</b><br>Min X-Shift : <b>"+str(round(min(post['CENTRING_X_DIFFERENCE']),2))+\
                    "</b>\t Min Y-Shift : <b>"+str(round(min(post['CENTRING_Y_DIFFERENCE']),2))
                    ,showarrow=False,font=dict(family="Courier New, monospace",size=12,color="black"),row=1,col=1,
                xref="paper", yref="paper",x=1800, y=5,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
        ##adding illumination
        fig2.add_trace(go.Scatter(y=post['MEAN_RED_INTENSITY'],mode='lines',name='Mean Red',
                                line=dict(color='red')),row=1,col=2)
        fig2.add_trace(go.Scatter(y=post['MEAN_BLUE_INTENSITY'],mode='lines',name='Mean Blue',
                                line=dict(color='blue')),row=1,col=2)
        fig2.add_trace(go.Scatter(y=post['MEAN_GREEN_INTENSITY'],mode='lines', name='Mean Green',
                                line=dict(color='green')),row=1,col=2)
        fig2.update_yaxes(title="Intensity",range=[150, 260],row=1,col=2)
        fig2.update_xaxes(title="Z Steps",row=1,col=2)

        fig2.update_layout(showlegend=False,font=dict(family="Courier New, monospace",size=16,color="Black"),width=1800,height=800)
        fig2.add_annotation(text="<b>Calibration Data for <b>"+str(glob.glob(path+"/*.db")[0].split("/")[-1].split("-")[0]),xref="paper", yref="paper",showarrow=False,x=0, y=1.11,font=dict(family="Courier New, monospace",
                size=24,color="RebeccaPurple"))
        # fig2.show()
        print("CENTERING DONE")
        print(glob.glob(path+"/*.db")[0])
        conn = sqlite3.connect(glob.glob(path+"/*.db")[0])

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

        primary = pd.read_csv(path+"/disp_est_output/primary_dir.txt", sep="\t")
        secondary = pd.read_csv(path+"/disp_est_output/secondary_dir.txt", sep="\t")
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
                                mode='markers',marker=dict(color='yellow',size=8)),row=1,col=1)

        fig5.add_trace(go.Scatter(x = primary[(primary['actual_y']>=20)|(primary['actual_y']<=-20)]['actual_x'],
                                y = primary[(primary['actual_y']>=20)|(primary['actual_y']<=-20)]['actual_y'],
                                mode='markers',marker=dict(color='yellow',size=8)),row=1,col=1)

        fig5.add_trace(go.Scatter(x = secondary[(secondary['actual_x']>=20)|(secondary['actual_x']<=-20)]['actual_x'],
                                y = secondary[(secondary['actual_x']>=20)|(secondary['actual_x']<=-20)]['actual_y'],
                                mode='markers',marker=dict(color='yellow',size=8)),row=1,col=2)

        fig5.add_trace(go.Scatter(x = secondary[(secondary['actual_y']>=20)|(secondary['actual_y']<=-20)]['actual_x'],
                                y = secondary[(secondary['actual_y']>=20)|(secondary['actual_y']<=-20)]['actual_y'],
                                mode='markers',marker=dict(color='yellow',size=8)),row=1,col=2)


        fig5.add_trace(go.Scatter(x = primary[(primary['actual_x']>=32)|(primary['actual_x']<=-32)]['actual_x'],
                                y = primary[(primary['actual_x']>=32)|(primary['actual_x']<=-32)]['actual_y'],
                                mode='markers',marker=dict(color='red',size=8)),row=1,col=1)

        fig5.add_trace(go.Scatter(x = primary[(primary['actual_y']>=32)|(primary['actual_y']<=-32)]['actual_x'],
                                y = primary[(primary['actual_y']>=32)|(primary['actual_y']<=-32)]['actual_y'],
                                mode='markers',marker=dict(color='red',size=8)),row=1,col=1)

        fig5.add_trace(go.Scatter(x = secondary[(secondary['actual_x']>=32)|(secondary['actual_x']<=-32)]['actual_x'],
                                y = secondary[(secondary['actual_x']>=32)|(secondary['actual_x']<=-32)]['actual_y'],
                                mode='markers',marker=dict(color='red',size=8)),row=1,col=2)

        fig5.add_trace(go.Scatter(x = secondary[(secondary['actual_y']>=32)|(secondary['actual_y']<=-32)]['actual_x'],
                                y = secondary[(secondary['actual_y']>=32)|(secondary['actual_y']<=-32)]['actual_y'],
                                mode='markers',marker=dict(color='red',size=8)),row=1,col=2)

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
        fig5.add_annotation(text="<b>Displacement Data for <b>"+str(glob.glob(path+"/*.db")[0].split("/")[-1].split("-")[0]),xref="paper", yref="paper",showarrow=False,x=0, y=1.11,font=dict(family="Courier New, monospace",
                size=24,color="RebeccaPurple"))
        fig5.add_annotation(text="Outside<br>Spec limit", align='left',showarrow=False,
                            xref='paper',yref='paper',x=1,y=0.9,bordercolor='red',borderwidth=2)
        fig5.add_annotation(text="Outside<br>Warning limit", align='left',showarrow=False,
                            xref='paper',yref='paper',x=1,y=0.8,bordercolor='yellow',borderwidth=2)

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
        # fig5.show()
        print("DISPLACEMENT DONE")

        with open(path+'/'+str(glob.glob(path+"/*.db")[0].split("/")[-1].split("-")[0])+'_report.html', 'a') as f:
            f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig5.to_html(full_html=False, include_plotlyjs='cdn'))
    except Exception as error:
        print("ERROR : ",error)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
    

    plot_mutiple(path)
