import glob,os,sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def post_CI(path):
    post = pd.read_csv(path)


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
            xref="paper", yref="paper",x=1600, y=1,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
    ##adding illumination
    fig2.add_trace(go.Scatter(y=post['MEAN_RED_INTENSITY'],mode='lines',name='Mean Red',
                             line=dict(color='red')),row=1,col=2)
    fig2.add_trace(go.Scatter(y=post['MEAN_BLUE_INTENSITY'],mode='lines',name='Mean Blue',
                             line=dict(color='blue')),row=1,col=2)
    fig2.add_trace(go.Scatter(y=post['MEAN_GREEN_INTENSITY'],mode='lines', name='Mean Green',
                             line=dict(color='green')),row=1,col=2)
    fig2.update_yaxes(title="Intensity",range=[150, 260],row=1,col=2)
    fig2.update_xaxes(title="Z Steps",row=1,col=2)

    fig2.update_layout(hovermode="x unified",showlegend=False,font=dict(family="Courier New, monospace",size=16,color="Black"),width=1800,height=800)
    fig2.add_annotation(text="<b>Calibration Data for <b>: "+path.split("/")[-1].split(".")[0],xref="paper", yref="paper",showarrow=False,x=0, y=1.11,font=dict(family="Courier New, monospace",
            size=24,color="RebeccaPurple"))
    fig2.add_annotation(text="<b>Number of Steps : <b>"+str(len(post))+"<br><b>Min μ Red : "+\
                   str(round(min(post['MEAN_RED_INTENSITY']),2))+"\t | Max μ Red : <b>"+\
    str(round(max(post['MEAN_RED_INTENSITY']),2))+"</b><br> Min μ Green : <b>"+str(round(min(post['MEAN_GREEN_INTENSITY']),2))+\
               "</b>\t | Max μ Green : <b>"+str(round(max(post['MEAN_GREEN_INTENSITY']),2))+\
               "</b><br> Min μ Blue : <b>"+str(round(min(post['MEAN_BLUE_INTENSITY']),2))+\
               "</b>\t | Max μ Blue : <b>"+str(round(max(post['MEAN_BLUE_INTENSITY']),2))
               ,showarrow=False,font=dict(family="Courier New, monospace",size=12,color="black"),row=1,col=2,
                x=60, y=254,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ffffff",opacity=0.8)
#     fig2.show()
    fig2.write_image(os.path.split(path)[0]+"/"+path.split("/")[-1]+".png")
    #fig2.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]

    post_CI(path)
    
    # for i in glob.glob(path+"/*.csv"):
    #     post_CI(i)
    #     print("Saved for : ",os.path.split(i)[-1])
