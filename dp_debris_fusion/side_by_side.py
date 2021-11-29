import sqlite3
import plotly.express as px
import os,glob
import pandas as pd
import cv2,sys
import io
import numpy as np
from PIL import Image

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

def side2side(slide_path):
    db_path = glob.glob(slide_path+"/*.db")[0]
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("select * from fusion_info ;", conn)

    txt = glob.glob(slide_path+"/grid_*/raw_images/*.bmp")

    for i in txt:
        try:

            g = i.split('/')[-3].split('_')[-1]
            aoi = i.split('/')[-1].split(".")[0]

            fig = px.bar(x=df[(df['aoi_name'] == aoi)&(df['grid_id'] == int(g))]['stack_index'],
            y=df[(df['aoi_name'] == aoi)&(df['grid_id'] == int(g))]['percentage'],
            text=round(df[(df['aoi_name'] == aoi)&(df['grid_id'] == int(g))]['percentage'],2),
            height=1192,width=1912)
            fig.update_xaxes(title='<b>Stack Index')
            fig.update_yaxes(title='<b>Focused Content %',range=[0,60])
            fig.update_layout(uniformtext_minsize=50, uniformtext_mode='hide',title="Distribution of Focused Content for AOI: <b>"+aoi+"</b> and grid: <b>"+g)

            # fig.write(os.path.split(os.path.split(i)[0])[0]+"/plot/"+aoi+".bmp")

            img1 = cv2.imread(i)
            img2 = plotly_fig2array(fig)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


            im_h = cv2.hconcat([img1, img2])
            if not os.path.exists(slide_path+"/"+i.split('/')[-3]+"/side2side"):
                os.makedirs(slide_path+"/"+i.split('/')[-3]+"/side2side")
            cv2.imwrite(slide_path+"/"+i.split('/')[-3]+"/side2side/"+aoi+".png",im_h)

            print("saved for : ",i)
        except Exception as msg:
            print("ERROR in : ",i)





if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("Inavlid input arguments\n\n<python merged.py>\n\t"\
                "1.Input Slide Path\n")
    path =sys.argv[1]
    side2side(path)
