try:
    import pandas as pd
    import sqlite3
    import sys,os
    import glob
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import time
except Exception as msg:
    print("notthere")


def replace_shift(path):

    if os.path.exists(glob.glob(path+"/*.db")[0]):

        if not os.path.exists(path+"/Shift_analysis"):
            os.makedirs(path+"/Shift_analysis")
        
        conn = sqlite3.connect(glob.glob(path+"/*.db")[0])

        dfq = pd.DataFrame()

        df = pd.read_sql_query("select * from registration_info;",conn)
        aoi = pd.read_sql_query("select * from aoi where focus_metric >= 7 and color_metric >= 40 and bg_state_acq = 0 and capture_status = 1",conn)
        df = df[df['aoi_name'].isin(aoi["aoi_name"].to_list())].reset_index(drop=True)

        df['axes'] = ["img_"+str(y)+" & img_" + str(x) if x > y else "img_"+str(x)+" & img_" + str(y) for x,y in zip(df['stack_index_1'],df['stack_index_2'])]
        df.to_csv(path+"/Shift_analysis"+"/all_shift_data.csv",index=False)
        df = df.sort_values(by=['stack_index_1'])

        dfx = pd.DataFrame()
        dfy = pd.DataFrame()
        aoi_name = []
        max_xshift = []
        max_yshift = []
        maxX_index = []
        maxY_index = []
        for i in df['aoi_name'].unique():
            idx = df[df['aoi_name']==i]['x_shift'].abs().idxmax()
            idy = df[df['aoi_name']==i]['y_shift'].abs().idxmax()

            aoi_name.append(i)
            max_xshift.append(df.iloc[idx][4])
            max_yshift.append(df.iloc[idy][5])
            maxX_index.append(df.iloc[idx][-1])
            maxY_index.append(df.iloc[idy][-1])
            
            dfx = pd.concat([df.iloc[[idx]], dfx], axis=0).reset_index(drop=True)
            dfy = pd.concat([df.iloc[[idy]], dfy], axis=0).reset_index(drop=True)

        pd.DataFrame(list(zip(aoi_name,max_xshift,max_yshift,maxX_index,maxY_index)),columns=["aoi_name","max_Xshift","max_Yshift","index_Xshift","index_Yshift"]).\
            to_csv(path+"/Shift_analysis"+"/max_shift.csv",index=False)

        if not os.path.exists(path+"/Shift_analysis/Analysis_plot"):
            os.makedirs(path+"/Shift_analysis/Analysis_plot")
        #overall_plot
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
        fig3.add_annotation(text="<b>Stack Shift Data for <b>S18-14571-1-57-1_H01EBB47P-15858",xref="paper", yref="paper",showarrow=False,x=0, y=1.11,font=dict(family="Courier New, monospace",
                size=24,color="RebeccaPurple"))
        fig3.add_vline(x=-11.5, line_width=3, line_dash="dash", line_color="red")
        fig3.add_vline(x=11.5, line_width=3, line_dash="dash", line_color="red")


        if round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2) > 95:

            fig3.add_annotation(x=16,y=(df['x_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
                    text="Percentage of distribution <br>of X-Shift : "+str(round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2)),
                    showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                    borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=1)
        else:
            fig3.add_annotation(x=16,y=(df['x_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
                text="Percentage of distribution of X-Shift : "+str(round((len(df[(df['x_shift']<=12)&(df['x_shift']>=-12)])/len(df))*100,2)),
                showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="crimson",opacity=0.8,row=1,col=1)

        if round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2) > 95:
            fig3.add_annotation(x=16,y=(df['y_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
                    text="Percentage of distribution <br>of Y-Shift : "+str(round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2)),
                    showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                    borderwidth=2,borderpad=4,bgcolor="green",opacity=0.8,row=1,col=2)
        else:
            fig3.add_annotation(x=16,y=(df['y_shift'].value_counts().iloc[0]),xref="paper",yref="paper",
                text="Percentage of distribution <br>of Y-Shift : "+str(round((len(df[(df['y_shift']<=12)&(df['y_shift']>=-12)])/len(df))*100,2)),
                showarrow=True,font=dict(family="Courier New, monospace",size=16,color="#ffffff"),align="center",bordercolor="#c7c7c7",
                borderwidth=2,borderpad=4,bgcolor="crimson",opacity=0.8,row=1,col=2)

        fig3.write_image(path+"/Shift_analysis"+"/shift.png")


        ### XY cluster
        fig4 = px.scatter(x=df['x_shift'],y=df['y_shift'],facet_col=df['axes'],facet_col_wrap=2,facet_col_spacing=0.025,)
        fig4.for_each_annotation(lambda a: a.update(text="<b> Stack Wise Shift for : "+a.text.split("=")[-1]))
        fig4.update_yaxes(title="Y-Shift Values",showticklabels=True)
        fig4.update_xaxes(title="X-shift Values",showticklabels=True,tickangle=45)
        fig4.update_layout(width=1800,height=1400)
        fig4.update_traces(marker_color='green')
        fig4.add_shape(type="rect",
            xref="x", yref="y",
            x0=-12, y0=12, x1=12, y1=-12,row="all",col='all',
            line_color="red")
        fig4.write_image(path+"/Shift_analysis"+"/XY_cluster.png")

        start_time = time.time()
        # make plot AOI wise
        for i in df['aoi_name'].unique():
            try:
                df2 = df[df['aoi_name'] == i]

                fig = make_subplots(rows=1, cols=2,subplot_titles=("X-Shift","Y-Shift"))

                fig.add_trace(
                    go.Scatter(y=df2['axes'],x=df2['x_shift'],mode='lines+markers+text',name="X-Shift",marker=dict(color="blue"),
                    text=df2['x_shift'],texttemplate = "%{text}",textposition = "bottom center"),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(y=df2['axes'],x=df2['y_shift'],mode='lines+markers+text',name="Y-Shift",marker=dict(color="purple"),
                    text=df2['y_shift'],texttemplate = "%{text}",textposition = "bottom center"),
                    row=1, col=2
                )
                fig.update_xaxes(dtick=5,tickangle=0,range=[-17,17],title="Shift Values")
                fig.update_layout(title="AOI : <b>"+i,width=1200)
                fig.write_image(path+"/Shift_analysis/Analysis_plot"+"/"+i+".png")
            except Exception as msg:
                print("AOI wise plot error for : ",i)
        print("Time taken to for 1 DB file",round((time.time() - start_time),2)) 
      
    else:
        print("error No db file Present")

    
            
if __name__ == "__main__":
    if len(sys.argv) == 2:
        slide_path = sys.argv[1]
    replace_shift(slide_path)