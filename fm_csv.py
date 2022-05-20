import pandas as pd
import plotly.express as px
import os,sys


def fmcsv(path):
    df = pd.read_csv(path)

    lst = []
    for c in range(len(df)):
        lst.append("block_"+str(c))
        
    df['block_index'] = lst


    df2 = df.melt(id_vars=["block_index"], var_name='Stack_index', value_name='FM')

    df2['block_value'] = [x.split("_")[-1] for x in df2['block_index']]

    df2['block_value'] = df2['block_value'].astype(int)


    fig = px.scatter(x=df2['Stack_index'],y=df2['FM'],facet_col=df2['block_value'],facet_col_wrap=7,
        category_orders={"facet_col": [0,5,10,15,20,25,30,1,6,11,16,21,26,31,2,7,12,17,22,27,32
            ,3,8,13,18,23,28,33,4,9,14,19,24,29,34]}).update_traces(mode="lines+markers")
    fig.add_hline(y=7,row="all",col="all",line_width=3, line_dash="dash", line_color="red")
    fig.update_yaxes(title='',title_font=dict(size=20),showticklabels=True)
    fig.update_xaxes(title='',title_font=dict(size=20),showticklabels=True)
    fig.update_layout(title="AOI : <b>"+path.split("/")[-1].split(".")[0],width=1600,height=1400,yaxis15=dict(title="Focus Metric"),xaxis4=dict(title="Stack Index"))
    fig.for_each_annotation(lambda a: a.update(text="<b>Blob Index :"+a.text.split("=")[-1]))

    # fig.show()
    fig.write_image(os.path.split(path)[0]+"/"+path.split("/")[-1].split(".")[0]+".png")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]

    fmcsv(path)