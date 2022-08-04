import pandas as pd
import plotly.express as px
import sqlite3
import sys


def plot_shift(path):

    try:
        conn = sqlite3.connect(path)

        df = pd.read_sql_query("select * from registration_info;",conn)
        df = df.sort_values(by=['stack_index_1'])
        df['axes'] = ["img_"+str(y)+" & img_" + str(x) if x > y else "img_"+str(x)+" & img_" + str(y) for x,y in zip(df['stack_index_1'],df['stack_index_2'])]
        #df
        print(df['axes'].nunique())

        fig4 = px.scatter(x=df['x_shift'],y=df['y_shift'],facet_col=df['axes'],facet_col_wrap=2,
                        facet_col_spacing=0.025,)
        fig4.for_each_annotation(lambda a: a.update(text="<b> Stack Wise Shift for : "+a.text.split("=")[-1]))
        fig4.update_yaxes(title="Y-Shift Values",showticklabels=True)
        fig4.update_xaxes(title="X-shift Values",showticklabels=True,tickangle=45)
        fig4.update_layout(width=1800,height=1400)
        fig4.update_traces(marker_color='green')
        fig4.add_shape(type="rect",
        xref="x", yref="y",
        x0=-12, y0=12, x1=12, y1=-12,row="all",col='all',
        line_color="red")
        fig4.show()
        fig4.update_layout(title="Shift Across Stack for IHC Slide")
        fig4.write_image("/home/adminspin/Desktop/I.png")
    except Exception as error:
        print("ERROR : ",error)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print("Please enter DB path")
    

    plot_shift(path)
