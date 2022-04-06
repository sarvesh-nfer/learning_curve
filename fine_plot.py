import pandas as pd
import plotly.express as px
import os

def fine_plot(path):
    df = pd.read_csv(path)
    slide = path.split("/")[-3]
    file = path.split("/")[-1].split(".")[0]
    fig = (px.scatter(y = df['Focus Metric']).update_traces(mode='lines+markers'))
    fig.update_layout(title="Best-Z Focus Metric for Slide :<b>"+slide)
    fig.update_xaxes(title="Stack Index")
    fig.update_yaxes(title="Focus Metric Value")
    fig.add_hline(y=7)
    fig.write_image(os.path.split(path)[0]+"/"+file+"_fm.png")
    fig = (px.scatter(y = df['Color Metric']).update_traces(mode='lines+markers'))
    fig.update_layout(title="Best-Z Color Metric for Slide :<b>"+slide)
    fig.update_xaxes(title="Stack Index")
    fig.update_yaxes(title="Color Metric Value")
    fig.add_hline(y=40)
    fig.write_image(os.path.split(path)[0]+"/"+file+"_cm.png")
    fig = (px.scatter(y = df['Hue Metric']).update_traces(mode='lines+markers'))
    fig.update_layout(title="Best-Z Hue Metric for Slide :<b>"+slide)
    fig.update_xaxes(title="Stack Index")
    fig.update_yaxes(title="Hue Metric Value")
    fig.add_hline(y=1000)
    fig.write_image(os.path.split(path)[0]+"/"+file+"_hm.png")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        slide_path = sys.argv[1]
    fine_plot(slide_path)
