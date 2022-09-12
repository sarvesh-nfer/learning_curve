import pandas as pd
import plotly.express as px
import sys

def generate_html_1(path,cluster):

    df = pd.read_excel(path,engine="openpyxl",sheet_name="Basket Level Errors")
    df = df[df["Cluster ID"] == "CS00"+cluster]

    fig = px.scatter(x=df["Dates"].astype(str), y=df["Error type"],size = df['Number of Occurance'],color = df['Scanner'],
                 color_discrete_sequence=["magenta", "green", "blue", "purple"],
                 size_max=16,text=df['Number of Occurance']).update_traces(textposition="top center")
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(
        title="Error Trend for cluster CS00"+cluster,
        xaxis_title="Date of Occurence",
        yaxis_title="Error Type",
        legend_title="Scanner",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"))
    fig.show()

def generate_html(path):

    df = pd.read_excel(path,engine="openpyxl",sheet_name="Basket Level Errors")
    # df = df[df["Cluster ID"] == "CS00"+cluster]

    fig = px.scatter(x=df["Dates"].astype(str), y=df["Error type"],size = df['Number of Occurance'],color = df['Scanner'],
                 color_discrete_sequence=["magenta", "green", "blue", "purple"],
                 size_max=16,text=df['Number of Occurance']).update_traces(textposition="top center")
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(
        title="Error Trend for cluster CS00"+cluster,
        xaxis_title="Date of Occurence",
        yaxis_title="Error Type",
        legend_title="Scanner",
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"))
    fig.show()


if __name__ == "__main__":
    try:
        path = sys.argv[1]
        cluster = sys.argv[2]
    except:
        print("its ok")

    print(len(sys.argv))

    if len(sys.argv) == 3:
        generate_html_1(path,cluster)
    if len(sys.argv) == 2:
        generate_html(path)
