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

df = pd.json_normalize(datapoints1)

print("!!DF saved!!")
if not os.path.exists("/home/adminspin/Music/scripts/dftimings"):
    os.makedirs("/home/adminspin/Music/scripts/dftimings")
df.to_csv("/home/adminspin/Music/scripts/cs005_dftimings"+str(time.strftime("%d-%m-%Y"))+".csv",index=False)
