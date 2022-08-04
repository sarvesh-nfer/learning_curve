import pandas as pd
import numpy as np
import cv2
import sqlite3
import json


conn = sqlite3.connect("/home/adminspin/Desktop/cameratemp/H01EBB54P-8406/H01EBB54P-8406.db")

df = pd.read_sql_query("select * from aoi",conn)

mask = np.zeros(((max(df['aoi_row_idx'])+1),(max(df['aoi_col_idx'])+1)),np.uint8)

graph_json_obj = json.load(open("/home/adminspin/Desktop/cameratemp/H01EBB54P-8406/input_for_graph_algorithm.json"))

for i,j,k in zip(df['aoi_name'],df['aoi_row_idx'],df['aoi_col_idx']):
    print(i,",",j,",",k)