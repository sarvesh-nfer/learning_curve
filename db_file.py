#sarvesh
import os
import pandas as pd
import numpy as np
import sqlite3

                                                                                                                                                                                                       
input_folder_path = '/datadrive/wsi_data/compressed_data/'
input_list = os.listdir(input_folder_path)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
for item in input_list:
    db_path = os.path.join(input_folder_path, item)
    print(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor() 
    cur.execute("select * from validation_info limit 5;")
    results = cur.fetchall()
    item = pd.read_sql_query("select * from validation_info ;", conn)
    print(item)
