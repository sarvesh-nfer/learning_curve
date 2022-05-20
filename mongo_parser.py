import pandas as pd
import os
from pymongo import MongoClient
# df=pd.read_csv("/home/adminspin/Desktop/df2.csv")
client = MongoClient()
db1 = client.scanner

collection1 = db1.slide_placement_info


datapoints1 = list(db1.slide_placement_info.find({}))
datapoints1
if not os.path.exists("/home/adminspin/Music/scripts"):
    os.makedirs("/home/adminspin/Music/scripts")
df = pd.json_normalize(datapoints1)
df.to_csv("/home/adminspin/Music/scripts/sp_info.csv",index=False)
print("file successfully saved in path /home/adminspin/Music/scripts \t len of df : ",len(df))
