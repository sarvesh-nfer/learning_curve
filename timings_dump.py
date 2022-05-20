import pymongo
import json
import pandas as pd
from pymongo import MongoClient
client = MongoClient()
db1 = client.viewerDB

collection1 = db1.slide_timings


datapoints1 = list(db1.slide_timings.find({}))
datapoints1

df = pd.json_normalize(datapoints1)

df.to_csv("/home/adminspin/Music/scripts/data.csv")
