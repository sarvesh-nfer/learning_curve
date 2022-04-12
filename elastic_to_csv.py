a = int(input("Enter the number of days of data you need: "))
import elasticsearch
from elasticsearch import Elasticsearch
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statistics
import numpy as np
es= Elasticsearch([{'host': '10.104.16.22','port': 9200}])
es.ping()
#retrieving data from elastic search
res = es.search(index="slide_picking_from_scanner", doc_type="", body={"_source":{}}, size=1000000,)
# print("slide_picking_from_scanner ACQUIRED SUCCESSFULLY")
slide_picking_from_scanner = pd.json_normalize(res['hits']['hits'])
slide_picking_from_scanner = slide_picking_from_scanner.sort_values(by = ['_source.data.row_index'], ascending = True)
slide_picking_from_scanner = slide_picking_from_scanner.sort_values(by = ['_source.data.col_index'], ascending = True)
slide_picking_from_scanner['row_col'] = (slide_picking_from_scanner['_source.data.row_index']+1).astype(str)+"_"+(slide_picking_from_scanner['_source.data.col_index']+1).astype(str)
slide_picking_from_scanner['_source.data.date'] = pd.to_datetime(slide_picking_from_scanner['_source.data.time_stamp']).dt.date
res1 = es.search(index="slide_placement", doc_type="", body={"_source":{}}, size=1000000,)
# print("slide_placement ACQUIRED SUCCESSFULLY")
slide_placement = pd.json_normalize(res1['hits']['hits'])
slide_placement = slide_placement.sort_values(by = ['_source.data.row_index'], ascending = True)
slide_placement = slide_placement.sort_values(by = ['_source.data.col_index'], ascending = True)
slide_placement['row_col'] = (slide_placement['_source.data.row_index']+1).astype(str)+"_"+(slide_placement['_source.data.col_index']+1).astype(str)
slide_placement['_source.data.date'] = pd.to_datetime(slide_placement['_source.data.time_stamp']).dt.date
res2 = es.search(index="slide_locking", doc_type="", body={"_source":{}}, size=1000000,)
# print("slide_locking ACQUIRED SUCCESSFULLY")
slide_locking = pd.json_normalize(res2['hits']['hits'])
slide_locking = slide_locking.sort_values(by = ['_source.data.row_index'], ascending = True)
slide_locking = slide_locking.sort_values(by = ['_source.data.col_index'], ascending = True)
slide_locking['_source.data.date'] = pd.to_datetime(slide_locking['_source.data.time_stamp']).dt.date
slide_locking['row_col'] = (slide_locking['_source.data.row_index']+1).astype(str)+"_"+(slide_locking['_source.data.col_index']+1).astype(str)
res3 = es.search(index="inline_corrections", doc_type="", body={"_source":{}}, size=1000000,)
# print("inline_corrections ACQUIRED SUCCESSFULLY")
inline_corrections = pd.json_normalize(res3['hits']['hits'])
inline_corrections = inline_corrections.sort_values(by = ['_source.data.row_index'], ascending = True)
inline_corrections = inline_corrections.sort_values(by = ['_source.data.col_index'], ascending = True)
inline_corrections['_source.data.date'] = pd.to_datetime(inline_corrections['_source.data.time_stamp']).dt.date
inline_corrections['row_col'] = (inline_corrections['_source.data.row_index']+1).astype(str)+"_"+(inline_corrections['_source.data.col_index']+1).astype(str)
res4 = es.search(index="basket_data", doc_type="", body={"_source":{}}, size=1000000,)
# print("basket_data ACQUIRED SUCCESSFULLY")
basket_data = pd.json_normalize(res4['hits']['hits'])
basket_data = basket_data.dropna(subset=['_source.data.row_index','_source.data.col_index'])
basket_data['_source.data.row_index'] = basket_data['_source.data.row_index'].astype(int)
basket_data['_source.data.col_index'] = basket_data['_source.data.col_index'].astype(int)
basket_data['row_col'] = (basket_data['_source.data.row_index']+1).astype(str)+"_"+(basket_data['_source.data.col_index']+1).astype(str)
basket_data = basket_data.sort_values(["_source.data.row_index","_source.data.col_index"], ascending = (True, True))
#amount of data required
inline_corrections = inline_corrections.sort_values(by = ['_source.data.date'], ascending = True)
lst = inline_corrections['_source.data.load_identifier'].unique()
df = pd.DataFrame(lst)
lt = df.tail(a)[0].tolist()
#data collection
inline = (inline_corrections[inline_corrections["_source.data.load_identifier"].isin(lt)]).to_csv("/home/adminspin/Downloads/inline.csv")
placement = (slide_placement[slide_placement["_source.data.load_identifier"].isin(lt)]).to_csv("/home/adminspin/Downloads/placement.csv")
locking = (slide_locking[slide_locking["_source.data.load_identifier"].isin(lt)]).to_csv("/home/adminspin/Downloads/locking.csv")
pickup = (slide_picking_from_scanner[slide_picking_from_scanner["_source.data.load_identifier"].isin(lt)]).to_csv("/home/adminspin/Downloads/pickup.csv")
basket = (basket_data[basket_data["_source.data.load_identifier"].isin(lt)]).to_csv("/home/adminspin/Downloads/basket.csv")
print("All the data has been saved onto your local device")