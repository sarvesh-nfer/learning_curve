import requests
import json
import pandas as pd
query = {
    "size":10000,
  "query": {
    "match_all": {}
  }
}
password = "=YZ=203Dkj5HG3HZ+fXh"
host_name = '10.104.16.22'
# index_name = "slide_picking_from_scanner"



index_list = ["basket_data","inline_corrections","slide_placement","slide_locking","slide_picking_from_scanner"]

for index_name in index_list:
  url = "https://elastic:{}@{}:9200/{}/_search".format(password, host_name, index_name)
  response4 = requests.get(url,json = query, verify = '/etc/elasticsearch/certs/http_ca.crt')
  response_json4 = json.loads(response4.text)
  slide_picking_from_scanner = pd.json_normalize(response_json4['hits']['hits'])
  print("slide_picking_from_scanner data acquired successfully")
  slide_picking_from_scanner = slide_picking_from_scanner.sort_values(by = ['_source.data.row_index'], ascending = True)
  slide_picking_from_scanner = slide_picking_from_scanner.sort_values(by = ['_source.data.col_index'], ascending = True)
  slide_picking_from_scanner['row_col'] = (slide_picking_from_scanner['_source.data.row_index']+1).astype(str)+"_"+(slide_picking_from_scanner['_source.data.col_index']+1).astype(str)
  slide_picking_from_scanner['_source.data.date'] = pd.to_datetime(slide_picking_from_scanner['_source.data.time_stamp']).dt.date
  slide_picking_from_scanner.to_csv('/home/adminspin/Music/scripts/'+index_name+'.csv')

