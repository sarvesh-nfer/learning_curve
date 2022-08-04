import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statistics
import numpy as np
import warnings
warnings.filterwarnings("ignore")
query = {"query": {"match_all": {}},"size": 10000,"sort": [{"data.time_stamp": {"order": "desc"}}]}

password = "bqIUDWf+4Q*kX2BUh6RX"
host_name = '10.10.6.148'

index_name = "basket_data"
url = "https://elastic:{}@{}:9200/{}/_search".format(password, host_name, index_name)
response5 = requests.get(url,json = query, verify = '/etc/elasticsearch/certs/http_ca.crt')
response_json5 = json.loads(response5.text)
basket_data = pd.json_normalize(response_json5['hits']['hits'])
print("basket_data data acquired successfully")
