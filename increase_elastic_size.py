import requests
import json
import pandas as pd

query = {"index" : {"max_result_window" : 10000000}}

password = "=YZ=203Dkj5HG3HZ+fXh"
host_name = '10.104.16.22'
# index_name = "basket_data"

index_list = ["basket_data","inline_corrections","slide_placement","slide_locking","slide_picking_from_scanner","rescanned_slide_info","post"]
# index_list = ["rescanned_slide_info"]

for index_name in index_list:
    url = "https://elastic:{}@{}:9200/{}/_settings".format(password, host_name, index_name)
    response4 = requests.put(url,json = query, verify = '/etc/elasticsearch/certs/http_ca.crt')

    print(index_name," : ",response4)
    # print(requests.put(url, verify = '/etc/elasticsearch/certs/http_ca.crt'))
