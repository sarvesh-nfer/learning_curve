#sarvesh
import pandas as pd
import numpy as np
import os,glob
import shutil 


input_path = "/wsi_app/service_logs/post_processing"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/home/adminspin/Music/scripts/newly"
if not os.path.exists(output_path):
	os.makedirs(output_path)

df = pd.read_csv("/home/adminspin/Music/scripts/white_logs.csv")


# txt = glob.glob(input_path+'/*/*.log')

for i in df['slide_name']:
    try:
        pp_path = os.path.join(input_path,i+".log")
        shutil.copy2(pp_path, output_path)
    except Exception as msg:
        print("error in slide : ",i,'\t msg :',msg)

