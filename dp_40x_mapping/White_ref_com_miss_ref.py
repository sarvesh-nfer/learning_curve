# take out onex time for white ref comparision for missing dates

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import cv2
import sys
import os
import glob

input_p = '/datadrive/wsi_data/onex_images'
df = pd.read_excel('/home/adminspin/Downloads/gt.xlsx','non')
input_list = df['Slide Name']
a_dict = {}
b_dict = {}
a = 'No'
for i in input_list:
    try:
        input_path = os.path.join(input_p,i)
        #print(input_path)
        white_p = os.path.join(input_path,'onex_white_ref.png')
        log_p = glob.glob(os.path.join(input_path,'*.log'))[0]
        with open(log_p) as file:
            for line in file:
                if 'Acquisition started for:' in line:
                    bgtime = str(line.split(':')[-1].strip())
                if 'Extracted time stamp in Local Timezone ISO format' in line:
                    bgtime2 = str(line.split(' ')[-2].strip())
            a_dict[bgtime] = bgtime2
        if not os.path.exists(white_p):
            b_dict[i] = a

    except:
        ("No ref image for slide : ",i)       
dict_df = pd.DataFrame(a_dict.items(),columns=['slide_name','date'])
dict_df2 = pd.DataFrame(b_dict.items(),columns=['slide_name','onex'])
dict_df.to_csv('/home/adminspin/Music/UnitTests/40x_1x_Mapping/adict_non.csv',index=False)
dict_df2.to_csv('/home/adminspin/Music/UnitTests/40x_1x_Mapping/bdict_non.csv',index=False)
