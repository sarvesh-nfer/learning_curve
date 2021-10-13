import pandas as pd
import os,glob
import shutil 

txt = glob.glob('/datadrive/wsi_data/acquired_data/Acqusitions/Bubble_Analysis/*/metadata.json')

output_p = '/home/adminspin/Desktop/sarvesh'
for i in txt:
    output_path = os.path.join(output_p,i.split('/')[-2])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        shutil.copy2(i, output_path)
        print("Copy success : ",i.split('/')[-2])
    except Exception as msg:
        print("error in slide : ",i.split('/')[-2])
    