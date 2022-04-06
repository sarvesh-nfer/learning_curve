import pandas as pd
import os,glob

txt = glob.glob("/datadrive/wsi_data/scan_data/20dec_04R/*/metadata.json")

data = {}
for i in txt:
    df = pd.read_json(i)
    scan_area = 0
    for j in range(len(df)):
        try:
            scan_area = scan_area + df['data']['grid_info'][j]['area']
        except Exception as msg:
            print("no area")
    print(scan_area)
    data[i.split('/')[-2]] = scan_area

df = pd.DataFrame(data.items(), columns=['slide_name', 'scan_area'])
folder = txt[0].split('/')[-3]
df.to_csv("/home/adminspin/Desktop/"+str(folder)+"_scanArea.csv",index=False)