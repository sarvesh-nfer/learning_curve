import pandas as pd
import os
import shutil
df=pd.read_csv('/home/adminspin/Desktop/transfer-all.csv')

#src = df['no_onex']
# dst = df['transferx']

input_path = "/datadrive/wsi_data/onex_images"

for row in df['no_onex']:
    src = os.path.join(input_path,row,'onex_white_ref.png')
    try:
        if os.path.exists(src):
            os.remove(src)
            print("white ref Successfully removed for slide : ",row)
    except:
        print('Slide has no ref : ',row)

