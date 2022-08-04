import pandas as pd
import shutil,os

path = "/ssd_drive/restored_data"

output = "/home/adminspin/Music/scripts/basetils"

df = pd.read_csv("/home/adminspin/Music/scripts/basetils.csv")

for i in df['slide_name']:
    try:
        output_path = output + "/" + i
        input_path = "/ssd_drive/restored_data/"+i+"/grid_merged/base_tiles/000_files/9"

        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        
        shutil.copytree(input_path, output_path)
    except Exception as msg:
        print(i,msg)
