#copy import
#sarvesh

import os
import shutil 
import glob
import pandas as pd

input_path = "/datadrive/wsi_data/onex_images"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/home/adminspin/Music/UnitTests/40x_1x_Mapping/non-HE/"
if not os.path.exists(output_path):
	os.mkdir(output_path)
df = pd.read_excel('/home/adminspin/Downloads/gt.xlsx','non')

images= df['Slide Name']

for slide_name in images :
    print("name = ", slide_name)
    slide_path_input = os.path.join(input_path, str(slide_name))
    slide_path = os.path.join(output_path, str(slide_name))
    if not os.path.exists(output_path):
        os.mkdir(slide_path)
    dst = slide_path
    shutil.copytree(slide_path_input, dst)
