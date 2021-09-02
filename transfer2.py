#sarvesh
import cv2
import numpy as np
import os
import shutil 
import glob

input_path = "/datadrive/wsi_data/compressed_data"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/datadrive/wsi_data/compressed_data/output2/"
if not os.path.exists(output_path):
	os.mkdir(output_path)
images=os.listdir(input_path)

for slide_name in images :

	print("name = ", slide_name)
	slide_path_input = os.path.join(input_path, slide_name)
	slide_path = os.path.join(output_path, slide_name)

	src2 = glob.glob(os.path.join(slide_path_input,"grid_1","debug_data"))
	for src in src2:
	    if (os.path.exists(src)) :
		    if os.path.exists(slide_path):
			    continue
		    os.mkdir(slide_path)
		    dst = slide_path + "/"+"grid_1"+"/"+"debug_data"
		    shutil.copytree(src, dst)
	    else:
	        print("No debug folder", slide_name)
