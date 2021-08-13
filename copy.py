#sarvesh
import cv2
import numpy as np
import os
import shutil 


input_path = "/datadrive/wsi_data/compressed_data"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/datadrive/wsi_data/output/"
if not os.path.exists(output_path):
	os.mkdir(output_path)
images=os.listdir(input_path)

for slide_name in images :

	try :
		print("name = ", slide_name)
		slide_path_input = os.path.join(input_path, slide_name)
		slide_path = os.path.join(output_path, slide_name)

		src = slide_path_input + "/"+"loc_output_data"
		src2 = slide_path_input + "/" + slide_name +".log"
		if (os.path.exists(src)) :
			if os.path.exists(slide_path):
				continue
			os.mkdir(slide_path)
			dst = slide_path + "/"+"loc_output_data"
			dst2 = slide_path +"/"
			shutil.copytree(src, dst)
			shutil.copy2(src2, dst2)
		else:
		    print("No loc_out folder", slide_name)

	except Exception as msg:
		print(" Error for Slide : ", msg,"\t", slide_name)
