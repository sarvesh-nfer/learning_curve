#sarvesh
import cv2
import numpy as np
import os,glob
import shutil 


input_path = "/home/adminspin/wsi_app/acquired_data"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/home/adminspin/Music/sarvesh/validation"
if not os.path.exists(output_path):
	os.mkdir(output_path)
images=['H01FBA08R_2469','H01FBA08R_2470','H01FBA08R_2467','H01FBA08R_2463','H01FBA08R_2465','H01FBA08R_2472','H01FBA08R_2466','H01FBA08R_2471','H01FBA08R_2473','H01FBA08R_2468']

for slide_name in images :

	try :
		print("name = ", slide_name)
		slide_path_input = os.path.join(input_path, slide_name)
		slide_path = os.path.join(output_path, slide_name)

		src = slide_path_input + "/"+"loc_output_data"
		src2 = glob.glob(slide_path_input  +"/*.log")[0]
		src3 = slide_path_input + "/" + slide_name +".db"
		src4 = slide_path_input + "/" + slide_name +".jpeg"
		if not os.path.exists(slide_path):
			os.mkdir(slide_path)
		dst = slide_path + "/"+"loc_output_data"
		dst2 = slide_path +"/"
		shutil.copytree(src, dst)
		shutil.copy2(src2, dst2)
		shutil.copy2(src3, dst2)
		shutil.copy2(src4, dst2)

	except Exception as msg:
		print(" Error for Slide : ", msg,"\t", slide_name)
