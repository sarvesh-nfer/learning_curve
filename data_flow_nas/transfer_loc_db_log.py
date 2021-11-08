#sarvesh
import cv2
import numpy as np
import os
import shutil 


input_path = "/home/adminspin/wsi_app/acquired_data"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/home/adminspin/Music/sarvesh/validation"
if not os.path.exists(output_path):
	os.mkdir(output_path)
images=['H01CBA04R_192','PPP5230','H01CBA04R_195','H01CBA04R_194','PPP5276','SPTEST09','PPP5236','PPP5264',
'H01CBA04R_212','H01CBA04R_213','H01CBA04R_214','H01CBA04R_215']
for slide_name in images :

	try :
		print("name = ", slide_name)
		slide_path_input = os.path.join(input_path, slide_name)
		slide_path = os.path.join(output_path, slide_name)

		src = slide_path_input + "/"+"loc_output_data"
		src2 = slide_path_input + "/" + slide_name +".log"
		src3 = slide_path_input + "/" + slide_name +".db"
		if not os.path.exists(slide_path):
			os.mkdir(slide_path)
		dst = slide_path + "/"+"loc_output_data"
		dst2 = slide_path +"/"
		#shutil.copytree(src, dst)
		#shutil.copy2(src2, dst2)
		shutil.copy2(src3, dst2)

	except Exception as msg:
		print(" Error for Slide : ", msg,"\t", slide_name)
