#sarvesh
import cv2
import numpy as np
import os,glob
import shutil 


txt = glob.glob("/home/adminspin/wsi_app/acquired_data/*/grid_*/BI_bg_7")
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/home/adminspin/Music/sarvesh/validation"
if not os.path.exists(output_path):
	os.mkdir(output_path)
#images=os.listdir(input_path)

for i in txt :

	try :
		a = "BI_bg_7"
		b = i.split('/')[-2]
		c = i.split('/')[-3]
		dst = os.path.join(output_path,c,b,a)
		if os.path.exists(dst):
			os.makedirs(dst)
		shutil.copytree(i, dst)
			

	except Exception as msg:
		print(" Error for Slide : ", msg)
