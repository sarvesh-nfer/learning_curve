#sarvesh
import cv2
import numpy as np
import os,glob
import shutil 

txt = glob.glob("/wsi_app/acquired_data/*/*/stack_fusion.json")

output_path = "/home/adminspin/Music/sarvesh/newly"
if not os.path.exists(output_path):
	os.mkdir(output_path)

for i in txt:
	try:
		s = i.split('/')[-3]
		g = i.split('/')[-2]
		dst = os.path.join(output_path,s,g)
		if not os.path.exists(dst):
			os.makedirs(dst)
		shutil.copy2(i,dst)
	except Exception as msg:
		print("ERROR:\n",msg)