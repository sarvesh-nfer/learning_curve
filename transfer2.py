#sarvesh
import cv2
import numpy as np
import os
import shutil 
import glob
import pandas as pd

df = pd.read_csv("")
input_path = "/datadrive/wsi_data/compressed_data"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/datadrive/wsi_data/sarvesh/"
if not os.path.exists(output_path):
	os.mkdir(output_path)
images= df['slide_name']

for slide_name in images :

	print("name = ", slide_name)
	slide_path_input = os.path.join(input_path, slide_name)
	slide_path = os.path.join(output_path, slide_name)
	src1 = os.path.join(slide_path_input,slide_name+'.db')
	src2 = os.path.join(slide_path_input,"metadata.json")
	src3 = os.path.join(slide_path_input,"loc_output_data/finalMergedBbox.jpeg")
	dst = slide_path
	shutil.copytree(src, dst)
	shutil.copytree(src, dst)
	shutil.copytree(src, dst)

for slide_name in images:

	fem = glob.glob(os.path.join(input_path, slide_name)+"/grid_*"+"/focus_errorMap.jpeg")
	log = glob.glob(os.path.join(input_path, slide_name)+"/grid_*"+"/log")
	json = glob.glob(os.path.join(input_path, slide_name)+"/grid_*"+"/out_of_focus_aois_coordinates.json")

	for i in fem:
		try:
			dst = i.replace(str('compressed_data'),str('sarvesh')
			shutil.copytree(i, dst)
		except Exception as msg:
			print("exception in : fem")
	for i in log:
		try:
			dst = i.replace(str('compressed_data'),str('sarvesh')
			shutil.copytree(i, dst)
		except Exception as msg:
			print("exception in : log")
	for i in json:
		try:
			dst = i.replace(str('compressed_data'),str('sarvesh')
			shutil.copytree(i, dst)
		except Exception as msg:
			print("exception in : json")
