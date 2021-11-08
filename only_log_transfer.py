#sarvesh
import cv2
import numpy as np
import os,glob
import shutil 


input_path = "/datadrive/wsi_data/Analysis_Acq_pipeline/System_Testing"
#input_path = '/datadrive/wsi_data/compressed_data/'
output_path = "/home/adminspin/Music/sarvesh/newly"
if not os.path.exists(output_path):
	os.makedirs(output_path)

txt = glob.glob(input_path+'/*/*.log')

for i in txt:
    try:
        a = i.split('/')[-2]
        slide_path = os.path.join(output_path, a)
        if not os.path.exists(slide_path):
            os.mkdir(slide_path)
        shutil.copy2(i, slide_path)
    except Exception as msg:
        print("error in slide : ",a,'\t msg :',msg)

