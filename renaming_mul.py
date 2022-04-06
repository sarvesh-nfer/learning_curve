# import glob, os
# # print(glob.glob("/datadrive/wsi_data/data_for_validation/*/*/class/*/*.bmp"))
# for filename in glob.glob("/datadrive/wsi_data/data_for_validation/*/*/class/*/*.jpg"):
#     print(filename)
#     os.rename(filename, filename[:-4] + '.bmp')

import os,glob
import cv2
# from PIL import Image
# input_path = '/home/adminspin/Desktop/slides_data/H01CBA04R_8021/grid_1/raw_images'
# files = os.listdir(input_path)
# print(glob.glob("/datadrive/wsi_data/data_for_validation/*/*/class/*/*.bmp"))
count = 1
txt = glob.glob("/datadrive/wsi_data/data_for_validation/*/*/class/*/*.bmp")
for i in txt:
    img = cv2.imread(i) 

    print(count ,"OUT OF ",len(txt))
    count = count + 1
    cv2.imwrite(os.path.split(i)[0] +'/'+str(os.path.split(i)[-1].split('.')[0]) + '.jpeg',img)
    