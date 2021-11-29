import cv2
import pandas as pd
import numpy as np

import os,glob
from PIL import Image

def select_crop(filename,x,y):
    img = Image.open(filename)
    width, height = img.size
    
    file_extension = os.path.splitext(filename)[-1]
    save_path = os.path.split(filename)[0]
    
    start_pos = start_x, start_y = (0, 0)
    cropped_image_size = w, h = (x, y)

    count = 1
    count2 = 1
    stack = os.path.split(filename)[-1].split('.')[0]
    for col_i in range(0, width, w):
        for row_i in range(0, height, h):
            crop = img.crop((col_i, row_i, col_i + w, row_i + h))
            if not os.path.exists(save_path+"/"+"blob_" + str(count2)):
                os.makedirs(save_path+"/"+"blob_" + str(count2))
            if count == 1 or count == 2 or count == 4 or count == 5 or count == 7 or count == 8:
                crop.save(save_path+"/"+"blob_" + str(count2)+"/stack_"+stack+"_blob_" + str(count2) + file_extension)
                print(save_path+"/"+"blob_" + str(count2)+"/stack_"+stack+"_blob_" + str(count2) + file_extension)
                count2 += 1
            count += 1

def laplacian_mean(path):

    txt = glob.glob("/home/adminspin/Documents/stack_images/stack_images/*/*/*.bmp")

    df = pd.DataFrame(columns = ['aoi_name','blob_info','laplacian_mean'])
    count = 0
    for i in txt:
        try:
            img = cv2.imread(i)
            green_channel = img[:,:,1]
            laplacian_img  = abs(cv2.Laplacian(green_channel, cv2.CV_32FC1))
            laplacian_mean  = np.mean(laplacian_img)

            sar[i.split('/')[-3]] = laplacian_mean
            sar2[i.split('/')[-3]] = i.split('/')[-1].split('.')[0]
            df.loc[count] = (i.split('/')[-3],i.split('/')[-1].split('.')[0],laplacian_mean)
            print("saved for : ",i)
            count +=1
        except Exception as msg:
            print("error in slide : ",i,msg)
    df[['stack', 'blob']] = df['blob_info'].str.split('_', 1, expand=True)

    ## calculate best blob throughout the stack
    for i in df['aoi_name'].unique():
        for j in df['blob'].unique():
            print(i,"_---_",j)
            df2 = df[(df['aoi_name'] == i) & (df['blob'] == j)]
            print(df[(df['aoi_name'] == i) & (df['blob'] == j)].max()[1])
            best = df[(df['aoi_name'] == i) & (df['blob'] == j)].max()[1]
            df2['best_idx'] = best
            df3 = df3.append(df2)
            

txt = glob.glob("/home/adminspin/Documents/stack_images/stack_images/*/*.bmp")
sar = 0
for i in txt:
    select_crop(i,512,512)