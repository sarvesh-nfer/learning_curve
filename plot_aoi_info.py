import sqlite3
import os, glob
import sys
import cv2
import pandas as pd
from os.path import exists, join

def get_aoi_info(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    query = "SELECT aoi_x, aoi_y, aoi_name, status, is_sampled FROM focus_sampling_info JOIN aoi USING (aoi_name)"
    # print(query)
    cursor.execute(query)

    aoi_data = cursor.fetchall()

    aoi_info_list = []

    for i in range(0, len(aoi_data)):
        x_pos = aoi_data[i][0]
        y_pos = aoi_data[i][1]
        aoi_name = aoi_data[i][2]
        status = aoi_data[i][3]
        sample = aoi_data[i][4]
        print(aoi_name)

        aoi_info_list.append([x_pos, y_pos, aoi_name, status, sample])


    query = "SELECT aoi_x, aoi_y, aoi_name, point_type FROM focus_sampling_info JOIN aoi USING (aoi_name) WHERE point_type = 0"
    # print(query)
    cursor.execute(query)

    z_data = cursor.fetchall()

    bestz_list = []

    for i in range(0, len(z_data)):
        x_pos = z_data[i][0]
        y_pos = z_data[i][1]
        aoi_name = z_data[i][2]

        print(aoi_name)

        bestz_list.append([x_pos, y_pos, aoi_name])

    return aoi_info_list , bestz_list
    
def map_aois_to_1x(aoi_info_list, bestz_list, onex_img):
    fov_width = 12.486578525641
    fov_height = 7.8297385620915
    
    for j in range(0,len(aoi_info_list)):
        print(j)
        x_pos = int(aoi_info_list[j][0])
        y_pos = int(aoi_info_list[j][1])
        status = aoi_info_list[j][3]
        sample = aoi_info_list[j][4]
        x2 = int(x_pos + fov_width)
        y2 = int(y_pos + fov_height)
        # if status == 0 and sample == 0:
        # # cv2.rectangle(onex_img, (x_pos, y_pos),(x2, y2),(0,0,255),4)
        #     cv2.circle(onex_img,(x_pos,y_pos),1,(128,0,128),10)
        if status == 0 and sample == 1:
            cv2.circle(onex_img,(x_pos,y_pos),1,(0,255,255),10)
        if status == 0 and sample == 0:
            cv2.circle(onex_img,(x_pos,y_pos),1,(0, 0, 255),10)
        
    for k in range(0,len(bestz_list)):

        x_pos = int(bestz_list[k][0])
        y_pos = int(bestz_list[k][1])

        cv2.circle(onex_img,(x_pos,y_pos),1,(0,255,0),10)
        continue
    return onex_img


if __name__ == '__main__':
    slide_path = sys.argv[1]
    
    txt = glob.glob(slide_path+'/*')
    for slide in txt:
        try:
            # slide_name = slide.split("/")[-1]
            # slide_name = 'H01JBA21P-9317'
            db_path = glob.glob(slide_path + "/*.db")[0]
            # db_path = join(slide, slide_name + ".db")

            print("DB path: ", db_path)

            # Connect to db and get aoi data.
            aoi_info_list,bestz_list = get_aoi_info(db_path)

            # Map the aois to 1x image.
            onex_img_path = join(slide_path, "loc_output_data", "updatedInputImage.png")
        
            onex_img = cv2.imread(onex_img_path)

            mapped_img = map_aois_to_1x(aoi_info_list, bestz_list, onex_img)

            cv2.imwrite(slide_path + '/mapped_img.png', mapped_img)
            # break
        except Exception as msg:
            print("error due to",msg)