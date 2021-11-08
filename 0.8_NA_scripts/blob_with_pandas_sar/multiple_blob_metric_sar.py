import numpy as np
import cv2
import os,glob
import matplotlib.pyplot as plt
import sqlite3
import csv
import pandas as pd
import shutil 

def populateFocusColorMetricValuesFromDB(slide_path):
        uintMax = 255
        color_list = [7,16,24,30,42,80,105,120,134,149,164,175]

        blobXCoord = [0,0,0,0,0,138,138,138,138,138,276,276,276,276,276,414,414,414,414,414,552,552,552,552,552,690,690,690,690,690,828,828,828,828,828]
        blobYCoord = [0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484]

        blob_height = 121
        blob_width = 138
        grid_path = glob.glob(slide_path+"/grid_*")
        # path1 = os.path.join(grid_path, "raw_images")

        db_file_path = glob.glob(slide_path+'/*.db')[0]
        conn = sqlite3.connect(db_file_path)
        df1 = pd.read_sql_query("select * from acquisition_blob_info ;", conn)
        df2 = pd.read_sql_query("select * from aoi ;",conn)

        df2 = df2[['aoi_name','grid_id','best_idx']]

        df3 = pd.merge(df2,df1,on=['aoi_name','grid_id'])

        df = df3[df3['stack_index'] == df3['best_idx']]
        stack_size = 5


        for sar in grid_path:

            out_path = os.path.join(sar, "20x_fm_cm_hm")
            if not os.path.exists(out_path) : os.mkdir(out_path)

            _grid_id = sar.split('/')[-1].split('_')[-1]

            
            path1 = os.path.join(sar,"raw_images")

            list_dir = os.listdir(path1)
            list_dir = sorted(list_dir)

            for item in list_dir:
                # item = 'aoi0252.bmp'
                # print(item)
                # if item == "aoi0150.bmp" : break
                aoi_name = item.split(".")[0]
                input_image = cv2.imread(os.path.join(path1, item))

                print("**"*50)
                print("image path : ",os.path.join(path1, item))
                print("**"*50)
                resized = cv2.resize(input_image, (1936,1216), cv2.INTER_NEAREST)
                

                c = conn.cursor()

                c.execute("select best_idx from aoi where aoi_name == '"+aoi_name+"' and grid_id =="+str(_grid_id))
                index = c.fetchone()[0]
                # print("index: ",index)
                if index == -1 :
                    cv2.putText(resized, "{}".format("Background - No Stack Captured"), (125,125),\
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,255), 2)
                    cv2.imwrite(os.path.join(out_path, aoi_name+".jpeg"), resized)
                    continue
                
                blob_index_list = []
                fm_list_data = []
                cm_list_data = []
                stack_index_list = []
                hue_metric_list = []
                
                blob_index_list = df[(df['grid_id'] == int(_grid_id)) & (df['aoi_name'] == aoi_name)]['blob_index'].to_list()
                fm_list_data = df[(df['grid_id'] == int(_grid_id)) & (df['aoi_name'] == aoi_name)]['focus_metric'].to_list()
                cm_list_data = df[(df['grid_id'] == int(_grid_id)) & (df['aoi_name'] == aoi_name)]['color_metric'].to_list()
                stack_index_list = df[(df['grid_id'] == int(_grid_id)) & (df['aoi_name'] == aoi_name)]['stack_index'].to_list()
                hue_metric_list = df[(df['grid_id'] == int(_grid_id)) & (df['aoi_name'] == aoi_name)]['hue_metric'].to_list()
                    # print(_vals) 
                # fm_list_data = c.fetchall()
                print("blob_index_list: ", blob_index_list)
                print("fm_list_data: ",fm_list_data)
                print("cm_list_data: ",cm_list_data)
                # print("One val", fm_list_data[2][0])

                valid_blobs = 0        
                print(len(blob_index_list))
                # start_id = 0
                # if int(len(fm_list_data)/(stack_size*35)) > 0:
                #     start_id = (int(len(fm_list_data)/(stack_size*35)) -1) * (stack_size*35)
                # # exit()
                # counter = 0 
                print(aoi_name,"------",len(blob_index_list),"-->","start_id")
                for blob_id in range(len(blob_index_list)):
                    # _id_start = start_id + blob_id * stack_size

                    # sub_list_fm = fm_list_data[_id_start:_id_start+stack_size]
                    # sub_list_cm = cm_list_data[_id_start:_id_start+stack_size]
                    # sub_list_hm = hue_metric_list[_id_start:_id_start+stack_size]
                    # print(sub_list_fm)
                    print("BLOB INDEX : ",blob_index_list[blob_id])
                    print("FM LIST : ",fm_list_data[blob_id])
                    blob_best_id = blob_index_list[blob_id]
                    fm_val = round(float(fm_list_data[blob_id]),2)
                    cm_val = round(float(cm_list_data[blob_id]),2)
                    hm_val = round(float(hue_metric_list[blob_id]),2)
                    # p += stack_size
                    # print(sub_list_fm)
                    # print("max_id ", blob_best_id)
                    # exit()    
                    # print(blob_id)
                    # if cm_val < 7:
                    #     cv2.rectangle(resized, ((2*self.blobXCoord[blob_id])+2, (2*self.blobYCoord[blob_id])+2),(2*self.blobXCoord[blob_id]+((2*self.blob_width)-2), \
                    #         2*self.blobYCoord[blob_id]+((2*self.blob_height)-2)), (0,0,255), 2)
                    # else:
                    cv2.rectangle(resized, ((2*blobXCoord[blob_id])+2, (2*blobYCoord[blob_id])+2),(2*blobXCoord[blob_id]+((2*blob_width)-2), \
                        2*blobYCoord[blob_id]+((2*blob_height)-2)), (0,0,0), 1)


                    cv2.putText(resized, "{}".format("fm:"+str(fm_val)), (int(2*(blobXCoord[blob_id])+blob_width), int(2*(blobYCoord[blob_id])+blob_height)),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                    cv2.putText(resized, "{}".format("cm:"+str(cm_val)), (int(2*(blobXCoord[blob_id])+blob_width), int(2*(blobYCoord[blob_id])+blob_height)+25),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
                    cv2.putText(resized, "{}".format("hm:"+str(hm_val)), (int(2*(blobXCoord[blob_id])+blob_width), int(2*(blobYCoord[blob_id])+blob_height)+50),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
                    cv2.putText(resized, "{}".format(str(blob_id)), (int(2*(blobXCoord[blob_id])+blob_width), int(2*(blobYCoord[blob_id])+blob_height)+75),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
                    cv2.putText(resized, "{}".format(str(index)), (int(2*(blobXCoord[blob_id])+blob_width), int(2*(blobYCoord[blob_id])+blob_height)+100),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
                cv2.imwrite(os.path.join(out_path, aoi_name+".jpeg"), resized)
                # exit()

path = "/home/adminspin/wsi_app/acquired_data"
slide_name = ['H01FBA08R_2469', 'H01FBA08R_2470', 'H01FBA08R_2467', 'H01FBA08R_2463', 'H01FBA08R_2465', 'H01FBA08R_2472', 'H01FBA08R_2466', 'H01FBA08R_2471', 'H01FBA08R_2473', 'H01FBA08R_2468']

for k in slide_name:
    slide_path = path + '/' + k
    populateFocusColorMetricValuesFromDB(slide_path)
# slide_path = path + '/' + "H01FBA08R_2425"
# populateFocusColorMetricValuesFromDB(slide_path)




