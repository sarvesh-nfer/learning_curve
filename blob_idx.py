import os
import pandas as pd
import numpy as np
import sqlite3
import cv2

class blob:
    def populateFocusColorMetricValuesFromDB(raw_path,db_path,grid_no):
        int_a = 1
        uintMax = 255
        color_list = [7,16,24,30,42,80,105,120,134,149,164,175]

        blobXCoord = [0,0,0,0,0,138,138,138,138,138,276,276,276,276,276,414,414,414,414,414,552,552,552,552,552,690,690,690,690,690,828,828,828,828,828]
        blobYCoord = [0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484]

        height = 121
        width = 138
        
        grid_path = "/home/adminspin/Desktop/bubble/04r/sarvesh/H01CBA04R_8420/grid_3"
        path1 = os.path.join(grid_path, "raw_images")

        db_file_path = "/home/adminspin/Desktop/bubble/04r/sarvesh/H01CBA04R_8420/H01CBA04R_8420.db"
        _grid_id = grid_no
        stack_size = 5

        dbConn = sqlite3.connect(db_file_path)

        out_path = os.path.join(grid_path, "20x_fm_cm_hm")
        if not os.path.exists(out_path) : os.mkdir(out_path)

        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)

        for item in list_dir:
            # item = 'aoi0133.bmp'
            print(item)
            # if item == "aoi0150.bmp" : break
            aoi_name = item.split(".")[0]
            input_image = cv2.imread(os.path.join(path1, item))

            resized = cv2.resize(input_image, (1936,1216), cv2.INTER_NEAREST)

            c = dbConn.cursor()

            c.execute("select best_idx from aoi where aoi_name == '"+aoi_name+"' and grid_id =="+str(_grid_id))
            index = c.fetchone()[0]
            # print("index: ",index)
            if index == -1 :
                cv2.putText(resized, "{}".format("Background - No Stack Captured"), (125,125),\
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,255), 2)
                cv2.imwrite(os.path.join(out_path, aoi_name+".jpeg"), resized)
                continue

            c.execute("SELECT blob_index, focus_metric, color_metric, stack_index, hue_metric from acquisition_blob_info WHERE aoi_name = '"+aoi_name+"' and grid_id =="+str(_grid_id))
            # for item in c.fetchone
            blob_index_list = []
            fm_list_data = []
            cm_list_data = []
            stack_index_list = []
            hue_metric_list = []
            all_data = c.fetchall()
            for _vals in all_data:
                blob_index_list.append(_vals[0])
                fm_list_data.append(_vals[1])
                cm_list_data.append(_vals[2])
                stack_index_list.append(_vals[3])
                hue_metric_list.append(_vals[4])
                # print(_vals) 
            # fm_list_data = c.fetchall()
            print("blob_index_list: ", blob_index_list)
            print("fm_list_data: ",fm_list_data)
            print("cm_list_data: ",cm_list_data)
            # print("One val", fm_list_data[2][0])

            valid_blobs = 0        
            print(len(blob_index_list))
            start_id = 0
            if int(len(blob_index_list)/(stack_size*35)) > 0:
                start_id = (int(len(blob_index_list)/(stack_size*35)) -1) * (stack_size*35)
            # exit()
            counter = 0 
            for blob_id in range(35):
                _id_start = start_id + blob_id * stack_size
                # blob_id = blob_index_list[p]
                # fm_val = round(fm_list_data[p],2)
                # cm_val = round(cm_list_data[p],2)
                # hm_val = hue_metric_list[p]
                # stack_id = stack_index_list[_id_start]

                sub_list_fm = fm_list_data[_id_start:_id_start+stack_size]
                sub_list_cm = cm_list_data[_id_start:_id_start+stack_size]
                sub_list_hm = hue_metric_list[_id_start:_id_start+stack_size]

                blob_best_id = np.argmax(sub_list_fm)
                fm_val = round(sub_list_fm[blob_best_id],2)
                cm_val = round(sub_list_cm[blob_best_id],2)
                hm_val = round(sub_list_hm[blob_best_id],2)
                # p += stack_size
                print(sub_list_fm)
                # print("max_id ", blob_best_id)
                # exit()    
                print(blob_id)
                # if cm_val < 7:
                #     cv2.rectangle(resized, ((2*self.blobXCoord[blob_id])+2, (2*self.blobYCoord[blob_id])+2),(2*self.blobXCoord[blob_id]+((2*self.width)-2), \
                #         2*self.blobYCoord[blob_id]+((2*self.height)-2)), (0,0,255), 2)
                # else:
                cv2.rectangle(resized, ((2*blobXCoord[blob_id])+2, (2*blobYCoord[blob_id])+2),(2*blobXCoord[blob_id]+((2*width)-2), \
                    2*blobYCoord[blob_id]+((2*height)-2)), (0,0,0), 1)


                cv2.putText(resized, "{}".format("fm:"+str(fm_val)), (int(2*(blobXCoord[blob_id])+width), int(2*(blobYCoord[blob_id])+height)),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                cv2.putText(resized, "{}".format("cm:"+str(cm_val)), (int(2*(blobXCoord[blob_id])+width), int(2*(blobYCoord[blob_id])+height)+25),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
                cv2.putText(resized, "{}".format("hm:"+str(hm_val)), (int(2*(blobXCoord[blob_id])+width), int(2*(blobYCoord[blob_id])+height)+50),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            cv2.imwrite(os.path.join(out_path, aoi_name+"_"+str(index)+".jpeg"), resized)
            # exit()
raw_path ="/home/adminspin/Desktop/bubble/04r/sarvesh/H01CBA04R_8420/grid_3"
db_path = "/home/adminspin/Desktop/bubble/04r/sarvesh/H01CBA04R_8420/H01CBA04R_8420.db"
grid_no = 3

blob.populateFocusColorMetricValuesFromDB(raw_path,db_path,grid_no)
