import os
import pandas as pd
import numpy as np
import sqlite3
import cv2

class blob:
    def populateFocusColorMetricValuesFromDB(raw_path,db_path):
        blobXCoord = [0,0,0,0,0,138,138,138,138,138,276,276,276,276,276,414,414,414,414,414,552,552,552,552,552,690,690,690,690,690,828,828,828,828,828]
        blobYCoord = [0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484]
        height = 121
        width = 138
        path1 = os.path.join(raw_path,"raw_images")
        print(path1)
        db_file_path = os.path.join(db_path)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        dbConn = sqlite3.connect(db_file_path)

        out_path = os.path.join(raw_path,"20x_fm_cm")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)
        _grid_id = 4

        for item in list_dir:
            # item = 'aoi0133.bmp'
            print(item)
            # if item == "aoi0150.bmp" : break
            aoi_name = item.split(".")[0]
            input_image = cv2.imread(os.path.join(path1, item))

            resized = cv2.resize(input_image, (1936, 1216), cv2.INTER_NEAREST)

            c = dbConn.cursor()

            c.execute(
                "select best_idx from aoi where aoi_name == '"
                + aoi_name
                + "' and grid_id =="
                + str(_grid_id)
            )
            index = c.fetchone()[0]
            # print("index: ",index)
            if index == -1:
                cv2.imwrite(os.path.join(out_path, aoi_name + ".jpeg"), resized)
                continue

            c.execute(
                "SELECT blob_index, focus_metric, color_metric, stack_index from acquisition_blob_info WHERE aoi_name = '"
                + aoi_name
                + "' and grid_id =="
                + str(_grid_id)
                + " and color_metric >= 0.1"
            )
            # for item in c.fetchone
            blob_index_list = []
            fm_list_data = []
            cm_list_data = []
            stack_index_list = []
            all_data = c.fetchall()
            for _vals in all_data:
                blob_index_list.append(_vals[0])
                fm_list_data.append(_vals[1])
                cm_list_data.append(_vals[2])
                stack_index_list.append(_vals[3])
                # print(_vals)
            # fm_list_data = c.fetchall()
            # print("blob_index_list: ", blob_index_list)
            # print("fm_list_data: ", fm_list_data)
            # print("cm_list_data: ", cm_list_data)
            # print("One val", fm_list_data[2][0])
            print("len(fm_list_data)\t",len(fm_list_data))

            valid_blobs = 0
            for p in range(0, len(blob_index_list)):
                blob_id = blob_index_list[p]

                fm_val = round(fm_list_data[p], 2)
                cm_val = round(cm_list_data[p], 2)
                stack_id = stack_index_list[p]

                if cm_val < 7:
                    cv2.rectangle(
                        resized,
                        (
                            (2 * blobXCoord[blob_id]) + 2,
                            (2 * blobYCoord[blob_id]) + 2,
                        ),
                        (
                            2 * blobXCoord[blob_id] + ((2 * width) - 2),
                            2 * blobYCoord[blob_id] + ((2 * height) - 2),
                        ),
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.rectangle(
                        resized,
                        (
                            (2 * blobXCoord[blob_id]) + 2,
                            (2 * blobYCoord[blob_id]) + 2,
                        ),
                        (
                            2 * blobXCoord[blob_id] + ((2 * width) - 2),
                            2 * blobYCoord[blob_id] + ((2 * height) - 2),
                        ),
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    resized,
                    "{}".format("fm:" + str(fm_val)),
                    (
                        int(2 * (blobXCoord[blob_id]) + width),
                        int(2 * (blobYCoord[blob_id]) + height),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    resized,
                    "{}".format("cm:" + str(cm_val)),
                    (
                        int(2 * (blobXCoord[blob_id]) + width),
                        int(2 * (blobYCoord[blob_id]) + height) + 25,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    resized,
                    "{}".format("bestIdx:" + str(stack_id)),
                    (
                        int(2 * (blobXCoord[blob_id]) + width),
                        int(2 * (blobYCoord[blob_id]) + height) + 50,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
            cv2.imwrite(
                os.path.join(out_path, aoi_name + "_" + str(index) + ".jpeg"), resized
            )
            # exit()
raw_path ="/datadrive/wsi_data/compressed_data/H01CBA07P_13137/grid_4"
db_path = "/datadrive/wsi_data/compressed_data/H01CBA07P_13137/H01CBA07P_13137.db"

blob.populateFocusColorMetricValuesFromDB(raw_path,db_path)
