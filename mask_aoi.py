from hashlib import new
from cv2 import imshow
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sqlite3
import csv
import shutil
import math
import tarfile
import time
import pandas as pd
from os.path import join, exists 
import glob
import sys
import pandas as pd
from mpl_toolkits import mplot3d
import json

class FoldDetection():

    def __init__(self):
        self.uintMax = 255
    
    # |----------------------------------------------------------------------------|
    # dataParsing
    # |----------------------------------------------------------------------------|
    def dataParsing(self, filePath):
        frame = pd.read_csv(filePath, sep ='\s+', header = None)
        return frame
    # |----------------------End of dataParsing---------------------------|

    # |----------------------------------------------------------------------------|
    # get_coord_list
    # |----------------------------------------------------------------------------|
    def get_coord_list(self, logPath):
        status = False
        displacementMST_Y = logPath + "/final_displacements_Y.txt"
        displacementMST_X = logPath + "/final_displacements_X.txt"
        # print("displacementMST_X: ",displacementMST_X)
        if os.path.exists(displacementMST_Y) and\
            os.path.exists(displacementMST_Y):
            self.dataAfterOptimizationX = self.dataParsing(displacementMST_X)
            self.dataAfterOptimizationY = self.dataParsing(displacementMST_Y)
            status = True
        else:
            print(" couldn't find the coord files---")
        return status
    # |----------------------End of get_coord_list---------------------------|

    # |----------------------------------------------------------------------------|
    # get_aoi_panorama_coords
    # |----------------------------------------------------------------------------|
    def get_aoi_panorama_coords(self, rowIdx, colIdx):
        x_value = float(self.dataAfterOptimizationX.iloc[rowIdx, colIdx])
        y_value = float(self.dataAfterOptimizationY.iloc[rowIdx, colIdx])
        
        return x_value, y_value
    def get_row_col_idx(self, aoi_name, grid_rows, grid_cols):
        aoi_idx = int(aoi_name.split('aoi')[-1]) - 1

        row_id = int(aoi_idx/grid_cols)
        if row_id % 2 == 0:
            col_id = aoi_idx % grid_cols
        else:
            col_id = grid_cols - (aoi_idx % grid_cols) - 1
        return row_id, col_id


    def tile_classification(self, all_slides_path):
        for slide_name in os.listdir(all_slides_path):
            print("slide_name: ", slide_name)
            # try:
            slide_path = join(all_slides_path, slide_name)
            db_file_path = join(slide_path, slide_name + ".db")
            graph_json_obj = json.load(open(join(slide_path,"input_for_graph_algorithm.json")))  # read input_for_graph_algorithm.json file

            disp_out_path = join(slide_path, "disp_est_output")
            if not exists(disp_out_path): 
                print("Path doesn't exists: ", disp_out_path)
            logFlag = self.get_coord_list(disp_out_path)

            self._dbconn_sqlite = sqlite3.connect(db_file_path)
            c1 = self._dbconn_sqlite.cursor()

            c1.execute("select slide_row_idx, slide_col_idx,aoi_name from aoi")
            all_data = c1.fetchall()
        
            row_indices = [item[0] for item in all_data]
            col_indices = [item[1] for item in all_data]
            aoi_names_list   = [item[2] for item in all_data]

            slide_rows = np.max(row_indices) + 1
            slide_cols = np.max(col_indices) + 1

            mask = np.zeros((5000,5000),np.uint8)

            for i in range(len(aoi_names_list)):
                aoi_row, aoi_col = self.get_row_col_idx(aoi_names_list[i], slide_rows, slide_cols)
                aoi_row = row_indices[i]
                aoi_col = col_indices[i]

                aoi_row = aoi_row - graph_json_obj["start_row"]
                aoi_col = aoi_col - graph_json_obj["start_col"]
                print("aoi_row: ", aoi_row)
                print("aoi_row: ", aoi_col)
                # aoi_panoromaX,aoi_panoromaY = self.get_aoi_panorama_coords(aoi_row,aoi_col)
                aoi_panoromaX,aoi_panoromaY = self.get_aoi_panorama_coords(aoi_row,aoi_col)
                aoi_panoromaX = aoi_panoromaX / pow(2,8)
                aoi_panoromaY = aoi_panoromaY / pow(2,8)
                print("aoi_panoromaX : ",aoi_panoromaX)
                print("aoi_panoromaY : ",aoi_panoromaY)
                
                mask[int(aoi_panoromaX),int(aoi_panoromaY)]=255
            # cv2.imshow("mask",mask)

if __name__ == "__main__":
    Obj = FoldDetection()
    input_path = "/home/adminspin/Desktop/cameratemp"
    Obj.tile_classification(input_path)
