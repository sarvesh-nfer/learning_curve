import cv2
import sqlite3
import numpy
import statistics 
import sys
import os
import json
import time
import sys
from numpy.lib.function_base import append
import xlsxwriter
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d import Axes3D
from _collections import deque
from scipy.signal import argrelextrema

class ZPlaneEstimation_bend_analysis():
    """
    Module for analyzing 3d plane using all acquisition points and also using some
    initial points for creating initial global plane for acquisition.
    @note we only need db file for this analysis
    """
# |----------------------------------------------------------------------------|
# class Variables
# |----------------------------------------------------------------------------|
    #no classVariables
# |----------------------------------------------------------------------------|
# Constructor
# |----------------------------------------------------------------------------|
    def __init__(self):
        self.stack_size = 5
        self.fmThesh = 6.0
        self.cmThresh = 40.0
        self.z_step_size  = 1.875
        self.num_grids_slide = 0
# |-------------------------End of Constructor------------------------------|
    def getAoidetails(self,folder,slide_name,path_to_save):
        gridId = 1
        dump_images = 0
        self._dump_images = 0
        self.num_grids_slide = 0
        self.connection = sqlite3.connect(self._dbPath )
        self.cursor = self.connection.cursor()
        connection = sqlite3.connect(self._dbPath)
        cursor = connection.cursor()
        
        # get row and column count
        query = ("SELECT row_count, column_count, pattern, direction, grid_id FROM grid_info ORDER BY grid_id ASC")
        #print(query)
        self.cursor.execute(query)
        for i in self.cursor.fetchall():
            self.rowCount = i[0]; rowCount = i[0]
            self.colCount = i[1]; colCount = i[1]
            pattern = i[2]
            direction = i[3]
            #print("Grid id: ", i[4]," Grid size:", rowCount, colCount)
            #print("pattern: ", pattern, "direction: ", direction, "\n")

        ### getting grid info and starting row and column index
        query = ("SELECT grid_id, start_x, start_y, end_x, end_y,row_count, column_count FROM grid_info ORDER BY grid_id ASC")
        #print(query)
        cursor.execute(query)
        num_grids = len(cursor.fetchall())
        self.num_grids_slide = num_grids
        self.cursor.execute(query)
        x_min = 100000; y_min = 1000000; x_max = 0; y_max = 0
        self.grid_info = np.zeros((7,num_grids))
        i = 0
        for gridInfo in self.cursor.fetchall():
            self.grid_info[0][i] = int(gridInfo[0])
            self.grid_info[1][i] = gridInfo[1]
            self.grid_info[2][i] = gridInfo[2]
            self.grid_info[3][i] = gridInfo[3]
            self.grid_info[4][i] = gridInfo[4]
            self.grid_info[5][i] = gridInfo[5]
            self.grid_info[6][i] = gridInfo[6]
            i +=1

        self.x_min = min(self.grid_info[1]); self.x_max = max(self.grid_info[3])
        self.y_min = min(self.grid_info[2]); self.y_max = max(self.grid_info[4])
        #print("grid info: \n", self.grid_info,"\n")
        
        ##getting xy movement of the scanner x_movement and y_movement
        self.y_movement = 0; self.x_movement = 0
        self.get_xy_movement()
        
        ##total number of rows and columns in new complete slide grid
        self.total_rows_slide = int((self.y_max - self.y_min)/(self.y_movement))
        self.total_columns_slide = int((self.x_max - self.x_min)/(self.x_movement))
        self.total_rows_slide += 2; self.total_columns_slide += 2
        #print("total rows: ", self.total_rows_slide, "total_col: ", self.total_columns_slide)
        
        ## starting row and column index of all the grids
        self.grid_xy_index = np.zeros((5,num_grids))

        for i in range(0,num_grids):
            x_start = self.grid_info[1][i]; y_start = self.grid_info[2][i]
            #print(self.grid_info[0][i], "x_start: ", x_start, "y_start: ",y_start )
            #print(self.x_movement, self.y_movement)
            row_index = (y_start - self.y_min)/(self.y_movement)
            column_index = (x_start - self.x_min)/(self.x_movement)
            if(row_index- int(row_index) > 0.5):
                row_index = int(row_index)+1
            else:
                row_index = int(row_index)
            if(column_index-int(column_index) > 0.5):
                column_index = int(column_index)+1
            else:
                column_index = int(column_index)
            #print("row_index: ", row_index, "column_index: ", column_index)
            self.grid_xy_index[0][i] = self.grid_info[0][i]
            self.grid_xy_index[1][i] = row_index
            self.grid_xy_index[2][i] = column_index
            self.grid_xy_index[3][i] = self.grid_info[5][i]
            self.grid_xy_index[4][i] = self.grid_info[6][i]

        #print("grid_index_info: \n", self.grid_xy_index,"\n")

        ## get all focus sampling points
        query = ("SELECT aoi_name, row_index, column_index, best_z, focus_metric, color_metric, x_pos, y_pos, grid_id, status FROM focus_sampling_info ORDER BY grid_id ASC, row_index ASC, column_index ASC")
        #print(query)
        self.cursor.execute(query)
        self.focus_aoiName = []
        self.focus_aoiRowIdx = []
        self.focus_aoiColIdx = []
        self.focus_best_z = []
        self.focus_fm = []
        self.focus_cm = []
        self.focus_x = []
        self.focus_y = []
        self.focus_new_aoiRowIdx = []
        self.focus_new_aoiColIdx = []
        self.bestZ_grid = np.zeros((4,num_grids))
        #print("num_grids: ", num_grids)
        
        #print("Focus shampling points:")
        for aoiInfo in self.cursor.fetchall():
            x = aoiInfo[6]; y = aoiInfo[7]; row_idx = aoiInfo[1]; col_idx = aoiInfo[2]; grid_num = aoiInfo[8]
            new_row_idx = int(aoiInfo[1]+self.grid_xy_index[1][aoiInfo[8]-1])
            new_col_idx = int(aoiInfo[2]+self.grid_xy_index[2][aoiInfo[8]-1])
            #print(aoiInfo[0] ,"grid_id",aoiInfo[8], "row_idx: ", aoiInfo[1],"col_idx: ",aoiInfo[2]," x:", aoiInfo[6]," y: ", aoiInfo[7])
            #print("new row_idx: ", new_row_idx , "new col_idx: ", new_col_idx, "\n")
            #print("status: ", aoiInfo[9])
            if(aoiInfo[9] == 1):
                self.focus_aoiName.append(aoiInfo[0])
                self.focus_aoiRowIdx.append(aoiInfo[1])
                self.focus_aoiColIdx.append(aoiInfo[2])
                self.focus_best_z.append(aoiInfo[3])
                self.focus_fm.append(aoiInfo[4])
                self.focus_cm.append(aoiInfo[5])
                self.focus_x.append(aoiInfo[6])
                self.focus_y.append(aoiInfo[7])
                self.focus_new_aoiRowIdx.append(new_row_idx)
                self.focus_new_aoiColIdx.append(new_col_idx)
                ##storing centroid
                if(aoiInfo[5] == -1 and aoiInfo[4] == -1):
                    #print(grid_num-1)
                    self.bestZ_grid[0][grid_num-1] = grid_num
                    self.bestZ_grid[1][grid_num-1] = new_row_idx
                    self.bestZ_grid[2][grid_num-1] = new_col_idx
                    self.bestZ_grid[3][grid_num-1] = aoiInfo[3]
        
        #print("\n centroid bestZ from each grid: ", self.bestZ_grid, "\n")
                
        focus_aoiName = self.focus_aoiName
        focus_aoiRowIdx = self.focus_aoiRowIdx
        focus_aoiColIdx  = self.focus_aoiColIdx
        focus_best_z = self.focus_best_z
        focus_fm = self.focus_fm
        focus_cm = self.focus_cm
        focus_x = self.focus_x
        focus_y  = self.focus_y
        focus_new_aoiRowIdx = self.focus_new_aoiRowIdx
        focus_new_aoiColIdx = self.focus_new_aoiColIdx
        
        ## ALL aoi information
        self.cursor.execute("SELECT aoi_name, aoi_row_idx, aoi_col_idx, sampling_z, bg_state_fs, bg_state_acq, focus_metric, color_metric, best_idx, best_z, ref_z, z_value, aoi_x, aoi_y, grid_id FROM aoi ORDER BY grid_id ASC, aoi_row_idx ASC, aoi_col_idx ASC")
        self.aoiName = []
        self.aoiRowIdx = []
        self.aoiColIdx = []
        self.sampling_z = []
        self.fgbg_sampling = []
        self.background = []
        self.focusMetric = []
        self.colorMetric = []
        self.aoiBestIdx = []
        self.best_z = []
        self.ref_z = []
        self.interpolated_z = []
        self.new_aoiRowIdx = []
        self.new_aoiColIdx = []
        self.grid_id = []

        for aoiInfo in self.cursor.fetchall():
            gridID = aoiInfo[14]
            _new_aoiRowIdx = int(self.grid_xy_index[1][gridID-1] + aoiInfo[1])
            _new_aoiColIdx = int(self.grid_xy_index[2][gridID-1] + aoiInfo[2])
            self.aoiName.append(aoiInfo[0])
            self.aoiRowIdx.append(aoiInfo[1])
            self.aoiColIdx.append(aoiInfo[2])
            self.sampling_z.append(aoiInfo[3])
            self.fgbg_sampling.append(aoiInfo[4])
            self.background.append(aoiInfo[5]) 
            self.focusMetric.append(aoiInfo[6])
            self.colorMetric.append(aoiInfo[7])
            if aoiInfo[8] >= self.stack_size:
                self.aoiBestIdx.append(self.stack_size - 1)
            else:
                self.aoiBestIdx.append(aoiInfo[8])
            self.best_z.append(aoiInfo[9])
            self.ref_z.append(aoiInfo[10])
            self.interpolated_z.append(aoiInfo[11])
            self.new_aoiRowIdx.append(_new_aoiRowIdx)
            self.new_aoiColIdx.append(_new_aoiColIdx)
            self.grid_id.append(aoiInfo[14])

        cursor.execute("SELECT aoi_name, aoi_row_idx, aoi_col_idx, sampling_z, bg_state_fs, bg_state_acq, focus_metric, color_metric, best_idx, best_z, ref_z, z_value, aoi_x, aoi_y, grid_id FROM aoi ORDER BY grid_id ASC, aoi_row_idx ASC, aoi_col_idx ASC")
        aoiName = []
        aoiRowIdx = []
        aoiColIdx = []
        best_z = []
        ref_z = []
        new_aoiRowIdx = []
        new_aoiColIdx = []
        aoiBestIdx = []
        
        for aoiInfo in cursor.fetchall():
            gridID = aoiInfo[14]
            _new_aoiRowIdx = int(self.grid_xy_index[1][gridID-1] + aoiInfo[1])
            _new_aoiColIdx = int(self.grid_xy_index[2][gridID-1] + aoiInfo[2])
            #print("aoi_x: ", aoiInfo[12], " aoi_y: ", aoiInfo[13], " xmin: ", self.x_min, " y_min: ", self.y_min)
            #print("_new_aoiRowIdx: ", _new_aoiRowIdx, "_new_aoiColIdx: ", _new_aoiColIdx, "\n")
            if(aoiInfo[8]!=-1 and aoiInfo[6]>self.fmThesh and aoiInfo[7]>self.cmThresh ):
                aoiName.append(aoiInfo[0])
                aoiRowIdx.append(aoiInfo[1])
                aoiColIdx.append(aoiInfo[2])
                best_z.append(aoiInfo[9])
                ref_z.append(aoiInfo[10])
                new_aoiRowIdx.append(_new_aoiRowIdx)
                new_aoiColIdx.append(_new_aoiColIdx)
                if aoiInfo[8] >= self.stack_size:
                    aoiBestIdx.append(self.stack_size - 1)
                else:
                    aoiBestIdx.append(aoiInfo[8])
        zdiff_best_ref= []
        
        
        ##plotting complete image
        self.image = np.zeros((self.total_rows_slide, self.total_columns_slide,3), dtype = "uint8")
        #print("image: ", self.image.shape)
        #plt.imshow(self.image, cmap="gray"); plt.show()
        
        ##ploting all points
        self.image_all_points = self.image.copy()
        for i in range(0,len(new_aoiRowIdx)):
            row = new_aoiRowIdx[i]; column = new_aoiColIdx[i]
            self.image_all_points[int(row),int(column),:] = [255, 255,255]
            
        ##ploting all focus sampling points
        #self.image_focus = self.image.copy()
        self.image_focus = self.image_all_points.copy()
        for i in range(0,len(self.focus_new_aoiRowIdx)):
            row = self.focus_new_aoiRowIdx[i]; column = self.focus_new_aoiColIdx[i]
            self.image_focus[int(row),int(column),:] = [0, 255,0]
        
        ##ploting centroid on slide image
        self.image_bestZ = self.image.copy()
        self.image_bestZ = self.image_all_points.copy()
        for i in range (0,self.bestZ_grid[0].size):
            row = self.bestZ_grid[1][i]; column = self.bestZ_grid[2][i]
            #print("grid num: ", self.bestZ_grid[0][i], "row: ", row, "column: ", column)
            self.image_bestZ[int(row),int(column),:] = [255, 0,0]
        
        # fig, ax= plt.subplots(1,3)
        # ax[0].imshow(self.image_all_points); ax[1].imshow(self.image_focus);ax[2].imshow(self.image_bestZ); plt.show()
        #

        complete_path = self.__path___ + "/" +'slide_FS' + '.png'; #print(complete_path)
        self.resizeImage(self.image_focus,complete_path)
        return (cursor,rowCount, colCount, aoiRowIdx, aoiColIdx, best_z , ref_z , zdiff_best_ref, focus_aoiName, focus_aoiRowIdx, focus_aoiColIdx, focus_best_z, focus_fm,focus_cm, new_aoiRowIdx ,new_aoiColIdx, focus_new_aoiRowIdx, focus_new_aoiColIdx, self.bestZ_grid, aoiName ,aoiBestIdx)

    def get_xy_movement(self):
        query = "SELECT aoi_x, aoi_y, aoi_row_idx, aoi_col_idx,grid_id FROM aoi ORDER BY grid_id ASC, aoi_row_idx ASC, aoi_col_idx ASC"
        #print(query)
        self.cursor.execute(query)
        i =0
        x_last = 1000000; y_last = 1000000
        x_movement =1000000; y_movement = 1000000
        for aoiInfo in self.cursor.fetchall():
            if(i==0):
                x_last = aoiInfo[0]; y_last = aoiInfo[1]
                row_index_last = aoiInfo[2]
                column_index_last = aoiInfo[3]
                i = i+1
                continue
            if(aoiInfo[3] != column_index_last and x_movement == 1000000):
                x_movement = aoiInfo[0]-x_last
            if(aoiInfo[2] != row_index_last and y_movement == 1000000 ):
                y_movement = aoiInfo[1]-y_last
            x_last = aoiInfo[0]; y_last = aoiInfo[1]
            row_index_last = aoiInfo[2]; column_index_last = aoiInfo[3]
            if(x_movement!=1000000 and y_movement!=1000000):
                break
        #print("x_movement: ", x_movement, " y_movement: ", y_movement)
        self.x_movement = x_movement; self.y_movement = y_movement
        
    def plot_points(self, focus_aoiColIdx, focus_aoiRowIdx, focus_best_z, aoiColIdx, aoiRowIdx, best_z):
        focus_aoiColIdx = self.focus_new_aoiColIdx
        focus_aoiRowIdx = self.focus_new_aoiRowIdx
        focus_best_z = focus_best_z
 
        ### focus sampling points
        #print("\n\n","#"*10,"Plotting focus sampling points", "#"*10)
        y = np.array(focus_aoiColIdx)
        x = np.array(focus_aoiRowIdx)
        z = np.array(focus_best_z)

        #print("number of points: ",len(x), len(y), len(z))
        #print("max z: ", max(z), "min z: ", min(z), "z_diff: ", max(z)-min(z))

        ## plot 2
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(x, y, z, color='b');plt.show()
        
        ## all points
        aoiColIdx = aoiColIdx
        aoiRowIdx = aoiRowIdx
        best_z = best_z
        #print("\n\n","#"*10,"Plotting all points points", "#"*10)
        y = np.array(aoiColIdx)
        x = np.array(aoiRowIdx)
        z = np.array(best_z)
        
        #print("number of points: ",len(x), len(y), len(z))
        #print("max z: ", max(z), "min z: ", min(z), "z_diff: ", max(z)-min(z))

        ## plot 2
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z, color='b')
        complete_path =self.__path___ + '/AllPts_Scattered' + '.png'
        plt.savefig(complete_path , dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
        
        
            
    def best_points_from_centroid(self,focus_new_aoiRowIdx, focus_new_aoiColIdx , focus_best_z):
        focus_aoiRowIdx = focus_new_aoiRowIdx
        focus_aoiColIdx = focus_new_aoiColIdx
        focus_best_z = focus_best_z
        centroid_grid = self.bestZ_grid
        
        num_points = len(focus_aoiRowIdx)
        all_points = np.full((3,num_points), -1, dtype=np.float32)
        for j in range(0,len(focus_aoiRowIdx)):
            all_points[0][j] = focus_aoiRowIdx[j]
            all_points[1][j] = focus_aoiColIdx[j]
            all_points[2][j] = focus_best_z[j]
            
        distance = np.full((1,num_points), -1, dtype=np.float32)
    
        for j in range (0,len(focus_aoiRowIdx)):
            point_dis = 1000000000
            for i in range (0,centroid_grid[0].size):
                xc = centroid_grid[1][i]; yc = centroid_grid[2][i]
                x = focus_aoiRowIdx[j]; y =  focus_aoiColIdx[j]
                dis = (x-xc)*(x-xc) + (y-yc)*(y-yc)
                #print("\n xc: ",xc, " yc: ", yc, " dis: ", dis)
                if(dis <= point_dis):
                    distance[0][j] = dis
                    point_dis = dis

        #print("distance: ", distance)            
        #print("all points: \n", all_points)
        
        i = np.argsort(-1*distance)
        all_points = all_points[:,i]
        #print("\n distance: ", distance)            
        #print("all points: \n", all_points)
        return(all_points)
       
 
    ##Computing least-squares solution to equation Ax = b
    def plane_fitting_max_point(self,x,y,z):
        y = y
        x = x
        z = z
        num_points = len(x)
        z_step_size = 1.875

        # plot raw data
        # plt.figure()
        # ax = plt.subplot(111, projection='3d')
        # ax.scatter(x, y, z, color='b')

        ## ax+by+c = z, Ax=B
        tmp_A = []
        tmp_b = []
        for i in range(len(x)):
            tmp_A.append([x[i], y[i], 1])
            tmp_b.append(z[i])

        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)

        from scipy.linalg import lstsq
        fit, residual, rnk, s = lstsq(A, b)
        errors =  A * fit - b
        
        absolute_sum = np.sum(abs(errors))
        absolute_sum_mean = absolute_sum/len(errors)
        _max_step_size =  max(abs(errors))/z_step_size
        _max_positive_step_size =  max(errors)/self.z_step_size
        _max_negative_step_size = min(errors)/self.z_step_size
        #print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
        #print("absolute error: ",absolute_sum," absolute mean error: ", absolute_sum_mean)
        #print("max shift: ", max(abs(errors)), "max_step_size: ", max(abs(errors))/z_step_size)
        #print("max positive shift: ", max(errors), "max_step_size: ", _max_positive_step_size)
        #print("max negative shift: ", min(errors), "max_step_size: ", _max_negative_step_size)

        # plot plane
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
        #                   np.arange(ylim[0], ylim[1]))
        # Z = np.zeros(X.shape)
        # for r in range(X.shape[0]):
        #     for c in range(X.shape[1]):
        #         Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
        # ax.plot_wireframe(X,Y,Z, color='k')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        return(fit)
         
    def plot_plane_on_points(self,fit,x,y,z):
        y = y
        x = x
        z = z
        
        # plot raw data
        # plt.figure()
        # ax = plt.subplot(111, projection='3d')
        # ax.scatter(x, y, z, color='b')
        #
        # # plot plane
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
        #                   np.arange(ylim[0], ylim[1]))
        # Z = np.zeros(X.shape)
        # for r in range(X.shape[0]):
        #     for c in range(X.shape[1]):
        #         Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
        # ax.plot_wireframe(X,Y,Z, color='k')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        
    def get_max_diff_z_values(self, x, y,z,x_movement, y_movement, one_pixel_x, one_pixel_y):
        act_max_z_diff = 0
        act_max_z_diff = max(z)-min(z)
        act_minz_index = np.argmin(z)
        act_zmin = z[act_minz_index]
        act_minz_x = x[act_minz_index]
        act_minz_y = y[act_minz_index]
        act_zmax_index = np.argmax(z)
        act_zmax = z[act_zmax_index]
        act_maxz_x = x[act_zmax_index]
        act_maxz_y = y[act_zmax_index]
        col_diff = abs(act_maxz_y - act_minz_y)
        row_diff = abs(act_maxz_x - act_minz_x)
        dis_away_aoi = int(np.sqrt(np.square(row_diff) + np.square(col_diff)))
        x_diff_um = col_diff * x_movement * one_pixel_x
        y_diff_um = row_diff * y_movement * one_pixel_y
        dis_away_um = int(np.sqrt(np.square(x_diff_um) + np.square(y_diff_um)))
        dis_away_mm = dis_away_um * 0.001
        z_range = 2*dis_away_mm
        return(act_max_z_diff, dis_away_mm, dis_away_aoi)
        
    def matrix_calculation(self,x,y,z,z_plane,fit,errors):
        z_step_size = 1.875
        x_movement = self.x_movement #12.41294642857143
        y_movement =  self.y_movement #7.778333333333364
        one_pixel_y = 41.5512
        one_pixel_x = 41.1764
        num_points = len(errors)
        sensitive_step_range = 2 #5 steps-> -2,-1,0,1,2
        num_0_steps = 0; num_minus1_steps = 0; num_1_step = 0
        num_minus2_steps = 0; num_2steps = 0
        num_more_steps = 0; num_minusmore_steps = 0
        pts_in_range = 0; perc_in_range = 0
        pts_out_range = 0; perc_out_range = 0
        act_max_z_diff = 0;dis_away_mm =0;  dis_away_aoi =0
        pln_max_z_diff =0; pln_dis_away_mm=0; pln_dis_away_aoi =0
        row_start =0 ; col_start =0
        row_extreme = 170; col_extreme =50

        ##numpy matrix to numpy array
        z_plane = np.squeeze(np.asarray(z_plane.copy()))
        _errors = np.squeeze(np.asarray(errors.copy()))
        _step_error = np.fix(np.divide(_errors,z_step_size))
        number_list = np.array(_step_error)
        (unique, counts) = numpy.unique(number_list, return_counts=True)
        frequencies = numpy.asarray((unique, counts)).T
        #print("frequency of all steps:",frequencies)
        
        for f in range(frequencies.shape[0]):
            _step = frequencies[f,0]
            freq = frequencies[f,1]
            if(_step < (-1*sensitive_step_range)):
                num_minusmore_steps += freq
            elif(_step == -2 ):
                num_minus2_steps += freq
            elif(_step == -1 ):
                num_minus1_steps += freq
            elif(_step == 0 ):
                num_0_steps += freq
            elif(_step == 1 ):
                num_1_step += freq
            elif(_step == 2 ):
                num_2steps += freq
            elif(_step > sensitive_step_range):
                num_more_steps += freq
                
        #print( num_minusmore_steps, num_minus2_steps, num_minus1_steps, num_0_steps, num_1_step, num_2steps, num_more_steps)
        pts_out_range = num_minusmore_steps + num_more_steps
        perc_out_range = (pts_out_range/num_points)*100
        perc_in_range = 100 - perc_out_range
        #print("perc_in_range_2: ", perc_in_range, "perc_out_range_2: ", perc_out_range)
        
        ##maximum z difference between aois from actual z , same for plane z
        act_max_z_diff, dis_away_mm, dis_away_aoi = self.get_max_diff_z_values(x, y,z,x_movement, y_movement, one_pixel_x, one_pixel_y)
        pln_max_z_diff, pln_dis_away_mm, pln_dis_away_aoi = self.get_max_diff_z_values(x, y,z_plane,x_movement, y_movement, one_pixel_x, one_pixel_y)
        act_max_z_diff_xy_dis_mm = dis_away_mm
        pln_max_z_xy_dis_mm = pln_dis_away_mm
        #print("act_max_z_diff: ", act_max_z_diff, "dis_away_mm: ", dis_away_mm, "dis_away_aoi: ", dis_away_aoi)
        #print("pln_max_z_diff: ", pln_max_z_diff, "pln_dis_away_mm: ", pln_dis_away_mm, "pln_dis_away_aoi: ", pln_dis_away_aoi)

        ##checking the extremities, plane is correct or not
        row_start =0 ; col_start =0
        row_extreme = 170; col_extreme =50
        z_start = fit[0]*row_start + fit[1]*col_start + fit[2]
        z_end = fit[0]*row_extreme + fit[1]*col_extreme + fit[2]
        z_diff_extreme1 = z_end - z_start
        
        z_top_right = fit[0]*row_start + fit[1]*col_extreme+ fit[2]
        z_bottom_left = fit[0]*row_extreme + fit[1]*col_start + fit[2]
        z_diff_extreme2 = z_top_right - z_bottom_left
        
        x_diff_um = (col_extreme-col_start) * x_movement * one_pixel_x
        y_diff_um = (row_extreme - row_start) * y_movement * one_pixel_y
        dis_away_um = int(np.sqrt(np.square(x_diff_um) + np.square(y_diff_um)))
        dis_away_extreme_mm = dis_away_um * 0.001
        #print("dis_away_extreme_mm: ", dis_away_extreme_mm,"z_diff_extreme1: ", z_diff_extreme1, "z_diff_extreme2: ", z_diff_extreme2 )
        z_diff_extreme1_per_mm = z_diff_extreme1/dis_away_extreme_mm
        z_diff_extreme2_per_mm = z_diff_extreme2/dis_away_extreme_mm
        
        ##angles with xy plane, yz plane and zx plane
        denomitator = np.sqrt(np.square(fit[0])+np.square(fit[1])+np.square(-1))
        cos_xy = -1/denomitator
        if(fit[0]==0 and fit[1] == 0): cos_xy = 1 ##z=0 xy plane
        angle_xy_plane = math.degrees (math.acos (cos_xy))
        cos_yz = fit[0]/denomitator
        angle_yz_plane = math.degrees (math.acos (cos_yz))
        cos_zx = fit[1]/denomitator
        angle_zx_plane = math.degrees (math.acos (cos_zx))
        #print("angle_xy: ", angle_xy_plane, "angle_yz: ",angle_yz_plane, "angle_zx: ", angle_zx_plane)

        return(perc_in_range, act_max_z_diff, dis_away_mm , pln_max_z_diff, pln_dis_away_mm, z_diff_extreme1, z_diff_extreme2, dis_away_extreme_mm, num_minus2_steps, num_minus1_steps, num_0_steps, num_1_step, num_2steps, angle_xy_plane,angle_yz_plane, angle_zx_plane )     
        
    def plane_correlation_with_new_points(self,fit, x,y,z,name,slide_name,allpointAnalysis,count):
        z_step_size = 1.875
        ### all points
        y = y
        x = x
        z = z
        num_points = len(x)

        ## ax+by+c = z, Ax=B
        tmp_A = []
        tmp_b = []
        for i in range(len(x)):
            tmp_A.append([x[i], y[i], 1])
            tmp_b.append(z[i])

        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)
        b_plane = A * fit
        errors = b_plane - b
        plt.hist(errors)
        plt.xlabel("Z diff = plane_z-best_z")
        plt.ylabel("frequency")
        if(name == "all_points"):
            complete_path =self.__path___ + '/hist_OfAllPts_AllPtsPlane' + '.png'
        plt.savefig(complete_path , dpi=300, bbox_inches='tight')
        # plt.show()
        plt.clf()
        ## sum of absolute error
        absolute_sum = np.sum(abs(errors))
        absolute_sum_mean = absolute_sum/len(errors)
        _max_step_size =  max(abs(errors))/z_step_size
        _max_positive_step_size =  max(errors)/self.z_step_size
        _max_negative_step_size  = min(errors)/self.z_step_size
        
        if(_max_negative_step_size >0): _max_negative_step_size[0,0]=0
        if(_max_positive_step_size <0): _max_positive_step_size[0,0]=0

        #print("absolute mean error: ", absolute_sum_mean)
        #print("max shift: ", max(abs(errors)), "max_step_size: ", max(abs(errors))/z_step_size)
        #print("max positive shift: ", max(errors), "max_step_size: ", _max_positive_step_size)
        #print("max negative shift: ", min(errors), "max_step_size: ", _max_negative_step_size)
        
        list_to_store = []
        list_to_store.append(slide_name)
        list_to_store.append(absolute_sum_mean)
        list_to_store.append(num_points)
        list_to_store.append(int(_max_positive_step_size[0,0]))
        list_to_store.append(int(_max_negative_step_size[0,0]))
        
        ##____________________________matrix calculation______________________##
        #print("P"*30,"matrix_calculation","P"*30)
        perc_in_range, act_max_z_diff, act_max_z_diff_xy_dis_mm , pln_max_z_diff,pln_max_z_xy_dis_mm, z_diff_extreme1, z_diff_extreme2,dis_away_extreme_mm, num_minus2_steps, num_minus1_steps, num_0_steps, num_1_step, num_2steps, angle_xy_plane,angle_yz_plane, angle_zx_plane = self.matrix_calculation(x,y,z,b_plane,fit,errors)
        
        if(act_max_z_diff_xy_dis_mm == 0):    act_max_z_diff_xy_dis_mm =1
        if(pln_max_z_xy_dis_mm == 0): pln_max_z_xy_dis_mm=1
        
        act_max_z_diff_per_mm = act_max_z_diff/act_max_z_diff_xy_dis_mm
        pln_max_z_diff_per_mm =  pln_max_z_diff/pln_max_z_xy_dis_mm
        z_diff_extreme1_per_mm = z_diff_extreme1/dis_away_extreme_mm
        z_diff_extreme1_per_mm = z_diff_extreme1_per_mm[0]
        z_diff_extreme2_per_mm = z_diff_extreme2/dis_away_extreme_mm
        z_diff_extreme2_per_mm = z_diff_extreme2_per_mm[0]
         
        #print("\nMAE: ", absolute_sum_mean)
        #print("perc_in_range_2: ", perc_in_range)    
        #print("act_max_z_diff_per_mm: ", act_max_z_diff_per_mm, "\npln_max_z_diff_per_mm: ", pln_max_z_diff_per_mm)
        #print("z_diff_extreme1_per_mm: ", z_diff_extreme1_per_mm, "\nz_diff_extreme2_per_mm: ", z_diff_extreme2_per_mm)
        #print("P"*80)
        ##_____________________#matrix calculation ends________________________##

        list_to_store.append(perc_in_range)
        list_to_store.append(act_max_z_diff)
        list_to_store.append(act_max_z_diff_xy_dis_mm)
        list_to_store.append(act_max_z_diff_per_mm)
        list_to_store.append(pln_max_z_diff)
        list_to_store.append(pln_max_z_xy_dis_mm)
        list_to_store.append(pln_max_z_diff_per_mm)
        list_to_store.append(z_diff_extreme1)
        list_to_store.append(z_diff_extreme1_per_mm)
        list_to_store.append(z_diff_extreme2)
        list_to_store.append(z_diff_extreme2_per_mm)
        list_to_store.append(num_minus2_steps)
        list_to_store.append(num_minus1_steps)
        list_to_store.append(num_0_steps)
        list_to_store.append(num_1_step)
        list_to_store.append(num_2steps)
        list_to_store.append(angle_xy_plane)
        list_to_store.append(angle_yz_plane)
        list_to_store.append(angle_zx_plane)
        list_to_store.append(self.num_grids_slide)
        
        bow_pattern = 0
        bow_pattern = self.is_bow_present(fit)
        list_to_store .append(bow_pattern)
        
        for _num in range(0,len(list_to_store)):
            if(name == "all_points"):
                allpointAnalysis.write(count+1, _num, list_to_store[_num])
            
    def resizeImage(self, image, imgName):
        self.rowCount = self.total_rows_slide
        self.colCount = self.total_columns_slide
        factor = 7000//self.rowCount
        dim = (self.colCount*factor, self.rowCount * factor)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        resizedWithGrid = self.plotGrid(resized, factor)
        cv2.imwrite(imgName, resizedWithGrid)
        return resizedWithGrid
    
    def plotGrid(self, image, factor):
        img3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rowNo, colNo = img3.shape
        for rowIdx in range(0, rowNo, factor):
            for colIdx in range(0, colNo):
                image[rowIdx, colIdx, :] = 50
                image[rowIdx+1, colIdx, :] = 50
        for colIdx in range(0, colNo, factor):
            for rowIdx in range(0, rowNo):
                image[rowIdx, colIdx, :] = 50
                image[rowIdx, colIdx+1, :] = 50
        return image
        
    def generate_plane_estimation_best_index_heat_map(self, aoiName, zDiff, RowIdx, ColIdx , stack_size,name, zValue):
        try:
            self._stack_size_list = stack_size
            self._folderPath = self._folderPath
            aoiName = aoiName
            zDiff = zDiff
            RowIdx = RowIdx
            ColIdx = ColIdx
            zValue = zValue
            self.maxIdx = self._stack_size_list - 1
            # Best index map with color codes, upto stack of 9
            img = self.image.copy()
            for i in range(0, len(aoiName)):
                if(zValue[i] != -1):
                    z_diff = zDiff[i]
                    steps = int(z_diff / self.z_step_size)
                    best_index = int(self._stack_size_list/2) + steps
                    if best_index >= self._stack_size_list:
                        best_index = self._stack_size_list - 1
                    if best_index < 0:
                        best_index = 0

                    if best_index >= 0:
                        if(steps == -9):
                            img[RowIdx[i], ColIdx[i], :] = [0, 153, 255]  # light orange                   
                        elif(steps == -8):
                            img[RowIdx[i], ColIdx[i], :] = [255, 255,0 ]  # shining blue
                        elif(steps == -7):
                            img[RowIdx[i], ColIdx[i], :] = [128, 128, 0]  # military green
                        elif(steps == -6):
                            img[RowIdx[i], ColIdx[i], :] = [255, 128, 128]  # cream pink                      
                        elif(steps == -5):
                            img[RowIdx[i], ColIdx[i], :] = [255,204, 204]  # purple
                        elif(steps == -4):
                            img[RowIdx[i], ColIdx[i], :] = [0, 0, 255]  # Red
                        elif(steps == -3):
                            img[RowIdx[i], ColIdx[i], :] = [0, 165, 255]  # Orange
                        elif(steps == -2):
                            img[RowIdx[i], ColIdx[i], :] = [0, 255, 255]  # Yello
                        elif(steps == -1):
                            img[RowIdx[i], ColIdx[i], :] = [0, 255, 0]  # Green
                        elif(steps == 0):
                            img[RowIdx[i], ColIdx[i], :] = [255, 0, 0]  # Blue
                        elif(steps == 1):
                            img[RowIdx[i], ColIdx[i], :] = [130,0,75]  # Indigo
                        elif(steps == 2):
                            img[RowIdx[i], ColIdx[i], :] = [238,130,238]  # violet
                        elif(steps == 3):
                            img[RowIdx[i], ColIdx[i], :] = [127,127,127]  # grey
                        elif(steps == 4):
                            img[RowIdx[i], ColIdx[i], :] = [255,153,204]  # light blue
                        elif(steps == 5):
                            img[RowIdx[i], ColIdx[i], :] = [204,255,204]  # light green
                        elif(steps == 6):
                            img[RowIdx[i], ColIdx[i], :] = [153,204,255]  # cream
                        elif(steps == 7 ):
                            img[RowIdx[i], ColIdx[i], :] = [0,51,51]  # dark grey
                        elif(steps == 8 ):
                            img[RowIdx[i], ColIdx[i], :] = [0,51,153]  # maroon
                        elif(steps == 9 ):
                            img[RowIdx[i], ColIdx[i], :] = [102,51,153]  # maroon pink
                        elif(steps == 10 ):
                            img[RowIdx[i], ColIdx[i], :] = [51,51,51]  # darl grey
                        else:
                            img[RowIdx[i], ColIdx[i], :] = [255,255,255]  # white
                            
            path = self.__path___
            if not os.path.exists(path): os.mkdir(path)
            complete_path = path + "/" +name +'_plane_estimated_best_index_heat_map' + '.jpeg'
            #print(complete_path)
            self.resizeImage(img,complete_path)
        except Exception as msg:
            print("Exception occurred, ", msg)
            print(i)
        
    def plot_from_all_points_plane(self,new_aoiRowIdx ,new_aoiColIdx,best_z,aoiName ,aoiBestIdx):
        #print("*"*30, " fitting all points ", "*"*30)
        x = new_aoiRowIdx
        y = new_aoiColIdx
        z = best_z
        num_points = len(x)
        fit = self.plane_fitting_max_point(x,y,z)
        
        ##predict for all Aois
        x = self.new_aoiRowIdx
        y = self.new_aoiColIdx
        z = self.best_z
        self.z_step_size = 1.875

        ## ax+by+c = z, Ax=B
        tmp_A = []
        tmp_b = []
        for i in range(len(x)):
            tmp_A.append([x[i], y[i], 1])
            tmp_b.append(z[i])

        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)
        #print(" calculating all points plane z value")
        z_new = A * fit
        zDiff = z_new - b
        #Acqusition sampling best index image
        self.generate_plane_estimation_best_index_heat_map(self.aoiName, zDiff, self.new_aoiRowIdx, self.new_aoiColIdx , 11 ,"allPoints_acquisition", self.aoiBestIdx)
        
        ##calculating on foreground where color metric and focus metric is more than threshold
        x = new_aoiRowIdx; y = new_aoiColIdx;  z = best_z
        ## ax+by+c = z, Ax=B
        tmp_A = []
        tmp_b = []
        for i in range(len(x)):
            tmp_A.append([x[i], y[i], 1])
            tmp_b.append(z[i])

        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)
        z_new = A * fit
        z_diff_plane_estimation = z_new - b
        self.generate_plane_estimation_best_index_heat_map(aoiName, z_diff_plane_estimation, new_aoiRowIdx, new_aoiColIdx , 11 ,"forg_allPoints_acquisition", aoiBestIdx)
        return fit
# |----------------------------------------------------------------------------|
# is_bow_present
# |----------------------------------------------------------------------------|
    def is_bow_present(self,allPts_fit):
        try:
            rows = self.total_rows_slide
            columns= self.total_columns_slide
            all_grid_aoi = np.full((int(rows),int(columns)),-1)   
            for num in range(0,len(self.aoiName)):
                all_grid_aoi[self.new_aoiRowIdx[num]][self.new_aoiColIdx[num]] = -1
                if(self.aoiBestIdx[num]!=-1 and self.focusMetric[num]>6.0 and self.colorMetric[num]>40):
                    all_grid_aoi[self.new_aoiRowIdx[num]][self.new_aoiColIdx[num]] = self.best_z[num]
            
            _count_below_plane = []
            _count_on_plane = []
            _count_above_plane = []
            _row_num = []
            for r in range(rows):
                count_below_plane = 0
                count_on_plane = 0
                count_above_plane = 0
                complete_row_background = 1
                for c in range(columns):
                    z_acq  = all_grid_aoi[r][c]
                    if(z_acq!=-1):
                        complete_row_background = 0
                        z_plane = (allPts_fit[0]*r) + (allPts_fit[1]*c) + (allPts_fit[2])
                        z_diff = z_plane - z_acq
                        if(z_diff < 0 ):
                            count_above_plane +=1
                        elif(z_diff == 0):
                            count_on_plane +=1
                        else:
                            count_below_plane +=1
                if(complete_row_background == 0 or True):
                    _row_num.append(r)
                    _count_above_plane.append(count_above_plane)
                    _count_on_plane.append(count_on_plane)
                    _count_below_plane.append(count_below_plane)
    
            
            plt.scatter(_row_num, _count_above_plane,marker = 'o', label="count_above_plane")
            plt.scatter(_row_num, _count_below_plane,marker='+',label="count_below_plane")
            plt.xlabel("_row_num")
            plt.ylabel("aoi_count")
            plt.legend(loc='upper left')
            complete_path =self.__path___ + '/bend_aoi_count_plane' + '.png'
            plt.savefig(complete_path , dpi=300, bbox_inches='tight')
            plt.clf()
            
            # analysis of count above plane -> minima maxima minima
            bend1=0
            bend2=0
            bend=0
            _array_above_plane= np.array(_count_above_plane)
            row_start = 0 
            row_1 = int(rows/3)
            row_2 = int((2*rows)/3)
            row_3 = int(rows)-1
            
            a_mean_1 = np.mean(_array_above_plane[row_start:row_1])
            a_mean_2 = np.mean(_array_above_plane[row_1:row_2])
            a_mean_3 = np.mean(_array_above_plane[row_2:row_3])
            # print("mean_1: ", a_mean_1, "mean_2: ", a_mean_2, "mean_3: ", a_mean_3)
            if(a_mean_2>a_mean_1 and a_mean_2>a_mean_3):
                bend1 = 1
                
            # analysis of count below plane -> maxima->minima->maxima
            _array_below_plane = np.array(_count_below_plane)
            
            b_mean_1 = np.mean(_array_below_plane[row_start:row_1])
            b_mean_2 = np.mean(_array_below_plane[row_1:row_2])
            b_mean_3 = np.mean(_array_below_plane[row_2:row_3])
            # print("mean_1: ", b_mean_1, "mean_2: ", b_mean_2, "mean_3: ", b_mean_3)
            if(b_mean_2<b_mean_1 and b_mean_2<b_mean_3):
                bend2 = 1
                
            if(bend1 ==1 and bend2 ==1):
                bend=1
            else:
                bend=0
            return(bend)
        except Exception as msg:
            print("Exception occurred, ", msg)
            print(0)     
# |----------------------End of is_bow_present---------------------------|
# |----------------------------------------------------------------------------|
# process
# |----------------------------------------------------------------------------|
    def process(self,path, slide_name,allpointAnalysis,count,path_to_save):
        try:
            #db path
            self._folderPath = path + "/" + slide_name +"/data/"
            self._dbPath = self._folderPath + '/' + slide_name + '.db'
            if not os.path.exists(self._dbPath):
                self._folderPath = path + "/" + slide_name
                self._dbPath = path + "/" + slide_name +'/' + slide_name + '.db'
            self.__path___ = path_to_save + "/" + "plane_analysis" + "/" + slide_name
            if not os.path.exists(self.__path___):
                os.makedirs(self.__path___)
                
            print("\n\n","*"*50,slide_name, "*"*50, "\n","*"*110)
            cursor,rowCount, colCount, aoiRowIdx, aoiColIdx, best_z , ref_z , zdiff_best_ref, focus_aoiName, focus_aoiRowIdx, focus_aoiColIdx, focus_best_z, focus_fm,focus_cm, new_aoiRowIdx ,new_aoiColIdx,  focus_new_aoiRowIdx, focus_new_aoiColIdx, bestZ_grid, aoiName ,aoiBestIdx  = self.getAoidetails(path,slide_name,path_to_save)
            self.plot_points(focus_new_aoiColIdx, focus_new_aoiRowIdx, focus_best_z,  new_aoiColIdx, new_aoiRowIdx, best_z)
            
            ###creating plane from all points
            allPts_fit = self.plot_from_all_points_plane(new_aoiRowIdx ,new_aoiColIdx,best_z,aoiName ,aoiBestIdx)
            #print("\n","#"*10, "Fitting all points on Plane using all points", "#"*10, "\n")
            y = np.array(new_aoiColIdx)
            x = np.array(new_aoiRowIdx)
            z = np.array(best_z)
            self.plane_correlation_with_new_points(allPts_fit, x,y,z,"all_points", slide_name,allpointAnalysis,count)
            print("\n\n","*"*50,"END", "*"*50, "\n")
        except Exception as msg:
            print("[error]: ", msg)
# |----------------------End of process---------------------------|
# |----------------------------------------------------------------------------|
# main
# |----------------------------------------------------------------------------|
def main():
    """
    @note path_to_save : folder where all data will be saved
    @note path: folder which contains different slides, which need to be processed
    @note Create folder by name plane_analysis
    @note Create excel sheet by name plane_analysis.xlsx
    """
    path_to_save = "/home/adminspin/Desktop/plane/sheet3_plane"
    path = "/home/adminspin/Desktop/plane/sheet3/"
    excel_path = os.path.join(path_to_save,"plane_analysis.xlsx")
    workbook_plane_analysis = xlsxwriter.Workbook(excel_path)
    allpointAnalysis = workbook_plane_analysis.add_worksheet("allpoint_Analysis")

    list = ["slideName",
            "MAE",
            "total_aoi",
            "max_pos_steps",
            "max_neg_steps",
            "% inRange Aois",
            "act_max_z_diff",
            "act_max_z_diff_xy_dismm",
            "act_max_z_diff_per_mm",
            "pln_max_z_diff",
            "pln_max_z_xy_dismm",
            "pln_max_z_diff_per_mm",
            "z_diff_extreme1",
            "z_diff_extreme1_per_mm",
            "z_diff_extreme2",
            "z_diff_extreme2_per_mm",
            "num_minus2_steps",
            "num_minus1_steps",
            "num_0_steps",
            "num_1_step",
            "num_2steps",
            "angle_xy",
            "angle_yz",
            "angle_zx",
            "num_grids",
            "bow_pattern"
             ]
    for _num in range(0,len(list)):
        allpointAnalysis.write(0,_num,list[_num])
    
    excel_line =0
    slide_names = os.listdir(path)
    slide_names = sorted(slide_names, reverse=True)
    #slide_names = ["X7422_90_1"]
    total_slide = len(slide_names)
    _obj = ZPlaneEstimation_bend_analysis()
    for i in range (0,len(slide_names)):
        if(i>=total_slide):
            break
        slide = slide_names[i]
        if slide == ".DS_Store": continue
        _obj.process(path, slide,allpointAnalysis,excel_line,path_to_save)
        excel_line +=1
    workbook_plane_analysis.close()
 # |----------------------End of main-----------------------------------|   
if __name__ == "__main__":
    main()
    """
    ########check######
    @note path_to_save in  main(): folder where all data will be saved
    @note path in main (): folder which contains different slides, which need to be processed
    @note _dbPath in process(): db file path
    """
