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
try:
    from disp_plots import DisplacementValuesComparison
except Exception as msg:
    print("unable to import DisplacementValuesComparison due to: ", msg)

class ZPlaneEstimation_with_best_focus_points():
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
        ##db path
        folderPath = folder + "/" + slide_name +"/data/"
        _dbPath = folderPath + '/' + slide_name + '.db'
        if not os.path.exists(_dbPath):
            _dbPath = folder + "/" + slide_name +'/' + slide_name + '.db'
            folderPath = folder + "/" + slide_name
        #print(_dbPath)
        self._folderPath = folderPath
        self._dbPath = _dbPath
        self.__path___ = path_to_save + "/" + "all_bestZ_plus_4linesCompleteSlide_pts" + "/" +slide_name
        if not os.path.exists(self.__path___):
            os.makedirs(self.__path___)
        gridId = 1
        dump_images = 0
        self._dump_images = 0
        self.num_grids_slide = 0
        self.connection = sqlite3.connect(self._dbPath )
        self.cursor = self.connection.cursor()
        connection = sqlite3.connect(_dbPath)
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
        # ax.scatter3D(x, y, z, color='b')
        # plt.show()
        
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
        plt.clf()

            
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

        return(perc_in_range, act_max_z_diff, dis_away_mm , pln_max_z_diff, pln_dis_away_mm, z_diff_extreme1, z_diff_extreme2, dis_away_extreme_mm, num_minus2_steps, num_minus1_steps, num_0_steps, num_1_step, num_2steps, angle_xy_plane,angle_yz_plane, angle_zx_plane)     
          
    def plane_correlation_with_new_points(self,fit, x,y,z,name,slide_name, NewPtsAnalysis,allpointAnalysis,count,new_points_metric):
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
        errors = A * fit - b
        plt.hist(errors)
        plt.xlabel("Z diff = plane_z-best_z")
        plt.ylabel("frequency")
        if(name == "new_points"):
            complete_path =self.__path___ + '/hist_OfAllPts_newPtsPlane' + '.png'
        elif(name == "all_points"):
            complete_path =self.__path___ + '/hist_OfAllPts_AllPtsPlane' + '.png'
        elif(name == "new_points_metric"):
            complete_path =self.__path___ + '/hist_newPts_newPtsPlane' + '.png'
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
        for _num in range(0,len(list_to_store)):
            if(name == "new_points"):
                NewPtsAnalysis.write(count+1, _num, list_to_store[_num])
            elif(name == "all_points"):
                allpointAnalysis.write(count+1, _num, list_to_store[_num])
            elif(name == "new_points_metric"):
                new_points_metric.write(count+1, _num, list_to_store[_num])
            
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
            
    def get_plane_estimated_z_value_focus_sampling(self,fit):
        x = self.focus_new_aoiRowIdx
        y = self.focus_new_aoiColIdx
        z = self.focus_best_z
        self.z_step_size = 1.875
        num_points = len(x)
        
        ## ax+by+c = z, Ax=B
        tmp_A = []
        tmp_b = []
        for i in range(len(x)):
            tmp_A.append([x[i], y[i], 1])
            tmp_b.append(z[i])

        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)
        #print("writting self._plane_estimated_z_value_focus_sampling")
        b_new = A * fit
        errors = b_new - b
        #print("b_new.size()", b_new.size, "size of aoiname: ", len(self.focus_aoiName))
        self._plane_estimated_z_value_focus_sampling = []
        self.z_diff_plane_estimation_focus_sampling = []
        self.z_step_plane_estimation_focus_sampling = []
        for i in range(0,len(self.focus_aoiName)):
            if (self.focus_best_z[i] != -1):
                z_diff = b_new[i,0]-b[i,0]
                z_step = int(z_diff/self.z_step_size)
                #z_index = z_step+int(self.stack_size/2)
                self._plane_estimated_z_value_focus_sampling.append(b_new[i,0])
                self.z_diff_plane_estimation_focus_sampling.append(z_diff)
                self.z_step_plane_estimation_focus_sampling.append(z_step)
            else:
                self._plane_estimated_z_value_focus_sampling.append(-1)
                self.z_diff_plane_estimation_focus_sampling.append(0)
                self.z_step_plane_estimation_focus_sampling.append(0)
                
        ####################################################################
        #focus sampling best index image
        self.generate_plane_estimation_best_index_heat_map(self.focus_aoiName, self.z_diff_plane_estimation_focus_sampling, self.focus_new_aoiRowIdx, self.focus_new_aoiColIdx , 11 ,"focus",self.focus_best_z)
                
    def get_plane_estimated_z_value(self,fit, new_aoiRowIdx, new_aoiColIdx, best_z, aoiName ,aoiBestIdx):
        x = self.new_aoiRowIdx
        y = self.new_aoiColIdx
        z = self.best_z
        num_points = len(x)
        self.z_step_size = 1.875

        ## ax+by+c = z, Ax=B
        tmp_A = []
        tmp_b = []
        for i in range(len(x)):
            tmp_A.append([x[i], y[i], 1])
            tmp_b.append(z[i])

        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)
        #print("writting self._plane_estimated_z_value")
        b_new = A * fit
        errors = b_new - b
        #print("b_new.size()", b_new.size, "size of aoiname: ", len(self.aoiName))
        self._plane_estimated_z_value = []
        self.z_diff_plane_estimation = []
        self.best_index_plane_estimation = []
        self.z_step_plane_estimation = []
        for i in range(0,len(self.aoiName)):
            if (self.aoiBestIdx[i] != -1):
                self._plane_estimated_z_value.append(b_new[i,0])
                z_diff = b_new[i,0]-b[i,0]
                self.z_diff_plane_estimation.append(z_diff)
                z_step = int(z_diff/self.z_step_size)
                z_index = z_step+int(self.stack_size/2)
                self.z_step_plane_estimation.append(z_step)
                self.best_index_plane_estimation.append(z_index)
            else:
                self._plane_estimated_z_value.append(-1)
                self.z_diff_plane_estimation.append(0)
                self.z_step_plane_estimation.append(0)
                self.best_index_plane_estimation.append(-1)
                
        ####################################################################
        #Accusition sampling best index image
        self.generate_plane_estimation_best_index_heat_map(self.aoiName, self.z_diff_plane_estimation, self.new_aoiRowIdx, self.new_aoiColIdx , 11 ,"acquisition", self.aoiBestIdx)
        
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
        self.generate_plane_estimation_best_index_heat_map(aoiName, z_diff_plane_estimation, new_aoiRowIdx, new_aoiColIdx , 11 ,"forg_acquisition", aoiBestIdx)

    def get_inline_plane_estimated_z_value_focus_sampling(self,fit,z_plane_points):
        #print("$"*50)
        self.z_step_size = 1.875
        self.fit = fit
        x = []; y = []; z=[]
        num_points = z_plane_points[0].size
        for i in range(0,num_points):
            x.append(z_plane_points[0][i])
            y.append(z_plane_points[1][i])
            z.append(z_plane_points[2][i])
        
        self.inline_plane_estimated_z_value_focus_sampling = []
        self.inline_z_diff_plane_estimation_focus_sampling = []
        self.inline_z_step_plane_estimation_focus_sampling = []
        for i in range(0,len(self.focus_aoiName)):
            #print(self.focus_aoiName[i], int(self.focus_new_aoiRowIdx[i]), int(self.focus_new_aoiColIdx[i]))
            if (self.focus_best_z[i] != -1):
                x_add = int(self.focus_new_aoiRowIdx[i])
                y_add = int(self.focus_new_aoiColIdx[i])
                z_add = self.focus_best_z[i]
                #print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
                z_new = (self.fit[0]*x_add) + (self.fit[1]*y_add) + self.fit[2]
                z_diff = z_new-z_add
                z_step = int(z_diff/self.z_step_size)
                #z_index = z_step+int(self.stack_size/2)
                self.inline_plane_estimated_z_value_focus_sampling.append(z_new)
                self.inline_z_diff_plane_estimation_focus_sampling.append(z_diff)
                self.inline_z_step_plane_estimation_focus_sampling.append(z_step)

                ##creating new plane
                new_point = True
                for j in range(0, len(x)):
                    if(x[j]==x_add and y[j]==y_add):
                        new_point = False
                        break
                if(new_point == True):
                    #print("*"*30, "creating new plan after adding new focus sampling points", "*"*30)
                    x.append(x_add); y.append(y_add); z.append(z_add)
                    x_plane = np.array(x); y_plane = np.array(y); z_plane = np.array(z)
                    self.fit = self.plane_fitting_max_point(x_plane,y_plane,z_plane)
            else:
                self.inline_plane_estimated_z_value_focus_sampling.append(-1)
                self.inline_z_diff_plane_estimation_focus_sampling.append(0)
                self.inline_z_step_plane_estimation_focus_sampling.append(0)

        ####################################################################
        self.inline_x = x; self.inline_y = y; self.inline_z = z
        #focus sampling best index image
        self.generate_plane_estimation_best_index_heat_map(self.focus_aoiName, self.inline_z_diff_plane_estimation_focus_sampling, self.focus_new_aoiRowIdx, self.focus_new_aoiColIdx , 11 ,"inline_focus",self.focus_best_z)

    def inline_get_plane_estimated_z_value(self):
        self.fit = self.fit
        x = self.inline_x; y = self.inline_y; z= self.inline_z
        
        self.inline_plane_estimated_z_value = []
        self.inline_z_diff_plane_estimation = []
        self.inline_best_index_plane_estimation = []
        self.inline_z_step_plane_estimation = []
        
        for i in range(0,len(self.aoiName)):
            if (self.aoiBestIdx[i] != -1):
                #print(self.aoiName[i])
                x_add = int(self.new_aoiRowIdx[i])
                y_add = int(self.new_aoiColIdx[i])
                z_add = self.best_z[i]
                #print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
                z_new = (self.fit[0]*x_add) + (self.fit[1]*y_add) + self.fit[2]
                z_diff = z_new-z_add
                z_step = int(z_diff/self.z_step_size)
                z_index = z_step+int(self.stack_size/2)
                self.inline_plane_estimated_z_value.append(z_new)
                self.inline_z_diff_plane_estimation.append(z_diff)
                self.inline_z_step_plane_estimation.append(z_step)
                self.inline_best_index_plane_estimation.append(z_index)
                
                ##creating new plane
                new_point = True
                for j in range(0, len(x)):
                    if(x[j]==x_add and y[j]==y_add):
                        new_point = False
                        break
                if(new_point == True):
                    #print("*"*30, "creating new plan after adding new points", "*"*30)
                    x.append(x_add); y.append(y_add); z.append(z_add)
                    x_plane = np.array(x); y_plane = np.array(y); z_plane = np.array(z)
                    self.fit = self.plane_fitting_max_point(x_plane,y_plane,z_plane)
            else:
                self.inline_plane_estimated_z_value.append(-1)
                self.inline_z_diff_plane_estimation.append(0)
                self.inline_z_step_plane_estimation.append(0)
                self.inline_best_index_plane_estimation.append(-1)
        
        #Accusition sampling best index image
        self.generate_plane_estimation_best_index_heat_map(self.aoiName, self.inline_z_diff_plane_estimation, self.new_aoiRowIdx, self.new_aoiColIdx , 11 ,"inline_acquisition", self.aoiBestIdx)
 
    def inline_step_analysis(self,slide_name, NewPtsAnalysis,allpointAnalysis,count):
        ## all focus sampling points
        z_step_size = self.z_step_size
        errors = np.array(self.inline_z_diff_plane_estimation_focus_sampling)
        absolute_sum_mean = np.sum(abs(errors))/len(errors)
        _max_step_size =  int(max(abs(errors))/z_step_size)
        _max_positive_step_size =  int(max(errors)/self.z_step_size)
        _max_negative_step_size  = int(min(errors)/self.z_step_size)
        if(_max_negative_step_size >0): _max_negative_step_size=0
        if(_max_positive_step_size <0): _max_positive_step_size=0
            
        NewPtsAnalysis.write(count+1, 29, slide_name)
        NewPtsAnalysis.write(count+1, 25, absolute_sum_mean)
        NewPtsAnalysis.write(count+1, 26, int(_max_step_size))
        NewPtsAnalysis.write(count+1, 27, int(_max_positive_step_size))
        NewPtsAnalysis.write(count+1, 28, int(_max_negative_step_size))

        ####all points
        errors = np.array(self.inline_z_diff_plane_estimation)
        absolute_sum_mean = np.sum(abs(errors))/len(errors)
        _max_step_size =  int(max(abs(errors))/z_step_size)
        _max_positive_step_size =  int(max(errors)/self.z_step_size)
        _max_negative_step_size  = int(min(errors)/self.z_step_size)
        if(_max_negative_step_size >0): _max_negative_step_size=0
        if(_max_positive_step_size <0): _max_positive_step_size=0
        
        allpointAnalysis.write(count+1, 29, slide_name)
        allpointAnalysis.write(count+1, 25, absolute_sum_mean)
        allpointAnalysis.write(count+1, 26, int(_max_step_size))
        allpointAnalysis.write(count+1, 27, int(_max_positive_step_size))
        allpointAnalysis.write(count+1, 28, int(_max_negative_step_size))
            
        
    def createExcel(self):
        # Create workbook
        path = self.__path___
        if not os.path.exists(path): os.mkdir(path)
        _folderPath = path
        #print(_folderPath + "/PLANE_centre_AcquisitionData.xlsx")
        workbook = xlsxwriter.Workbook(_folderPath + "/PLANE_centre_AcquisitionData.xlsx")
        
        # Define few formats
        # Add a format. Light red fill with dark red text.
        red = workbook.add_format({'bg_color': '#FFC7CE','font_color': '#9C0006'})
        # Add a format. Green fill with dark green text.
        green = workbook.add_format({'bg_color': '#C6EFCE','font_color': '#006100'})
        black_white = workbook.add_format({'bg_color': '#000000','font_color': '#FFFFFF'})
        white_black = workbook.add_format({'bg_color': '#FFFFFF','font_color': '#000000'})
        # Add Worksheets
        aoiName = workbook.add_worksheet("AOI Name")
        FocusSampling = workbook.add_worksheet("Focus Sampling")
        inlineFocusSampling = workbook.add_worksheet("inline Focus Sampling")
        inline_plane_estimation = workbook.add_worksheet("inline Plane Estimation")
        inline_zDiff = workbook.add_worksheet("inline Z Difference Plane")
        inline_z_step_plane_estimation = workbook.add_worksheet("inline Plane Estimation z step")
        plane_estimation = workbook.add_worksheet("Plane Estimation")
        zDiff = workbook.add_worksheet("Z Difference Plane")
        z_step_plane_estimation = workbook.add_worksheet("Plane Estimation z step")
        referenceZ = workbook.add_worksheet("Reference Z")
        bestIndex = workbook.add_worksheet("Best Index")
        bestZ = workbook.add_worksheet("Best Z")
        focusMetric = workbook.add_worksheet("Focus Metric")
        colorMetric = workbook.add_worksheet("Color Metric")

        # Add rows and columns into all sheets
        aoiName.write(0, 0, "row \ col")
        referenceZ.write(0, 0, "row \ col")
        bestIndex.write(0, 0, "row \ col")
        bestZ.write(0, 0, "row \ col")
        zDiff.write(0, 0, "row \ col")
        focusMetric.write(0, 0, "row \ col")
        colorMetric.write(0, 0, "row \ col")
        plane_estimation.write(0, 0, "row \ col")
        zDiff.write(0, 0, "row \ col")
        z_step_plane_estimation.write(0, 0, "row \ col")
        inline_plane_estimation.write(0, 0, "row \ col")
        inline_zDiff.write(0, 0, "row \ col")
        inline_z_step_plane_estimation.write(0, 0, "row \ col")

        for col in range(0, self.colCount):
            aoiName.write(0,col+1, col)
            referenceZ.write(0,col+1, col)
            bestIndex.write(0,col+1, col)
            bestZ.write(0,col+1, col)
            focusMetric.write(0,col+1, col)
            colorMetric.write(0,col+1, col)
            plane_estimation.write(0,col+1, col)
            z_step_plane_estimation.write(0,col+1, col)
            zDiff.write(0,col+1, col)
            inline_plane_estimation.write(0,col+1, col)
            inline_z_step_plane_estimation.write(0,col+1, col)
            inline_zDiff.write(0,col+1, col)
        for row in range(0, self.rowCount):
            aoiName.write(row+1,0, row)
            referenceZ.write(row+1,0, row)
            bestIndex.write(row+1,0, row)
            bestZ.write(row+1,0, row)
            focusMetric.write(row+1,0, row)
            colorMetric.write(row+1,0, row)
            plane_estimation.write(row+1,0, row)
            z_step_plane_estimation.write(row+1,0, row)
            zDiff.write(row+1,0, row)
            inline_plane_estimation.write(row+1,0, row)
            inline_z_step_plane_estimation.write(row+1,0, row)
            inline_zDiff.write(row+1,0, row)

                # Populate all data
        for i in range(0,len(self.aoiName)):
            if self.sampling_z[i] != -1:
                full_border = workbook.add_format({
                    "border": 10,
                    "border_color": "#000000"
                })
                aoiName.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.aoiName[i], full_border)
                referenceZ.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.ref_z[i],2), full_border)
                bestIndex.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.aoiBestIdx[i], full_border)
                bestZ.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.best_z[i],3), full_border)
                focusMetric.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.focusMetric[i],3), full_border)
                colorMetric.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.colorMetric[i],4), full_border)
                plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self._plane_estimated_z_value[i], full_border)
                z_step_plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.z_step_plane_estimation[i], full_border)
                zDiff.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.z_diff_plane_estimation[i], full_border)
                inline_plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.inline_plane_estimated_z_value[i], full_border)
                inline_z_step_plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.inline_z_step_plane_estimation[i], full_border)
                inline_zDiff.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.inline_z_diff_plane_estimation[i], full_border)
            else:
                full_border = workbook.add_format({
                    "border": 1,
                    "border_color": "#000000"
                })
                aoiName.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.aoiName[i], full_border)
                referenceZ.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.ref_z[i],2), full_border)
                bestIndex.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.aoiBestIdx[i], full_border)
                bestZ.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.best_z[i],3), full_border)
                focusMetric.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.focusMetric[i],3), full_border)
                colorMetric.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, round(self.colorMetric[i],4), full_border)
                plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self._plane_estimated_z_value[i], full_border)
                z_step_plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.z_step_plane_estimation[i], full_border)
                zDiff.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.z_diff_plane_estimation[i], full_border)
                inline_plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.inline_plane_estimated_z_value[i], full_border)
                inline_z_step_plane_estimation.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.inline_z_step_plane_estimation[i], full_border)
                inline_zDiff.write(self.aoiRowIdx[i]+1, self.aoiColIdx[i]+1, self.inline_z_diff_plane_estimation[i], full_border)

        # Conditional formating - generating heap maps
        letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        #print("Size of letter: ", len(letter))
        #print("Row count & col count: ", self.rowCount, self.colCount)
        factor = int(self.colCount / 26)
        if factor > 0:
            letter_str = "A"
            for index in range(0, factor):
                letter_str = letter_str + "A"
            end = letter_str+str(self.rowCount+1)
        else:
            end = letter[self.colCount]+str(self.rowCount+1)
        cells = 'B2:' + end
        #print("CELL RANGE:", cells)
        aoiName.conditional_format(cells, {'type': 'cell','criteria': '==','value': 1,'format': black_white})
        aoiName.conditional_format(cells, {'type': 'cell','criteria': '==','value': 0,'format': white_black})
        referenceZ.conditional_format(cells, {'type': '2_color_scale'})
        bestIndex.conditional_format(cells, {'type': '2_color_scale'})
        bestZ.conditional_format(cells, {'type': '2_color_scale'})
        focusMetric.conditional_format(cells, {'type': '2_color_scale','min_color': "#FF0000",'max_color': "#00FF00"})
        colorMetric.conditional_format(cells, {'type': 'cell','criteria': '<','value': self.cmThresh,'format': red})
        colorMetric.conditional_format(cells, {'type': 'cell','criteria': '>=','value': self.cmThresh,'format': green})
        
        ############### Focus Sampling #####################
        FocusSampling.write(0,0,"aoiName")
        FocusSampling.write(0,1,"aoiRowIdx")
        FocusSampling.write(0,2,"aoiColIdx") 
        FocusSampling.write(0,3,"focus_fm")
        FocusSampling.write(0,4,"focus_cm")
        FocusSampling.write(0,5,"best_z"); FocusSampling.set_column(0, 5, 15)
        FocusSampling.write(0,6,"plane focus_z"); FocusSampling.set_column(0, 6, 15)
        FocusSampling.write(0,7,"planez diff"); FocusSampling.set_column(0, 7, 15)
        FocusSampling.write(0,8,"planeZ step_size"); FocusSampling.set_column(0, 8, 15)
        for row in range(0,len(self.focus_aoiName)):
            FocusSampling.write(row+1,0,self.focus_aoiName[row],full_border)
            FocusSampling.write(row+1,1,self.focus_aoiRowIdx[row],full_border)
            FocusSampling.write(row+1,2,self.focus_aoiColIdx[row],full_border)
            FocusSampling.write(row+1,3,self.focus_fm[row],full_border)
            FocusSampling.write(row+1,4,self.focus_cm[row],full_border)
            FocusSampling.write(row+1,5,self.focus_best_z[row],full_border)
            FocusSampling.write(row+1,6,self._plane_estimated_z_value_focus_sampling[row],full_border)
            FocusSampling.write(row+1,7,self.z_diff_plane_estimation_focus_sampling[row],full_border)
            FocusSampling.write(row+1,8,self.z_step_plane_estimation_focus_sampling[row],full_border)
        
        ############### inline Focus Sampling ##################
        inlineFocusSampling.write(0,0,"aoiName")
        inlineFocusSampling.write(0,1,"aoiRowIdx")
        inlineFocusSampling.write(0,2,"aoiColIdx") 
        inlineFocusSampling.write(0,3,"focus_fm")
        inlineFocusSampling.write(0,4,"focus_cm")
        inlineFocusSampling.write(0,5,"best_z"); inlineFocusSampling.set_column(0, 5, 15)
        inlineFocusSampling.write(0,6,"plane focus_z"); inlineFocusSampling.set_column(0, 6, 15)
        inlineFocusSampling.write(0,7,"planez diff"); inlineFocusSampling.set_column(0, 7, 15)
        inlineFocusSampling.write(0,8,"planeZ step_size"); inlineFocusSampling.set_column(0, 8, 15)
        for row in range(0,len(self.focus_aoiName)):
            inlineFocusSampling.write(row+1,0,self.focus_aoiName[row],full_border)
            inlineFocusSampling.write(row+1,1,self.focus_aoiRowIdx[row],full_border)
            inlineFocusSampling.write(row+1,2,self.focus_aoiColIdx[row],full_border)
            inlineFocusSampling.write(row+1,3,self.focus_fm[row],full_border)
            inlineFocusSampling.write(row+1,4,self.focus_cm[row],full_border)
            inlineFocusSampling.write(row+1,5,self.focus_best_z[row],full_border)
            inlineFocusSampling.write(row+1,6,self.inline_plane_estimated_z_value_focus_sampling[row],full_border)
            inlineFocusSampling.write(row+1,7,self.inline_z_diff_plane_estimation_focus_sampling[row],full_border)
            inlineFocusSampling.write(row+1,8,self.inline_z_step_plane_estimation_focus_sampling[row],full_border)
        
        # Close workbook
        workbook.close()
        
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
        #Accusition sampling best index image
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
        return(fit)
        
    def points_on_image(self,x,y,z,all_3_points,name):
        image = self.image_all_points.copy()
        for i in range (0,len(x)):
            row = x[i]; column = y[i]
            image[int(row),int(column),:] = [255, 0,0]
        
        ##marking triangulation points from red
        x = all_3_points[0]
        y = all_3_points[1]
        
        for i in range (0,len(x)):
            row = x[i]; column = y[i]
            image[int(row),int(column),:] = [0, 255,0]
        
        path = self.__path___
        if not os.path.exists(path): os.mkdir(path)
        complete_path = path + "/" +name+'_pointsCentre_triangulation_points' + '.jpeg'
        #print(complete_path)
        self.resizeImage(image,complete_path)
        # plt.imshow(image); plt.show()
    
    def three_centroid_points(self):
        #print("_*_"*50)
        three_points = np.zeros((3,3))
        #print("three points: ", three_points)
        
        ## selecting biggest grid
        num_grids = self.grid_xy_index[0].size
        biggest_grid = -1
        largest_grid_area = -1
        for i in range(0,num_grids):
            rows = self.grid_xy_index[3][i]
            columns = self.grid_xy_index[4][i]
            grid_area = rows*columns
            if(grid_area > largest_grid_area):
                largest_grid_area = grid_area
                biggest_grid = i+1
        
        ##foreground data of the biggest grid
        rows = self.grid_xy_index[3][biggest_grid-1] ###number of rows
        columns = self.grid_xy_index[4][biggest_grid-1] ###number of columns
        gridId = int(biggest_grid)
        #print("Biggest gridId", gridId, "rows: ", rows , "columns: ", columns)
        all_grid_aoi = np.zeros((int(rows),int(columns)))
        ## all foreground x and y coordinates
        x_forg = []; y_forg = []
        
        self.cursor.execute("SELECT aoi_name, aoi_row_idx, aoi_col_idx, sampling_z, bg_state_fs, bg_state_acq, focus_metric, color_metric, best_idx, best_z, ref_z, z_value FROM aoi WHERE grid_id = '%s'" % gridId)
        for aoiInfo in self.cursor.fetchall():
            all_grid_aoi[aoiInfo[1]][aoiInfo[2]] = -1
            if(aoiInfo[8]!= -1 and aoiInfo[6]>6.0 and aoiInfo[7]>40.0):
                all_grid_aoi[aoiInfo[1]][aoiInfo[2]] = aoiInfo[9]
                x_forg.append(aoiInfo[1]); y_forg.append(aoiInfo[2])
        
        ###################################   Starting triangulation #######################################
        status = True
        
        n_max = 3 ##length to width ratio
        n_min = 1
        l_min = 10
        l_max = 170
        b_max = 50
        r_max = 25
        r_min = 3
        p_max = 30
        p_min = 4
        Q_min = 0
        Q_max = 120
        minimum_area = l_min * l_min
        maximum_area = l_max * b_max

        ##using grid centroid
        ## x in rows and y in columns
        x_grid_start = 0 ##rows
        y_grid_start = 0 ##columns
        x_grid_end = x_grid_start+rows
        y_grid_end = y_grid_start +columns
        Xc = -1
        Yc = -1
        ### using points centre
        #print("using point centre /n")
        Xc_points = int(np.sum(x_forg)/len(x_forg))
        Yc_points = int(np.sum(y_forg)/len(y_forg))
        #print("Xc_points: ", Xc_points, "Yc_points: ", Yc_points)
        # Xc = Xc_points
        # Yc = Yc_points
        # using grid centre
        if(Xc == -1 or Yc == -1):
            Xc = x_grid_start + (rows)/2
            Yc = y_grid_start + (columns)/2
        new_rows = rows
        new_col = columns
        ##taking the smaller rectangle
        if( (Xc - x_grid_start) <= (x_grid_start + rows - Xc) ):
            new_rows = 2*(Xc - x_grid_start)
        else:
            new_rows = 2*(x_grid_start + rows - Xc)
        if( (Yc - y_grid_start) <= (y_grid_start + columns - Yc)):
            new_col = 2*(Yc - y_grid_start)
        else:
            new_col = 2*(y_grid_start + columns - Yc)
        

        x_start = max(int(Xc - (new_rows/2)), x_grid_start)
        y_start = max(int(Yc - (new_col/2)),x_grid_start)
        x_end = min(int(x_start + new_rows), x_grid_end)
        y_end = min(int(y_start + new_col), y_grid_end)
        
        if((x_start >=  x_end) or (y_start >= y_end)):
            #print("initial grid is not correct")
            status = False
            return(status,three_points)
            
        
        #print("x_start", x_start,"rows: ", new_rows, "x_end: ", x_end, "\n", "y_start: ",y_start , "cols: ", new_col, "y_end: ", y_end)
        
        l = max(x_end-x_start,y_end-y_start)
        b = min(x_end-x_start,y_end - y_start)
        n = l/b
        p_max = min(p_max, ((3*numpy.sqrt(3))/(16*n))*100)
        p_start = 28
        p = min(28,p_max) ##percentage of triangle to rectangle
        Q = 45 ##angle to rotate triangle
        

        
        local_a_max = l*(numpy.sqrt(p_max/(n*25*numpy.sqrt(3))))
        local_r_max = local_a_max/numpy.sqrt(3)
        local_a_min = l*(numpy.sqrt(p_min/(n*25*numpy.sqrt(3))))
        local_r_min = local_a_min/numpy.sqrt(3)
        
        a = l*(numpy.sqrt(p/(n*25*numpy.sqrt(3))))
        r = a/numpy.sqrt(3)
        
        a = min(a,local_a_max)
        r = min(r, local_r_max)
        
        if(r > 10):
            p = min(20,p_max)
            
        if(n > n_max or n < n_min or largest_grid_area<minimum_area or largest_grid_area>maximum_area or r < r_min or r > r_max):
            #print("triangle not possible")
            status = False
            return(status,three_points)
    
        if(b < l_min):
            #print("smalled length of grid is less than 10, triangle not possible")
            status = False
            return(status,three_points)
        
        ### taking diameter = 0.8*b
        if(2*r >= b):
            r = 0.8*(b/2)
        
        ## initializing all points
        Xpa = Xpb = Xpc = Ypa = Ypb = Ypc = -10
        p_old = p
        
        points_outside = True
        #print("p: ", p , "p_max: ", p_max, "p_min: ", p_min)
        while(points_outside == True and p <= p_max and p >= p_min ):
            #print(p)
            a = l*(numpy.sqrt(p/(n*25*numpy.sqrt(3))))
            r = a/numpy.sqrt(3)
#             Xpa = int(Xc+ r * numpy.cos(numpy.pi * (270-Q)/180)); Ypa = int(Yc+r*numpy.sin(numpy.pi*(270-Q)/180))
#             Xpb = int(Xc+ r * numpy.cos(numpy.pi * (30-Q)/180));  Ypb = int(Yc+r*numpy.sin(numpy.pi*(30-Q)/180))
#             Xpc = int(Xc+ r * numpy.cos(numpy.pi * (150-Q)/180)); Ypc = int(Yc+r*numpy.sin(numpy.pi*(150-Q)/180))
            ##as we have x in downward direction and y is in horizontal diorection
            Xpa = int(Xc+ r * numpy.sin(numpy.pi * (270-Q)/180)); Ypa = int(Yc+r*numpy.cos(numpy.pi*(270-Q)/180))
            Xpb = int(Xc+ r * numpy.sin(numpy.pi * (30-Q)/180));  Ypb = int(Yc+r*numpy.cos(numpy.pi*(30-Q)/180))
            Xpc = int(Xc+ r * numpy.sin(numpy.pi * (150-Q)/180)); Ypc = int(Yc+r*numpy.cos(numpy.pi*(150-Q)/180))
            if(Xpa > x_end or Ypa > y_end or Xpa < x_start or Ypa < y_start):
                points_outside = True
            elif(Xpb > x_end or Ypb > y_end or Xpb < x_start or Ypb < y_start):
                points_outside = True
            elif(Xpc > x_end or Ypc > y_end or Xpc < x_start or Ypc < y_start):
                points_outside = True
            else:
                points_outside = False
            
            ##decreasing the area percentage
            p_old = p
            p = 0.8*(p)
           
        if(points_outside == True):
            #print("Points outside triangle not possible")
            status = False
            return(status,three_points)
        
        p = p_old
        r_old = r
        Q_list = [45,40,25,20,15,10,5,35,50,55,60,65,70,75,80,85,95,100,105,110,115,90,0,30,120]
        while((all_grid_aoi[Xpa][Ypa] == -1 or all_grid_aoi[Xpb][Ypb] == -1 or all_grid_aoi[Xpc][Ypc] == -1) and r > b/8 and Q > 0 and r < r_max and r > r_min):   
            for Q_new in range(0,len(Q_list)):
                #print("radius: ", r ,"new Angle: ", Q_new)
                #Q = Q_new
                Q = Q_list[Q_new]
                # Xpa = int(Xc+ r * numpy.cos(numpy.pi * (270-Q)/180)); Ypa = int(Yc+r*numpy.sin(numpy.pi*(270-Q)/180))
                # Xpb = int(Xc+ r * numpy.cos(numpy.pi * (30-Q)/180));  Ypb = int(Yc+r*numpy.sin(numpy.pi*(30-Q)/180))
                # Xpc = int(Xc+ r * numpy.cos(numpy.pi * (150-Q)/180)); Ypc = int(Yc+r*numpy.sin(numpy.pi*(150-Q)/180))
                ##as we have x in downward direction and y is in horizontal diorection
                Xpa = int(Xc+ r * numpy.sin(numpy.pi * (270-Q)/180)); Ypa = int(Yc+r*numpy.cos(numpy.pi*(270-Q)/180))
                Xpb = int(Xc+ r * numpy.sin(numpy.pi * (30-Q)/180));  Ypb = int(Yc+r*numpy.cos(numpy.pi*(30-Q)/180))
                Xpc = int(Xc+ r * numpy.sin(numpy.pi * (150-Q)/180)); Ypc = int(Yc+r*numpy.cos(numpy.pi*(150-Q)/180))
            #print("raduis: ", r, "new_radius: ", 3*r/4)
            r_old = r; r = 3*r_old/4
            # print("raduis: ", r, "new_radius: ", r-1)
            # r_old = r; r = r_old-1
        
        r = r_old
        if (all_grid_aoi[Xpa][Ypa] == -1 or all_grid_aoi[Xpb][Ypb] == -1 or all_grid_aoi[Xpc][Ypc] == -1):
            #print("points are on background")
            status =False;
            return(status,three_points)
        #print("checking if points are outside the grid")
        if(Xpa > x_end or Ypa > y_end or Xpa < x_start or Ypa < y_start):
             status =False; return(status,three_points)
        elif(Xpb > x_end or Ypb > y_end or Xpb < x_start or Ypb < y_start):
            status =False; return(status,three_points)
        elif(Xpc > x_end or Ypc > y_end or Xpc < x_start or Ypc < y_start):
            status =False; return(status,three_points)
                    
        new_Xpa = Xpa + self.grid_xy_index[1][biggest_grid-1]
        new_Ypa = Ypa + self.grid_xy_index[2][biggest_grid-1]
        new_Xpb = Xpb + self.grid_xy_index[1][biggest_grid-1]
        new_Ypb = Ypb + self.grid_xy_index[2][biggest_grid-1]
        new_Xpc = Xpc + self.grid_xy_index[1][biggest_grid-1]
        new_Ypc = Ypc + self.grid_xy_index[2][biggest_grid-1]
        Zpa = all_grid_aoi[Xpa][Ypa]
        Zpb = all_grid_aoi[Xpb][Ypb]
        Zpc = all_grid_aoi[Xpc][Ypc]
        three_points[0][0] = new_Xpa
        three_points[1][0] = new_Ypa
        three_points[2][0] = Zpa  
        three_points[0][1] = new_Xpb
        three_points[1][1] = new_Ypb
        three_points[2][1] = Zpb          
        three_points[0][2] = new_Xpc
        three_points[1][2] = new_Ypc
        three_points[2][2] = Zpc   
        
        self.triangulation_points = three_points
        #print("three points: \n", three_points)
        #return(status,three_points)
        
        ########### picking nearby point which surrounded by foreground ##########
        old_three_points = np.zeros((3,3))
        old_three_points[0][0] = Xpa; old_three_points[1][0] = Ypa
        old_three_points[2][0] = Zpa  
        old_three_points[0][1] = Xpb; old_three_points[1][1] = Ypb
        old_three_points[2][1] = Zpb          
        old_three_points[0][2] = Xpc ; old_three_points[1][2] = Ypc
        old_three_points[2][2] = Zpc   
        new_three_points = self.point_surrounding_foreground(all_grid_aoi, old_three_points)
        new_three_points[0][0] = new_three_points[0][0] + self.grid_xy_index[1][biggest_grid-1]
        new_three_points[1][0] = new_three_points[1][0] + self.grid_xy_index[2][biggest_grid-1]
        new_three_points[2][0] = new_three_points[2][0]
        new_three_points[0][1] = new_three_points[0][1] + self.grid_xy_index[1][biggest_grid-1]
        new_three_points[1][1] = new_three_points[1][1] + self.grid_xy_index[2][biggest_grid-1]
        new_three_points[2][1] = new_three_points[2][1]        
        new_three_points[0][2] = new_three_points[0][2] + self.grid_xy_index[1][biggest_grid-1]
        new_three_points[1][2] = new_three_points[1][2] + self.grid_xy_index[2][biggest_grid-1]
        new_three_points[2][2] = new_three_points[2][2]
        #print("old three points: \n", three_points)
        #print("new three points: \n", new_three_points)
        return(status,new_three_points)
    
    def point_surrounding_foreground(self,all_grid_aoi, three_points):
        #print("searching AOI surrounded by foreground")
        num_points = three_points[0].shape
        grid_start_r = 0; grid_start_c = 0
        rows =    all_grid_aoi.shape[0]
        columns = all_grid_aoi.shape[1]
        grid_end_r = grid_start_r+rows; grid_end_c = grid_start_c + columns
        #print("grid rows: ", rows, "grid columns: ", columns)
        new_points = three_points.copy()
        x = three_points[0]
        y = three_points[1]
        
        #Gmax_num_sur = (3*3) -1
        Gmax_num_sur = (5*5) -1
        for i in range (0,len(x)):
            _row = int(three_points[0][i])
            _column = int(three_points[1][i])
            _new_row = -1; _new_col = -1; _new_z = -1
            if(_row < grid_end_r and _column < grid_end_c and all_grid_aoi[_row][_column]!=-1):
                #print("_rows: ", _row, "_columns: ", _column, "z: ", all_grid_aoi[_row][_column])
                ## start searching for this points
                map = np.full((all_grid_aoi.shape[0], all_grid_aoi.shape[1]),-1)
                q_row = []; q_col = []
                q_row.append(_row); q_col.append(_column)
                map[_row][_column] =1
                max_num_foreground = -1
                _new_row = -1; _new_col = -1; _new_z = -1
                
                max_calls = 100
                calls = 0
                while(len(q_row)!=0 and calls<max_calls):
                    calls +=1
                    _cur_row = q_row.pop(0)
                    _cur_col = q_col.pop(0)
                    #print("_cur_row: ", _cur_row, "_cur_col: ", _cur_col, "z: ", all_grid_aoi[_cur_row][_cur_col])
                    num_foregrounds = 0
                    for r in [-2,-1,0,1,2]:
                        for c in [-2,-1,0,1,2]:
                            if(r == 0 and c == 0):
                                continue
                            if( (_cur_row+r) >= grid_start_r and (_cur_row+r) < grid_end_r):
                                if((_cur_col+c) >= grid_start_c and (_cur_col+c) < grid_end_c):
                                    if(all_grid_aoi[_cur_row+r][_cur_col+c]!=-1):
                                        #print("new row: ", _cur_row+r, "new_col: ", _cur_col+c, "z: ",all_grid_aoi[_cur_row+r][_cur_col+c])
                                        num_foregrounds +=1
                                    if(map[_cur_row+r][_cur_col+c] == -1):
                                        map[_cur_row+r][_cur_col+c] = 1
                                        q_row.append(_cur_row+r)
                                        q_col.append(_cur_col+c)
                    
                    #print("num_foreground: ", num_foregrounds)
                    _new_z_ = all_grid_aoi[_cur_row][_cur_col]
                    if(num_foregrounds == Gmax_num_sur and _new_z_!=-1):
                        _new_row = _cur_row
                        _new_col = _cur_col
                        _new_z = all_grid_aoi[_new_row][_new_col]
                        break
                    if(num_foregrounds > max_num_foreground and _new_z_!=-1):
                        max_num_foreground = num_foregrounds
                        _new_row = _cur_row
                        _new_col = _cur_col
                        _new_z = all_grid_aoi[_new_row][_new_col]
                
                if(_new_row != -1 and _new_col != -1 and _new_z!=-1):
                    new_points[0][i] = _new_row
                    new_points[1][i] = _new_col
                    new_points[2][i] = _new_z
                    #print("new_rows: ", _new_row, "new_columns: ",_new_col, "new_z: ", _new_z,"\n")
            
        return(new_points) 
    
    def calculate_horizontal_points(self,row_start, row_end,y_start, y_end, all_grid_aoi):
        _corner_points = np.full((3,2),-1)
        point_num = 0
        max_dis_points = -1
        x1 = -1; y1= -1; z1 =-1
        x2 = -1; y2= -1; z2 =-1
        for h in range(row_start, row_end):
            dis = 0
            _x = -1; _y = -1; _z = -1
            for c in range(y_start, y_end):
                if(all_grid_aoi[int(h)][int(c)] !=-1):
                    _x = int(h); _y = int(c); _z = all_grid_aoi[_x][_y]
                    break
            _ex = -1; _ey = -1; _ez = -1
            for c in range(y_end-1, y_start-1,-1):
                if(all_grid_aoi[int(h)][int(c)] !=-1):
                    _ex = int(h); _ey = int(c); _ez = all_grid_aoi[_ex][_ey]
                    break
            dis = _ey - _y
            if(_ey!= -1 and _y!=-1 and dis > max_dis_points):
                max_dis_points = dis
                x1 = _x; y1= _y; z1 = _z
                x2 = _ex; y2= _ey; z2 = _ez
             
        _corner_points[0][point_num] = x1 
        _corner_points[1][point_num] = y1 
        _corner_points[2][point_num] = z1
        _corner_points[0][point_num+1] = x2 
        _corner_points[1][point_num+1] = y2
        _corner_points[2][point_num+1] = z2
        return(_corner_points)
    
    def calculate_vertical_points(self,col_start, col_end,x_start, x_end, all_grid_aoi):
        _corner_points = np.full((3,2),-1)
        point_num = 0
        max_dis_points = -1
        x1 = -1; y1= -1; z1 =-1
        x2 = -1; y2= -1; z2 =-1
        for c in range(col_start, col_end):
            dis = 0
            _x = -1; _y = -1; _z = -1
            for h in range(x_start, x_end):
                if(all_grid_aoi[int(h)][int(c)] !=-1):
                    _x = int(h); _y = int(c); _z = all_grid_aoi[_x][_y]
                    break
            _ex = -1; _ey = -1; _ez = -1
            for h in range(x_end-1, x_start-1,-1):
                if(all_grid_aoi[int(h)][int(c)] !=-1):
                    _ex = int(h); _ey = int(c); _ez = all_grid_aoi[_ex][_ey]
                    break
            dis = _ex - _x
            if(_ey!= -1 and _y!=-1 and dis > max_dis_points):
                max_dis_points = dis
                x1 = _x; y1= _y; z1 = _z
                x2 = _ex; y2= _ey; z2 = _ez
                
        _corner_points[0][point_num] = x1 
        _corner_points[1][point_num] = y1 
        _corner_points[2][point_num] = z1
        _corner_points[0][point_num+1] = x2 
        _corner_points[1][point_num+1] = y2
        _corner_points[2][point_num+1] = z2
        return(_corner_points)
    
    def four_lines_points(self):
        #print("_*_"*50)
        corner_points = np.full((3,8),-1)
        #print("corner_points: ", corner_points)
   
        rows = self.total_rows_slide
        columns= self.total_columns_slide
        all_grid_aoi = np.full((int(rows),int(columns)),-1)
        x_forg = []; y_forg = []       
        for num in range(0,len(self.aoiName)):
            all_grid_aoi[self.new_aoiRowIdx[num]][self.new_aoiColIdx[num]] = -1
            if(self.aoiBestIdx[num]!=-1 and self.focusMetric[num]>6.0 and self.colorMetric[num]>40):
                all_grid_aoi[self.new_aoiRowIdx[num]][self.new_aoiColIdx[num]] = self.best_z[num]
                x_forg.append(self.new_aoiRowIdx[num]); y_forg.append(self.new_aoiColIdx[num])
        ###################################   Starting triangulation #######################################
        status = True
        
        n_max = 3 ##length to width ratio
        n_min = 1
        l_min = 10
        b_min = 10
        l_max = 170
        b_max = 50
        minimum_area = l_min * b_min
        maximum_area = l_max * b_max

        ##using grid centroid
        ## x in rows and y in columns
        #print("\ntotal rows: ", rows, "total cols: ", columns,"\n")
        x_grid_start = 0 ##rows
        y_grid_start = 0 ##columns
        x_grid_end = x_grid_start+rows
        y_grid_end = y_grid_start +columns
        Xc = -1
        Yc = -1
        ### using points centre
        #print("using point centre /n")
        Xc_points = int(np.sum(x_forg)/len(x_forg))
        Yc_points = int(np.sum(y_forg)/len(y_forg))
        #print("Xc_points: ", Xc_points, "Yc_points: ", Yc_points)
        Xc = Xc_points
        Yc = Yc_points
        # using grid centre
        if(Xc == -1 or Yc == -1):
            Xc = x_grid_start + (rows)/2
            Yc = y_grid_start + (columns)/2
        new_rows = rows
        new_col = columns
        ##taking the smaller rectangle
        if( (Xc - x_grid_start) <= (x_grid_start + rows - Xc) ):
            new_rows = 2*(Xc - x_grid_start)
        else:
            new_rows = 2*(x_grid_start + rows - Xc)
        if( (Yc - y_grid_start) <= (y_grid_start + columns - Yc)):
            new_col = 2*(Yc - y_grid_start)
        else:
            new_col = 2*(y_grid_start + columns - Yc)
        
        x_start = max(int(Xc - (new_rows/2)), x_grid_start)
        y_start = max(int(Yc - (new_col/2)),x_grid_start)
        x_end = min(int(x_start + new_rows), x_grid_end)
        y_end = min(int(y_start + new_col), y_grid_end)
        
        if((x_start >=  x_end) or (y_start >= y_end)):
            #print("initial grid is not correct")
            status = False
            return(status,corner_points)
            
        #print("x_start", x_start,"rows: ", new_rows, "x_end: ", x_end, "\n", "y_start: ",y_start , "cols: ", new_col, "y_end: ", y_end)
        
        l = max(x_end-x_start,y_end-y_start)
        b = min(x_end-x_start,y_end - y_start)
        n = l/b
        
        height = x_end - x_start
        length = y_end - y_start
        ratio_hl = height/length
        threshold_hl = 3
        #threshold_hl = 1 ## experimenting with squarish grids
        #print("\nheight: ", height, "length: ", length, "ratio_hl: ", ratio_hl,"\n")
        point_num = 0
        start_list_horizontal = [x_start,  x_start + (3*height)/4]
        end_list_horizontal = [x_start+(height)/4, x_end]
        start_list_vertical = [y_start,  y_start + (3*length)/4]
        end_list_vertical = [y_start+(length)/4,  y_end]
        
        if(height > length and (height/length) >= threshold_hl):
            start_list_horizontal = [x_start,  x_start + (5*height)/8]
            end_list_horizontal = [x_start+(3*height)/8, x_end]
            start_list_vertical = [y_start]
            end_list_vertical = [y_end]
        elif(length > height and (length/height) >= threshold_hl):
            start_list_horizontal = [x_start]
            end_list_horizontal = [x_end]
            start_list_vertical = [y_start,  y_start + (5*length)/8]
            end_list_vertical = [y_start+(3*length)/4,  y_end]
            
        ############## horizontal line - finding horizontal points ############
        start_list = start_list_horizontal
        end_list = end_list_horizontal 
        for point in range(0,len(start_list)):
            row_start = int(start_list[point])
            row_end = int(end_list[point])
            new_corner_points =  self.calculate_horizontal_points(row_start, row_end,y_start, y_end, all_grid_aoi)
            corner_points[:,point_num:point_num+2] =  new_corner_points
            point_num +=2
        
        ############### vertical lines --> finding vertical points #############
        start_list = start_list_vertical
        end_list = end_list_vertical
        for point in range(0,len(start_list)):
            col_start = int(start_list[point])
            col_end = int(end_list[point])
            new_corner_points =  self.calculate_vertical_points(col_start, col_end,x_start, x_end, all_grid_aoi)
            corner_points[:,point_num:point_num+2] =  new_corner_points
            point_num +=2
        
        ### we got our points, we need to find nearest point surrounding 
        corner_points = corner_points[:,:point_num]
        #print("corner points: \n", corner_points)
        corner_points= self.point_surrounding_foreground(all_grid_aoi, corner_points)
        
        #return(status, corner_points)
        #-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-
        
        ####plotting points on image 
        image = self.image_all_points.copy()   
        x = corner_points[0]
        y = corner_points[1]
        z = corner_points[2]
        for i in range (0,4):
            row = x[i]; column = y[i]
            image[int(row),int(column),:] = [0, 255, 0]
        for i in range (4,len(x)):
            row = x[i]; column = y[i]
            image[int(row),int(column),:] = [0, 0, 255]
        
        path = self.__path___
        if not os.path.exists(path): os.mkdir(path)
        complete_path = path + "/" +'8pointsCentre_triangulation_points' + '.jpeg'
        #print(complete_path)
        self.resizeImage(image,complete_path)
        # plt.imshow(image); plt.show()
        #self.points_on_image(x,y,z,corner_points)
        #print("final corner points: \n", corner_points)
        return(status, corner_points)
                
    def point_filtering(self, corner_points):
        if(corner_points.shape[1] == 0):
            #print("input array is empty in point_filtering on points")
            return(corner_points)
        
        x_new = []; y_new = []; z_new = []
        map = np.full((corner_points.shape[1]),-1)
        row_diff = 10; col_diff = 5
        x_new.append(corner_points[0][0])
        y_new.append(corner_points[1][0])
        z_new.append(corner_points[2][0])
        map[0] = 1
        for i in range(1,corner_points.shape[1]):
            if(corner_points[2][i] ==-1 or map[i]==1):
                map[i] = 1
                continue
            
            point_away  = True
            x = corner_points[0][i]
            y = corner_points[1][i]
            z = corner_points[2][i]
            for j in range(0,len(x_new)):
                if(abs(x_new[j]-x)<=row_diff and abs(y_new[j]-y)<=col_diff):
                    point_away = False
            
            if(corner_points[2][i] !=-1 and  point_away==True):
                map[i]=1
                x_new.append(x)
                y_new.append(y)
                z_new.append(z)
               
        new_all_points = np.zeros((3,len(x_new)))
        new_all_points[0] = x_new
        new_all_points[1] = y_new
        new_all_points[2] = z_new
        
        return(new_all_points)
    
    def point_filtering_bestZ(self, corner_points, bestZ_points):
        if(corner_points.shape[1] == 0):
            #print("input array is empty in point_filtering_bestZ")
            return(corner_points)
        
        x_new = []; y_new = []; z_new = []
        row_diff = 10; col_diff = 5
        for j in range(0, corner_points.shape[1]):
            point_away  = True
            for i in range(0, bestZ_points.shape[1]):
                if(corner_points[2][j] == -1):
                    point_away = False
                    break
                
                x = corner_points[0][j]
                y = corner_points[1][j]
                z = corner_points[2][j]
                if(abs(bestZ_points[0][i]-x)<=row_diff and abs(bestZ_points[1][i]-y)<=col_diff):
                    point_away = False
                    break
                    
            if(point_away==True):
                x_new.append(x)
                y_new.append(y)
                z_new.append(z)
                
        new_all_points = np.zeros((3,len(x_new)))
        new_all_points[0] = x_new
        new_all_points[1] = y_new
        new_all_points[2] = z_new
        return(new_all_points)
    
    def store_new_points(self,x,y,z, pointStore, count,slide_name):
        '''
        !function to store points
        @param x x coordinates of the point
        @param y y coordinates of the point
        @param z z coordinate of the point
        @param pointStore excel sheet link
        @return store points in excel sheet
        '''
        #print("inside store new points")
        total_points = len(x)
        pointStore.write(count+1, 0, slide_name)
        pointStore.write(count+1, 1, total_points)
        point_num = 2
        for i in range(0,total_points):
            pointStore.write(count+1, point_num, x[i])
            pointStore.write(count+1, point_num+1, y[i])
            pointStore.write(count+1, point_num+2, z[i])
            point_num +=3
        return

def center_points_process(path, slide_name, NewPtsAnalysis,allpointAnalysis,count,pointStore,new_points_metric,path_to_save):
    try:
        print("\n\n","*"*50,slide_name, "*"*50, "\n","*"*110)
        three_points_obj = ZPlaneEstimation_with_best_focus_points()
        cursor,rowCount, colCount, aoiRowIdx, aoiColIdx, best_z , ref_z , zdiff_best_ref, focus_aoiName, focus_aoiRowIdx, focus_aoiColIdx, focus_best_z, focus_fm,focus_cm, new_aoiRowIdx ,new_aoiColIdx,  focus_new_aoiRowIdx, focus_new_aoiColIdx, bestZ_grid, aoiName ,aoiBestIdx  = three_points_obj.getAoidetails(path,slide_name,path_to_save)
        three_points_obj.plot_points(focus_new_aoiColIdx, focus_new_aoiRowIdx, focus_best_z,  new_aoiColIdx, new_aoiRowIdx, best_z)
        ###creating plane from all points
        allPts_fit = three_points_obj.plot_from_all_points_plane(new_aoiRowIdx ,new_aoiColIdx,best_z,aoiName ,aoiBestIdx)
        #print("\n","#"*10, "Fitting all points on Plane using all points", "#"*10, "\n")
        y = np.array(new_aoiColIdx)
        x = np.array(new_aoiRowIdx)
        z = np.array(best_z)
        three_points_obj.plane_correlation_with_new_points(allPts_fit, x,y,z,"all_points", slide_name, NewPtsAnalysis,allpointAnalysis,count,new_points_metric)

        ##searching for new points ###using 4 line points
        number_grids = bestZ_grid[0].size
        status, corner_points = three_points_obj.four_lines_points()
        
        #new_all_points  = three_points_obj.point_filtering(corner_points)
        bestZ_points = bestZ_grid[1:,:]
        new_all_points  = three_points_obj.point_filtering_bestZ(corner_points,bestZ_points)
        new_all_points  = three_points_obj.point_filtering(new_all_points)
        
        #print("new_all_points: ", new_all_points)
        num_points_take = number_grids + new_all_points.shape[1]
        #print("number of points used: ", num_points_take)
        if(num_points_take < 3):
            #print("number of points are less than 3, plane not possible")
            return
        
        #print("#"*30, "number of grids: ", number_grids, "#"*30,"\n")
        #print("#"*30, "num_points_take for plane: ", num_points_take, "#"*30,"\n")
         
        ##using far points and remianing points
        #print("*"*30,"\n calculating z plane points using nearby centroid points and 6 line points: ")
        z_plane_points = np.full((3,num_points_take), -1, dtype=np.float32)
        z_plane_points[:,:number_grids] = bestZ_grid[1:,:]
        z_plane_points[:,number_grids:]  = new_all_points[:,:]
        
        #print("z_plane_points: ", z_plane_points)
        x = z_plane_points[0]
        y = z_plane_points[1]
        z = z_plane_points[2]
        three_points_obj.store_new_points(x,y,z,pointStore, count,slide_name)
        name = "4line_"
        three_points_obj.points_on_image(x,y,z,new_all_points,name)
        fit = three_points_obj.plane_fitting_max_point(x,y,z)
        #print("\n","plane fit: \n",fit)
        #print("\n","#"*10, "Fitting new points on New Global Plane using new points", "#"*10, "\n")
        three_points_obj.plane_correlation_with_new_points(fit, x,y,z,"new_points_metric", slide_name, NewPtsAnalysis,allpointAnalysis,count,new_points_metric)


#         ###plot_plane_on_points
#         print("#"*10, "plot_plane_on_points : focus Sampling Plane", "#"*10, "\n")
#         y = np.array(focus_new_aoiColIdx)
#         x = np.array(focus_new_aoiRowIdx)
#         z = np.array(focus_best_z)
#         three_points_obj.plot_plane_on_points(fit,x,y,z)

        #print("#"*10, "plot_plane_on all points : New Global Plane", "#"*10, "\n")
        y = np.array(new_aoiColIdx)
        x = np.array(new_aoiRowIdx)
        z = np.array(best_z)
        three_points_obj.plot_plane_on_points(fit,x,y,z)

        #print("\n","#"*10, "Fitting all points on New Global Plane using new points", "#"*10, "\n")
        y = np.array(new_aoiColIdx)
        x = np.array(new_aoiRowIdx)
        z = np.array(best_z)
        three_points_obj.plane_correlation_with_new_points(fit, x,y,z,"new_points", slide_name, NewPtsAnalysis,allpointAnalysis,count,new_points_metric)

        #print("\n","#"*10, "Writing z for all focus sampling points from New Global Plane", "#"*10, "\n")
        three_points_obj.get_plane_estimated_z_value_focus_sampling(fit)
        
        #print("\n","#"*10, "Writing z for all points from New Global Plane", "#"*10, "\n")
        three_points_obj.get_plane_estimated_z_value(fit, new_aoiRowIdx, new_aoiColIdx, best_z, aoiName ,aoiBestIdx)

#         print("\n","#"*10, "Writting z for all focus sampling points from inline focus Sampling Plane", "#"*10, "\n")
#         three_points_obj.get_inline_plane_estimated_z_value_focus_sampling(fit,z_plane_points)

#         print("\n","#"*10, "Writting z for all points from inline Plane", "#"*10, "\n")
#         three_points_obj.inline_get_plane_estimated_z_value()

#         print("\n","#"*10, "Analyzing max_positive abf negative steps from inline Plane", "#"*10, "\n")
#         three_points_obj.inline_step_analysis(slide_name, NewPtsAnalysis,allpointAnalysis,count)

#         print("\n","#"*10, "creating EXCEL", "#"*10, "\n")
#         three_points_obj.createExcel()


        print("\n\n","*"*50,"END", "*"*50, "\n")
    except Exception as msg:
        print("[error]: ", msg)

# |----------------------------------------------------------------------------|
# main
# |----------------------------------------------------------------------------|
def main():
    """
    @note path_to_save : folder where all data will be saved
    @note path: folder which contains different slides, which need to be processed
    @note Create folder by name all_bestZ_plus_4linesCompleteSlide_pts
    @note Create excel sheet by name Plane_4line_analysis.xlsx
    """
    path_to_save = "/Users/priyanka/Documents/z_index_quality"
    path = "/Users/priyanka/Documents/z_index_quality/data_new_copy/"
    excel_path = os.path.join(path_to_save,"Plane_4line_analysis.xlsx")
    workbook_plane_analysis = xlsxwriter.Workbook(excel_path)
    details = workbook_plane_analysis.add_worksheet("details")
    NewPtsAnalysis = workbook_plane_analysis.add_worksheet("newPoints_Analysis")
    allpointAnalysis = workbook_plane_analysis.add_worksheet("allpoint_Analysis")
    new_points_metric = workbook_plane_analysis.add_worksheet("only_newPoints_Analysis")
    
    details.write(0,0,"details")
    details.write(0,1,"plane pts");details.write(0,2,"analysis pts")
    details.write(1,0,"newPoints_Analysis");details.write(1,1,"newPoints");details.write(1,2,"AllFGPts")
    details.write(2,0,"allpoint_Analysis");details.write(2,1,"AllFGPts");details.write(2,2,"AllFGPts")
    details.write(3,0,"only_newPoints_Analysis");details.write(3,1,"newPoints");details.write(3,2,"newPoints")
    
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
            "num_grids"
             ]
    for _num in range(0,len(list)):
        NewPtsAnalysis.write(0,_num,list[_num])
        allpointAnalysis.write(0,_num,list[_num])
        new_points_metric.write(0,_num,list[_num])
        
    ##point store
    pointStore = workbook_plane_analysis.add_worksheet("newPoints_coordinates")
    list2 = ["slideName", "total points", "row no.", "col no.", "z","row no.", "col no.", "z",
             "row no.", "col no.", "z"]
    for _num in range(0,len(list2)):
        pointStore.write(0,_num,list2[_num])
    
    excel_line =0
    slide_names = os.listdir(path)
    slide_names = sorted(slide_names, reverse=True)
    total_slide = 2
    for i in range (0,len(slide_names)):
        if(i>=total_slide):
            break
        slide = slide_names[i]
        if slide == ".DS_Store": continue
        center_points_process(path, slide,NewPtsAnalysis,allpointAnalysis,excel_line,pointStore,new_points_metric,path_to_save)
        excel_line +=1
    workbook_plane_analysis.close()
 # |----------------------End of main-----------------------------------|   
if __name__ == "__main__":
    main()
    """
    ########check######
    @note path_to_save in  main(): folder where all data will be saved
    @note path in main (): folder which contains different slides, which need to be processed
    @note _dbPath in getAoidetails(): db file path
    """