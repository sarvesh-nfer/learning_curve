import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import cv2
import sys
import os
import glob

class SlideDetectionFromSlot():
    """! Module to detect the slide from slot region.
        This class detects the region of the slide present in the 
        slot region based on the boundaries detected
    """

# |----------------------------------------------------------------------------|
# class Variables
# |----------------------------------------------------------------------------|
    #no classVariables
    
# |----------------------------------------------------------------------------|
# Constructor
# |----------------------------------------------------------------------------|
    def __init__(self):
        """! Init part of the code
        
        """
        ## Start x co-ordinate of the slot
        self._slot_start_x = 0
        ## Start y co-ordinate of the slot
        self._slot_start_y = 0
        ## Width of the slot
        self._slot_width = 0
        ## Height of the slot
        self._slot_height = 0
        ## Minimum threshold for canny detection
        self._canny_min = 6
        ## Maximum threshold for canny detection
        self._canny_max = 35
        ## Minimum threshold for noise removal
        self._whiteMinThreshold = 30
        ## Offset value to detect the boundaries of the slide
        self._edge_detection_offset = 200
# |-------------------------End of Constructor------------------------------|
# |----------------------------------------------------------------------------|
# _getBoundaries
# |----------------------------------------------------------------------------|
    def _getBoundaries(self, edgesImage, slotRegion, oneXImage, processedImagesPath):
        """! Method to estimate the boundaries of the slide.

        This function determines the boundaries of the slide by drawing parallel lines 
        at definite intervals on the gradient output image

        @param edgesImage Binary image which is the result of computing the gradient
        @param slotRegion Binary mask highlighting the slot region
        @param oneXImage RGB image where slide boundary is to be estimated
        @param processedImagesPath Debug path to save the outcomes

        @return Boundary of the slide in the order - x1, y1, x2, y2
        @return Angle of rotation of the rotated rectangle
        @return Binary mask of the slide
        """ 
        
        try:
            edgesImage = edgesImage//2
            lineStartX = 0
            lineEndX = edgesImage.shape[1]
            y_points, x_points = np.nonzero(slotRegion)
            min_y = np.amin(y_points)
            max_y = np.amax(y_points)
            x_border_start = []
            x_border_end = []
            outputMask = edgesImage.copy()
            arrayValues = []
            for j in range(min_y+self._edge_detection_offset, max_y-self._edge_detection_offset, self._edge_detection_offset):
                lineMask = np.zeros(edgesImage.shape, np.uint8)
                cv2.line(lineMask, (lineStartX, j), (lineEndX, j), 127, 1)
                combinedMask = edgesImage + lineMask
                outputMask += lineMask
                maxValue = np.amax(combinedMask)
                if maxValue > np.amax(edgesImage):
                    intersectLocs = np.where(combinedMask == maxValue)
                    min_x_present = np.amin(intersectLocs[1])
                    index_x = np.where(intersectLocs[1] == min_x_present)
                    y_for_x = intersectLocs[0][index_x[0]]
                    arrayValues.append((min_x_present, y_for_x[0]))
                    max_x_present = np.amax(intersectLocs[1])
                    index_x = np.where(intersectLocs[1] == max_x_present)
                    y_for_x = intersectLocs[0][index_x[0]]
                    arrayValues.append((max_x_present, y_for_x[0]))
                    x_border_start.append(min_x_present)
                    x_border_end.append(max_x_present)
            x_start = np.amin(x_border_start)
            x_end = np.amax(x_border_end)
            rect = cv2.minAreaRect(np.array(arrayValues))

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            anglesInBox = []
            maxDiff = []
            for j in range(0, len(box)):
                currentPoint = box[j]
                if j == len(box)-1:
                    nextPoint = box[0]
                else:
                    nextPoint = box[j+1]
                maxDiff.append(nextPoint[1] - currentPoint[1])
                if nextPoint[0] - currentPoint[0] == 0:
                    currentAngle = 90
                else:
                    angleVal = np.arctan((nextPoint[1] - currentPoint[1])/(nextPoint[0] - currentPoint[0]))
                    currentAngle = np.rad2deg(angleVal)
                anglesInBox.append(currentAngle)
            maxYDiffIndex = np.argmax(maxDiff)
            angleValue = anglesInBox[maxYDiffIndex]


            slideMask = np.zeros(oneXImage[:,:,1].shape, np.uint8)
            print("Angle: ", angleValue)
            for i,j in enumerate(box):
                # if (i != min_x_idx ) or (i != max_x_idx):
                x_new = int(j[0] + oneXImage.shape[0]*np.cos(np.deg2rad(angleValue)))
                y_new = int(j[1] + oneXImage.shape[0]*np.sin(np.deg2rad(angleValue)))
                cv2.line(slideMask, (j[0], j[1]), (x_new, y_new), 255, 2)
                x_new = int(j[0] - oneXImage.shape[0]*np.cos(np.deg2rad(angleValue)))
                y_new = int(j[1] - oneXImage.shape[0]*np.sin(np.deg2rad(angleValue)))
                cv2.line(slideMask, (j[0], j[1]), (x_new, y_new), 255, 2)

            cv2.line(slideMask, (0, min_y), (slideMask.shape[1]-1, min_y), 255,1)
            cv2.line(slideMask, (0, max_y), (slideMask.shape[1]-1, max_y), 255,1)
            slideMask = 255 - slideMask
            ccOut = cv2.connectedComponents(slideMask, 8, cv2.CV_32S)
            slideMask[slideMask != 0] = 0
            cv2.drawContours(slideMask,[box],-1, 255,-1)
            nonZeroSlide = np.nonzero(slideMask)
            labelVal = ccOut[1][nonZeroSlide]
            labelVal = np.unique(labelVal)
            for j in labelVal:
                if j != 0:
                    slideMask[ccOut[1] == j] = 255
            # cv2.imwrite(processedImagesPath + "/slideMask4_final.png", slideMask)
            cv2.drawContours(oneXImage,[box],-1,(255,0,0),2)
            return x_start, min_y, x_end, max_y, rect[-1], slideMask
        except Exception as msg:
            return -1, -1, -1, -1, 0, None
            print("Exception occured in getting boundary coordinates, ", msg)
# |----------------------End of _getBoundaries---------------------------|

# |----------------------------------------------------------------------------|
# processPipeline
# |----------------------------------------------------------------------------|
    def processPipeline(self, oneXimage, whiteImage, startX, startY,
                        width, height, processedImagesPath):
        """! Method to detect the slide from the slot region specified.
        This function estimates the boundaries of the slide present in the slot 
        by computing the intensity difference between neighbouring pixels present 
        in the image

        @param oneXimage RGB image where slide boundary is to be estimated
        @param whiteImage RGB image without the slide for reference
        @param startX Start x co-ordinate of the slot
        @param startY Start y co-ordinate of the slot
        @param width Width of the slot
        @param height Height of the slot
        @param processedImagesPath Debug path to save the images

        @return Boundary of the slide in the order - x1, y1, x2, y2
        """ 
        try:
            # cv2.imwrite(processedImagesPath + "/input_image.jpeg", oneXimage)

            endX = startX + width
            endY = startY + height
            slideMask = np.zeros(oneXimage.shape, np.uint8)
            bgSubtractedImage = oneXimage[:,:,1] - whiteImage[:,:,1]
            bgSubtractedImage = bgSubtractedImage + 160
            bgSubtractedImage[bgSubtractedImage < 0] = 0
            bgSubtractedImage = np.uint8(bgSubtractedImage)
            medianOut = cv2.medianBlur(bgSubtractedImage, 3)
            # showImage(medianOut, "median")
            slotRegion = np.zeros(oneXimage[:,:,1].shape, np.uint8)
            slotRegion[startY:endY, startX:endX] = 255  
            whiteGreen = whiteImage[:,:,1].copy()
            whiteGreen[whiteGreen > self._whiteMinThreshold] = 255
            whiteGreen[whiteGreen != 255] = 0
            oneXcopy = oneXimage.copy()
            oneXcopy[:,:,0] = oneXimage[:,:,1].copy()
            oneXcopy[:,:,2] = oneXimage[:,:,1].copy()
            tempMaskImage = np.zeros(oneXimage[:,:,1].shape, np.uint8)
            tempMaskImage[startY:endY, startX:endX] = 255
            edges = cv2.Canny(medianOut,self._canny_min,self._canny_max)
            # cv2.imwrite(processedImagesPath + "/totalEdges.jpeg", edges)
            requiredEdges = cv2.bitwise_and(slotRegion, edges)
            # cv2.imwrite(processedImagesPath + "/requiredEdges.jpeg", requiredEdges)
            kernel = np.ones((3,3), np.uint8)
            requiredEdges = cv2.dilate(requiredEdges, kernel, iterations=1)
            x_start, y_start, x_end, y_end, angle, slideMask = self._getBoundaries(requiredEdges, tempMaskImage, oneXcopy, processedImagesPath)
            cv2.rectangle(oneXcopy,(x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            # cv2.imwrite(processedImagesPath + "/boundaries_slide_withAngle.jpeg", oneXcopy)

            #Make sure the slide region doesnot exceed the slot region
            if x_start < startX:
                x_start = startX
            if y_start < startY:
                y_start = startY
            if x_end > startX + width:
                x_end = startX + width
            if y_end > startY + height:
                y_end = startY + height
            
            return x_start, y_start, x_end, y_end
        except Exception as msg:
            print("Exception occured in slide detection, ", msg)
            return -1, -1, -1, -1
    
    def segmentImage(input):
        pixSize = 40
        img_for_pallavi = cv2.imread(os.path.join(input,"segment.png"))

        dbConn = sqlite3.connect(glob.glob(os.path.join(input_path,'*.db'))[0])
        c1 = dbConn.cursor()
        c1.execute("select aoi_name, aoi_row_idx, aoi_col_idx, bg_state_acq, best_idx, stack_size, aoi_x, aoi_y from aoi where focus_metric > 6 and color_metric > 40 and bg_state_acq = 0")
        all_data = c1.fetchall()

        aoi_names_list = [item[0] for item in all_data]
        row_indices = [item[1] for item in all_data]
        col_indices = [item[2] for item in all_data]
        acq_bg_state = [item[3] for item in all_data]
        best_idx_list = [item[4] for item in all_data]
        stack_size_list = [item[5] for item in all_data]
        aoi_coord_x = [item[6] for item in all_data]
        aoi_coord_y = [item[7] for item in all_data]

        # out_of_focus_list = [item[6] for item in all_data]
        # aoi_names_list = self.aoiName
        # row_indices = self.aoiRowIdx
        # col_indices = self.aoiColIdx
        # acq_bg_state = self.background
        # best_idx_list = self.aoiBestIdx
        # stack_size_list = self._stack_size_list
        # out_of_focus_list = self._aoi_oof_list

        for i in range(len(aoi_names_list)):
            
            aoi_1x_width = 12.4138#12
            aoi_1x_height = 7.7783#7
            start_x = int(aoi_coord_x[i]) + 12 #+ (2*aoi_1x_width)
            start_y = int(aoi_coord_y[i]) - 7#+ (2*aoi_1x_height)
            end_x = int(start_x + aoi_1x_width)
            end_y = int(start_y + aoi_1x_height)
            # start_x = int(aoi_coord_x[i]) #+ (2*aoi_1x_width)
            # start_y = int(aoi_coord_y[i]) #+ (2*aoi_1x_height)
            # img_for_pallavi[start_y:end_y, start_x:end_x] = (0,255,0)


            tempCol = col_indices[i]
            tempRow = row_indices[i]
            stack_size = stack_size_list[i]

            startX = tempCol*pixSize
            endX = tempCol*pixSize + pixSize
            startY = tempRow * pixSize
            endY = tempRow*pixSize + pixSize

            # remove backgrounds
            if acq_bg_state[i] == 1:
                # gridImage[startY:endY, startX:endX] = (75,75,75)
                img_for_pallavi[start_y:end_y, start_x:end_x] = (0,255,255)
                continue
            img_for_pallavi[start_y:end_y, start_x:end_x] = (0,255,0)

        cv2.imwrite(os.path.join(input,"40xMAP.png"), img_for_pallavi)

if __name__ == '__main__':

    if len(sys.argv) < 1:
        print("Inavlid input arguments\n\n<python merged.py>\n\t"\
                "1.Input Path\n")
        sys.exit(0)
    else:
        input_path = sys.argv[1]

        onex_p = os.path.join(input_path,'input_image_0.png')
        white_p = os.path.join(input_path,'onex_white_ref.png')
        log_p = glob.glob(os.path.join(input_path,'*.log'))[0]
        input_img = cv2.imread(onex_p)
        white_img = cv2.imread(white_p)

        data = pd.read_table(log_p)
        print(data.shape)
        def processPipeline(foregroundImage, whiteReference, whiteCorrectionFactor):
            '''Divide the image captured with the white image saved to remove the gradient present in the image'''
            try:
                foregroundImage = foregroundImage
                whiteReferenceImage = whiteReference
                _whiteCorrectionFactor = whiteCorrectionFactor
                #Do white correction, clipping and convert it to uint8
                _whiteCorrectedImage = (foregroundImage/(whiteReferenceImage+1))*_whiteCorrectionFactor
                _whiteCorrectedImage = np.clip(_whiteCorrectedImage, 1, 255)
                _whiteCorrectedImage = _whiteCorrectedImage.astype(np.uint8)
                return _whiteCorrectedImage
            except Exception as msg:
                print("Exception occured in white correction, ", msg)

        for i in range(len(data)):
            if 'Crop Region' in data.iloc[i,:].values[0]:
                d = data.iloc[i+1,:].values[0].split(':')
                x = int(d[4].split(' ')[1])
                y = int(d[5].split(' ')[1])
                w = int(d[6].split(' ')[1])
                h = int(d[7].split(' ')[1])
                gh = int(d[8].split(' ')[1])
                
                #break
        ob = SlideDetectionFromSlot()
        # x1,y1,x2,y2 = ob.processPipeline(input_img, white_img, x, y, w, h, '/home/adminspin/Desktop/office/Localization/')
        output_img = processPipeline(input_img[y:y+h, x:x+w, :], white_img[y:y+h, x:x+w, :], 220)
        cv2.imwrite(os.path.join(input_path,'segment.png'), output_img)
        SlideDetectionFromSlot.segmentImage(input_path)
