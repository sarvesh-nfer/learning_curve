import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sqlite3
import csv
import shutil 


def showImage(image, name):
    """
    imshow with 
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 400, 600)
    cv2.imshow(name, image)
    # if(flag!=-1):
    # cv2.waitKey(0)
    return None

class Data():

    def __init__(self):
        int_a = 1
        self.uintMax = 255
        self.color_list = [7,16,24,30,42,80,105,120,134,149,164,175]

        self.blobXCoord = [0,0,0,0,0,138,138,138,138,138,276,276,276,276,276,414,414,414,414,414,552,552,552,552,552,690,690,690,690,690,828,828,828,828,828]
        self.blobYCoord = [0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484]

        self.height = 121
        self.width = 138

    def map_bubbles(self):
        all_slides_path = "/datadrive/wsi_data/Analysis_Acq_pipeline/Integration_Testing/2_Oct"
        for slide_name in os.listdir(all_slides_path):
            slide_path = os.path.join(all_slides_path, slide_name)

            for name_fold in os.listdir(slide_path):
                # print("name--", name_fold)
                if len(name_fold.split("_")) > 2: continue
                grid_name = name_fold.split("_")[0]

                if grid_name == 'grid':

                    grid_path = os.path.join(slide_path, name_fold)
                    raw_images_path = os.path.join(grid_path, "raw_images")
                    debug_fold_path = os.path.join(grid_path, "acquisition_debug_data/acq_intermediate_imgs")
                    
                    output_folder =  os.path.join(grid_path, "bubbles_mapped_images")
                    if not os.path.exists(output_folder):
                        os.mkdir(output_folder)

                    if os.path.exists(raw_images_path) and os.path.exists(debug_fold_path):
                        for aoi_full_name in os.listdir(raw_images_path):
                            aoi_name = aoi_full_name.split(".")[0]
                            aoi_path = os.path.join(debug_fold_path, aoi_name)

                            if os.path.exists(aoi_path):
                                bubble_mask_path = os.path.join(aoi_path, aoi_name + "_bubble_mask.jpeg")
                                if os.path.exists(bubble_mask_path):
                                    print(aoi_name)
                                    
                                    raw_image = np.zeros((1216,1936,3), np.uint8)
                                    input_image = cv2.imread(os.path.join(raw_images_path, aoi_full_name))
                                    # print(input_image.sh)
                                    raw_image[12:input_image.shape[0]+12, 12:input_image.shape[1]+12] = input_image
                                    # input_image 

                                    bubble_mask = cv2.imread(bubble_mask_path, 0)
                                    bubble_mask = cv2.resize(bubble_mask, (int(raw_image.shape[1]),int(raw_image.shape[0])), cv2.INTER_NEAREST)
                                    bubble_mask[bubble_mask > 150] = 255
                                    bubble_mask[bubble_mask < 150] = 0

                                    contours, _ = cv2.findContours(np.uint8(bubble_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                    for i in range(0, len(contours)):
                                        # if (cv2.contourArea(contours[i])< 50): continue
                                        cv2.drawContours(raw_image, contours, i, (255,255,0), 2)
                                    print("Output\t",os.path.join(output_folder, aoi_name+".jpeg"))
                                    cv2.imwrite(os.path.join(output_folder, aoi_name+".jpeg"), raw_image)

    def copy_whiteCorrectedImages(self):
        path = "/home/adminspin/Desktop/Annotations_done/Slides"
        # output_path = "/home/adminspin/Slides_data/Localisation_Repository/Test_UNET_NLMD_1/2_GradientImgs"
        output_path = "/home/adminspin/Desktop/Annotations_done/gt_masks"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            try:
                # input_img_path = os.path.join(path,item+"/loc_output_data/whiteCorrectedInput.png")
                input_img_path = os.path.join(path,item+"/gt_mask.png")

                '''
                input_img_path = os.path.join(path,item+"/loc_output_data/compositeImage.png")
                output_img_path = os.path.join(path,item+"/loc_output_data/composite_resized.png")
                input_img = cv2.imread(input_img_path)
                resized = cv2.resize(input_img, (int(input_img.shape[1]/2.2),int(input_img.shape[0]/2.2)), cv2.INTER_NEAREST)
                cv2.imwrite(output_img_path, resized)
                continue
                '''

                # print("Input Path : ", input_img_path)
                
                input_img = cv2.imread(input_img_path)

                out_img_path = os.path.join(output_path, item + ".png")
                if os.path.exists(out_img_path):
                    _flag = True
                    count = 1
                    while(_flag):
                        out_img_path = out_img_path.split(".png")[0]
                        slide_path_new = out_img_path+"_"+str(count)+".png"
                        if not os.path.exists(slide_path_new):
                            _flag = False
                        else:
                            count += 1
                    out_img_path = slide_path_new
                print(out_img_path)
                cv2.imwrite(out_img_path, input_img)
            except Exception as msg:
                print(item,"\t", msg)
    # |----------------------------------------------------------------------------|
    # removeNoise
    # |----------------------------------------------------------------------------|
    def removeNoise(self, input_mask, minArea):
        '''
        Function to remove the noise presnt in the input image 
        @Inputs : 1. Mask image
                  2. Minimum area threshold for the blob to called as noise
        @Output : Returms the mask after removing the noise present in the input image 
        '''
        nc, lb, st, cnt = cv2.connectedComponentsWithStats(input_mask)
        blobs_in_mask = nc 
        check_SingleBlob = np.zeros(input_mask.shape, np.uint8)
        return_img = np.zeros(input_mask.shape, np.uint8)

        for j in range(1, nc):

            area = st[j, 4]
            #Check for min area of the blob to remove the noise- Dont check for annotation on them
            if area > minArea : 
                check_SingleBlob = self.uintMax*np.uint8((lb == j))
                return_img += check_SingleBlob 
        return return_img
    # |---------------------- End of removeNoise ---------------------------|
    def separate_dark_nuclei_slides(self):
        path = "/home/adminspin/Slides_data/Stitching_data/dark_nuclei_slides_sample_data/input"
        output_path = "/home/adminspin/Slides_data/Stitching_data/dark_nuclei_slides_sample_data/output"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        dark_path = os.path.join(output_path, "dark_nuclei_slides")
        if not os.path.exists(dark_path) : os.mkdir(dark_path)

        other_path = os.path.join(output_path, "other_slides")
        if not os.path.exists(other_path) : os.mkdir(other_path)

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            try:
                input_img_path = os.path.join(path,item+"/whiteCorrectedInput.png")
                input_img = cv2.imread(input_img_path)

                green_img = input_img[:, :, 1]

                new_th_img = np.zeros(green_img.shape, np.uint8)
                new_th_img[green_img < 140] = 255
                new_th_img = self.removeNoise(new_th_img, 100)

                _, th_img = cv2.threshold(green_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

                th_img = new_th_img
                biopsy_area = np.count_nonzero(th_img)

                plt.figure("hist", figsize=(10,6))
                hist, bins = np.histogram(hsv[:, :, 0][th_img > 150], 180, [0,180])
                
                weight1 = 0
                weight2 = 0
                for i in range(130,180):
                    if i < 165:
                        weight1 += hist[i]
                    else:
                        weight2 += hist[i]
                print(item,"\t",weight1,"\t",weight2)
                mean, std = cv2.meanStdDev(hsv[:, :, 0][hsv[:, :, 0] > 120])
                # print(item, "\t", mean[0][0], "\t", std[0][0])
                plt.plot(bins[1:],hist, color = 'r')

                plt.xlim(0,180)
                # plt.ylim(1,np.max(hist)+500)
                plt.xlabel('Hue')
                plt.ylabel('Frequency')
                plt.title(str(item))
                # plt.pause(0.1)

                # plt.savefig(output_path+"/"+item+".png")
                showImage(input_img, "input_img")
                showImage(th_img, "th_img")
                showImage(new_th_img, "new_th_img")
                # cv2.waitKey(0)
                # plt.show()
                plt.clf()
                ratio = ((weight1 + weight2) / biopsy_area )*100
                if max(weight1, weight2) > 1000 and ratio > 70:
                    if weight1 > weight2:
                        cv2.imwrite(os.path.join(dark_path, item+".png"), input_img)
                    else:
                        cv2.imwrite(os.path.join(other_path, item+".png"), input_img)
                # cv2.imwrite(out_img_path, input_img)

            except Exception as msg:
                print(item,"\t", msg)

    def image_type_conversion(self):
        path = "/home/adminspin/Slides_data/Debris_Removal_Data/Single_Images_output"
        output_path = "/home/adminspin/Slides_data/Debris_Removal_Data/Single_Images_output_compressed"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            # input_img = cv2.imread(os.path.join(path, item),0)
            # print(input_img.shape[0],input_img.shape[1])
            # mask_img = np.zeros(input_img.shape, np.uint8)
            # cv2.imwrite(os.path.join(output_path, item), mask_img)

            aoi_name = item.split(".")[0]

            # Create inverse mask
            input_img = cv2.imread(os.path.join(path, item))
            input_img = cv2.resize(input_img, (1936,1216), cv2.INTER_NEAREST)
            
            cv2.imwrite(os.path.join(output_path, aoi_name+".jpeg"), input_img)

    def showBubbles(self):
        input_image = cv2.imread("/home/adminspin/Downloads/23.jpeg")
        input_mask = cv2.imread("/home/adminspin/Downloads/13875.0_41400.0_bubble_msk.jpeg", 0)

        input_mask[input_mask < 150]=0
        input_mask[input_mask > 150]=255

        input_mask = cv2.bitwise_not(input_mask)
        gradient = cv2.morphologyEx(input_mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))

    def copy_sampled_imgs_for_bestZ(self):
        """
        copy all the best z points images to one location
        """
        input_slides_path = "/home/adminspin/Slides_data/Focus_error_metric/ValidationData/all_slides"
        output_path = "/home/adminspin/Slides_data/Focus_error_metric/ValidationData/consolidated"
        # if not os.path.exists(output_path):
        #     os.mkdir(output_path)
        
        for slide_name in os.listdir(input_slides_path):
            validation_path = os.path.join(input_slides_path, slide_name+"/validation/acquisition_debug_data/sampled_images")
            for img_name in os.listdir(validation_path): 
                img = cv2.imread(os.path.join(validation_path, img_name))
                cv2.imwrite(os.path.join(output_path, img_name), img)
            # shutil.copytree(validation_path, output_path)

    def createMask(self):

        path = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_04March_S1/0_slide_images"
        output_path = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_04March_S1/1_gt_masks"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            # input_img = cv2.imread(os.path.join(path, item),0)
            # print(input_img.shape[0],input_img.shape[1])
            # mask_img = np.zeros(input_img.shape, np.uint8)
            # cv2.imwrite(os.path.join(output_path, item), mask_img)

            # Create inverse mask
            input_img = cv2.imread(os.path.join(path, item),0)
            print(input_img.shape[0],input_img.shape[1])
            
            mask_img = np.zeros(input_img.shape, np.uint8)
            # mask_img[input_img < 150] = 255
            # mask_img[input_img > 150] = 0
            
            cv2.imwrite(os.path.join(output_path, item), mask_img)
     

    def overlay_Mask_OnRGB(self):
        input_slides = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_25Feb/whiteCorrectedImages"
        input_masks = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_25Feb/FG_Masks_from_UNET"
        output_path = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_25Feb/overlayed_foreground_masks"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(input_slides)
        list_dir = sorted(list_dir)

        for item in list_dir:
            input_img = cv2.imread(os.path.join(input_slides, item))
            input_mask = cv2.imread(os.path.join(input_masks, item), 0)

            # input_img[:,:,1][input_mask > 150] = 220

            contours, _ = cv2.findContours(input_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for i in range(0, len(contours)):
                cv2.drawContours(input_img, contours, i, (255,0,255), 1)

            cv2.imwrite(os.path.join(output_path, item), input_img)

    def get_IoU(self):

        # input_slides = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/0_TestData/0_input_slides"
        # input_gtMasks = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/0_TestData/1_gt_masks"
        # input_masks = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/0_TestData/th_0.5"

        input_gtMasks = "/home/adminspin/office/Experiments/UnitTesting/1_gt_masks"
        input_masks = "/home/adminspin/office/Experiments/UnitTesting/22feb_v4"

        list_dir = os.listdir(input_gtMasks)
        list_dir = sorted(list_dir)

        iou_list = []
        for item in list_dir:
            # print(item)
            input_img = cv2.imread(os.path.join(input_slides, item))
            input_mask = cv2.imread(os.path.join(input_masks, item), 0)
            input_gtMask = cv2.imread(os.path.join(input_gtMasks, item), 0)

            or_img = cv2.bitwise_or(input_mask, input_gtMask)
            and_img = cv2.bitwise_and(input_mask, input_gtMask)

            or_img[or_img < 150] = 0
            or_img[or_img > 150] = 255

            and_img[and_img < 150] = 0
            and_img[and_img > 150] = 255

            intersection_area = np.count_nonzero(and_img)
            union_area = np.count_nonzero(or_img)

            _iou = (intersection_area/ union_area)
            iou_list.append(_iou)
            print(item,"\tIoU:\t",_iou )
        print("\n\nMean_IoU:\t", np.mean(iou_list))
    # |----------------------------------------------------------------------------|
    # contrastEnhancement
    # |----------------------------------------------------------------------------|
    def contrastEnhancement(self):
        path = "/home/adminspin/Pictures/0_For_Localization_Report/UNet/0_input_slides"
        output_path = "/home/adminspin/Pictures/0_For_Localization_Report/UNet/0_input_slides_enhanced"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            inputImage = cv2.imread(os.path.join(path, item))

            outputImage = inputImage.copy()
            claheGreen = cv2.createCLAHE(clipLimit=7, tileGridSize=(3, 3))
            claheRest = cv2.createCLAHE(clipLimit=4, tileGridSize=(3, 3))
            claheRed = cv2.createCLAHE(clipLimit=5, tileGridSize=(3, 3))

            for j in range(0, 3):
                currentChannel = inputImage[:,:,j]
                if j == 1:
                    clahe_output = claheGreen.apply(currentChannel)
                    output = cv2.fastNlMeansDenoising(clahe_output, 9, 20)
                    # output = cv2.fastNlMeansDenoising(currentChannel, 9, 20)
                elif j == 2:
                    clahe_output = claheRed.apply(currentChannel)
                    output = cv2.fastNlMeansDenoising(clahe_output, 9, 20)
                    # output = cv2.fastNlMeansDenoising(currentChannel, 9, 20)
                else:
                    clahe_output = claheRest.apply(currentChannel)
                    output = cv2.fastNlMeansDenoising(clahe_output, 9, 20)
                    # output = cv2.fastNlMeansDenoising(currentChannel, 9, 20)
                output = cv2.equalizeHist(currentChannel)
                outputImage[:,:,j] = output.copy()

            cv2.imwrite(os.path.join(output_path, item), outputImage)
    # |----------------------End of contrastEnhancement---------------------------|

    def resizeImages(self):
        path = "/home/adminspin/Slides_data/Debris_Removal_Data/Single_Images/TestCases_DarkBiopsy_AOis"
        output_path = "/home/adminspin/Slides_data/Debris_Removal_Data/Single_Images/TestCases_DarkBiopsy_AOis_re"
        if not os.path.exists(output_path) : os.mkdir(output_path)    

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            input_img = cv2.imread(os.path.join(path, item))

            # cv2.imshow("norm_img", norm_img)
            # cv2.waitKey(0)
            # cv2.imwrite(os.path.join(output_path, item), norm_img)
            # input_mask = cv2.imread(os.path.join(path2, item))
            print(input_img.shape)
            # exit()
            resized = cv2.resize(input_img, (1936,1216), cv2.INTER_NEAREST)
            # print(input_img.shape, "\t",resized.shape)
            cv2.imwrite(os.path.join(output_path, item), resized)

    '''
    def populateFocusColorMetricValuesFromDB_2(raw_path,db_path):
        blobXCoord = [0,0,0,0,0,138,138,138,138,138,276,276,276,276,276,414,414,414,414,414,552,552,552,552,552,690,690,690,690,690,828,828,828,828,828]
        blobYCoord = [0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484,0,121,242,363,484]
        height = 121
        width = 138
        path1 = os.path.join(raw_path,"raw_images")
        print(path1)
        db_file_path = os.path.join(db_path)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        dbConn = sqlite3.connect(db_file_path)
​
        out_path = os.path.join(raw_path,"20x_fm_cm")
        if not os.path.exists(out_path):
            os.mkdir(out_path)
​
        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)
        _grid_id = 4
​
        for item in list_dir:
            # item = 'aoi0133.bmp'
            print(item)
            # if item == "aoi0150.bmp" : break
            aoi_name = item.split(".")[0]
            input_image = cv2.imread(os.path.join(path1, item))
​
            resized = cv2.resize(input_image, (1936, 1216), cv2.INTER_NEAREST)
​
            c = dbConn.cursor()
​
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
​
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
​
            valid_blobs = 0
            for p in range(0, len(blob_index_list)):
                blob_id = blob_index_list[p]
​
                fm_val = round(fm_list_data[p], 2)
                cm_val = round(cm_list_data[p], 2)
                stack_id = stack_index_list[p]
​
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
​
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
    '''
    def zeroPadding(self):
        input_slides_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/With_Aug_08Feb/input_slides"
        input_masks_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/With_Aug_08Feb/input_masks"
        output_slides_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/With_Aug_08Feb/input_slides_padding"
        output_masks_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/With_Aug_08Feb/input_masks_padding"
        
        input_size = 704
        for item in os.listdir(input_slides_path):
            try:
                input_img = cv2.imread(os.path.join(input_slides_path, item))
                output_img = np.zeros((input_size, input_size, 3), np.uint8)
                output_img[0:input_img.shape[0], 0:input_img.shape[1]] = input_img
                cv2.imwrite(os.path.join(output_slides_path, item), output_img)

                input_mask = cv2.imread(os.path.join(input_masks_path, item),0)
                output_mask = np.zeros((input_size, input_size), np.uint8)
                output_mask[0:input_img.shape[0], 0:input_img.shape[1]] = input_mask
                cv2.imwrite(os.path.join(output_masks_path, item), output_mask)
            except Exception as msg:
                print(msg)

        # input_img = cv2.imread("/home/adminspin/Slides_data/IHC_Data/2001V401002_298/loc_output_data/whiteCorrectedInput.png")

        # output_img = np.zeros((input_size, input_size, 3), np.uint8)
        # output_img[0:input_img.shape[0], 0:input_img.shape[1]] = input_img
        # cv2.imwrite("/home/adminspin/Slides_data/IHC_Data/2001V401002_298/loc_output_data/whiteCorrectedInput.png", output_img)

    def split_data(self):

        path1 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/With_Aug_08Feb/"
        path2 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/With_Aug_08Feb/Split_Data/"

        imgs = 'input_slides'
        masks = 'input_masks'

        imgs_ip = os.path.join(path1, imgs)
        masks_ip = os.path.join(path1, masks)

        list_dir = os.listdir(imgs_ip)
        count = 0
        for item in list_dir:

            if count < 316:
                out = os.path.join(path2,'Test_Data')
            else:
                out = os.path.join(path2,'Train_Data')
            
            slide_img = cv2.imread(os.path.join(imgs_ip, item))
            mask_img = cv2.imread(os.path.join(masks_ip, item))

            output_slide_path = os.path.join(out, imgs)
            output_masks_path = os.path.join(out, masks)

            cv2.imwrite(os.path.join(output_slide_path, item), slide_img)
            cv2.imwrite(os.path.join(output_masks_path, item), mask_img)

            count += 1

    def verify_data(self):
        path1 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Without_Aug/input_slides"
        path2 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Without_Aug/input_masks"

        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)

        for item in list_dir:
            if not os.path.exists(os.path.join(path2, item)):
                print(item)
    
    def copy_image_withSameName(self):
        path1 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/DataSet_22Feb/Split_Data_Aug_2/combined_train_valid/inv_mask_images"
        path2 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/DataSet_22Feb/Split_Data_Aug_2/combined_train_valid/inv_mask_images"
        
        out_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/DataSet_22Feb/Split_Data_Aug_2/combined_train_valid/inv_mask_images_0_to_1_train_val"
        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)

        for item in list_dir:
            print(item)
            mask_path = os.path.join(path2, item)
            if not os.path.exists(mask_path): continue
            mask_img = cv2.imread(os.path.join(path2, item),0)


            # if not os.path.exists(os.path.join(path2, item)):
                # print(item)
            mask_img[mask_img < 155] =0
            mask_img[mask_img > 154] = 255
            mask_img = mask_img/255

            cv2.imwrite(os.path.join(out_path, item), mask_img)

    def two_images_withSameName_op(self):
        path1 = "/home/adminspin/Downloads/New/model1"
        path2 = "/home/adminspin/Downloads/New/model2"

        output_path = "/home/adminspin/Downloads/New/model1/model_ensembled"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)

        for item in list_dir:
            print(item)
            mask_path = os.path.join(path2, item)
            if not os.path.exists(mask_path): continue
            
            mask_img1 = cv2.imread(os.path.join(path1, item),0)
            mask_img2 = cv2.imread(os.path.join(path2, item),0)

            mask_img = cv2.bitwise_or(mask_img1, mask_img2)

            mask_img[mask_img < 155] =0
            mask_img[mask_img > 154] = 255

            cv2.imwrite(os.path.join(output_path, item), mask_img)
    
    def doThreshold(self):
        path = "/home/adminspin/Slides_data/Localisation_Repository/Test_UNET_NLMD/0_slide_images"
        output_path = "/home/adminspin/Slides_data/Localisation_Repository/Test_UNET_NLMD/th_for_bubbles"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            input_img = cv2.imread(os.path.join(path, item))
            green_image = input_img[:, :,1]
            
            # mask_img = green_image.copy()
            
            # mask_img[green_image < 178] = 0
            # mask_img[green_image > 196] = 0
            # mask_img[green_image > 0] = 255
            
            mask_img = cv2.inRange(green_image, 178, 196)
            # _, mask_img = cv2.threshold(green_image, 178, 196, cv2.THRESH_BINARY)
            showImage(input_img,"input_slide")
            showImage(mask_img, "mask")
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(output_path, item), mask_img)
    
    def gen_newMask_fromMasks(self):
        path1 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/4_TestData_For_Sharing_BB/m1_v2_results"
        path2 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/4_TestData_For_Sharing_BB/m2_v1_hue_results"
        
        out_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/4_TestData_For_Sharing_BB/m2_v1_biopsy_after_removingExtra"

        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)

        for item in list_dir:
            print(item)
            mask_path = os.path.join(path1, item)
            if not os.path.exists(mask_path): continue

            mask_img1 = cv2.imread(os.path.join(path1, item),0)
            mask_img2 = cv2.imread(os.path.join(path2, item),0)
            
            kernel = np.ones((3,3), np.uint8)
            mask_img1 = cv2.erode(mask_img1,kernel,iterations = 1)

            mask_img1[mask_img2 > 150] = 0
            # new_mask = mask_img1 - mask_img2
            cv2.imwrite(os.path.join(out_path, item), mask_img1)

    def huePlotsForImages(self):
        input_path = "/home/adminspin/Desktop/temp"
        output_path = "/home/adminspin/Desktop/temp_huePlots"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(input_path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            input_img = cv2.imread(os.path.join(input_path, item))
            mask_img = cv2.imread("/home/adminspin/Desktop/Test_Loc/slide_3/loc_output_data/nlmd_unet_combined.png",0)
            hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

            plt.figure("hist", figsize=(16,9))
            # fig.suptitle('Hue')

            hist, bins = np.histogram(hsv[:, :, 0][mask_img> 150], 180, [0,180])
            # for k in range(0, len(hist)):
            #     if hist[k]<2000:
            #         hist[k] = 0 
            plt.plot(bins[1:],hist, color = 'r')

            plt.xlim(0,180)
            plt.ylim(1,np.max(hist)+500)
            plt.xlabel('Hue')
            plt.ylabel('Frequency')
            plt.title(str(item))
            # plt.pause(0.1)

            plt.savefig(output_path+"/"+item+".png")
            # showImage(input_img, "input_img")
            # cv2.waitKey(0)
            plt.show()
            plt.clf()

    def getHuePlots(self):
        input_image = cv2.imread("/home/adminspin/Slides_data/Color Metric/aoi0384.bmp")
        mask_img = cv2.imread("/home/adminspin/Slides_data/Color Metric/aoi0384-1.png", 0)
        output_path = "/home/adminspin/Slides_data/Color Metric/huePlots"

        input_image = cv2.medianBlur(input_image, 3)
        input_float = np.float32(input_image)
        # Get HSV from BGR
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        # hue_image = np.uint8(hsv[:, :, 0])

        nc, lb, st, cnt = cv2.connectedComponentsWithStats(mask_img)

        for j in range(1, nc):
            if st[j, 4] < 150: continue
            print("blobId: ", j)
            img_copy = input_image.copy()
            check_SingleBlob = self.uintMax*np.uint8((lb == j))
            
            th_img = np.zeros(check_SingleBlob.shape, np.uint8)
            th_img[hsv[:, :, 0] < 60] = 255
            th_img[hsv[:, :, 0] > 280] = 255
            th_img[hsv[:, :, 0] == 0] = 0 

            img_copy[:,:,2][check_SingleBlob > 150] = 255
            cv2.imwrite(output_path+"/"+str(j)+"_img.png", img_copy)
            # cv2.imwrite(output_path+"/"+str(j)+"_blob.png", check_SingleBlob)


            hue_list = hsv[:, :, 0][check_SingleBlob == self.uintMax]

            plt.figure("Hue Histogram")

            hist, bins = np.histogram(hue_list, 180, [0,180])
            plt.plot(bins[1:],hist, label = str(j), color = 'r')
            plt.xlim(0,180)
            plt.xlabel('Hue')
            plt.ylabel('Occurance')
            plt.title("Hue - Blob ID: "+str(j))
            # plt.pause(0.1)

            plt.savefig(output_path+"/"+str(j)+"_plot.png")

            plt.show()
            plt.clf()

    def fillMasks(self):
        path = "/home/adminspin/Slides_data/Localisation_Repository/Loc_valid_data_11Feb_with_Unet"

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            try:
                # input_img_path = os.path.join(path,item+"/loc_output_data/whiteCorrectedInput.png")
                input_img_path = os.path.join(path,item+"/loc_output_data/foreground_region_after_combining_unet_nlmd.png")
                
                mask_img = cv2.imread(input_img_path, 0)
                
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # for i in range(0, len(contours)):
                cv2.drawContours(mask_img, contours, -1, 255, -1)

                out_img_path = os.path.join(path,item+"/loc_output_data/foreground_region_after_combining_unet_nlmd_filled.png")
                cv2.imwrite(out_img_path, mask_img)

            except Exception as msg:
                print(item,"\t", msg)
            

    def plotProbabilityMeanValues(self):
        input_slides_path = "/home/adminspin/Slides_data/Onex_Images_Repository/All_Slides/failure_cases/slides"
        input_masks_path = "/home/adminspin/Slides_data/Onex_Images_Repository/All_Slides/failure_cases/masks_original"
        input_prob_path = "/home/adminspin/Slides_data/Onex_Images_Repository/All_Slides/failure_cases/output_v4_prob"
        
        output_path = "/home/adminspin/Slides_data/Onex_Images_Repository/All_Slides/failure_cases/output_v4_prob_bbox"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        count = 0
        list_dir = os.listdir(input_slides_path)
        list_dir = sorted(list_dir)
        for item in list_dir:
            try:
                # if count > 100: break
                input_img = cv2.imread(os.path.join(input_slides_path, item))
                input_mask = cv2.imread(os.path.join(input_masks_path, item),0)
                input_prob_img = cv2.imread(os.path.join(input_prob_path, item),0) 

                output_img = input_img.copy()

                input_mask[input_mask < 254] = 0
                input_mask[input_mask > 254] = 255

                nc, lb, st, cnt = cv2.connectedComponentsWithStats(input_mask)
                for j in range(1, nc):

                    # Area of the blob 
                    area = st[j, 4]
                    if area < 50 : 
                        continue 

                    # Single blob image
                    check_SingleBlob = self.uintMax*np.uint8((lb == j))

                    # Get the Hue values list to compute the variance
                    vals_list = input_prob_img[check_SingleBlob == self.uintMax]
                    _mean = int(np.mean(vals_list))

                    cv2.rectangle(output_img, (st[j,0],st[j,1]), (st[j,0]+st[j,2],st[j,1]+st[j,3]), (0,255,0), 1)
                    cv2.putText(output_img, "{}".format(str(_mean)), (st[j,0]+5, st[j,1]+15),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                cv2.imwrite(os.path.join(output_path, item), output_img)
                count += 1

            except Exception as msg:
                print(msg)

    def getAccuracyMetrics(self):

        input_slides_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/0_TestData/0_input_slides"
        input_gt_masks_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/0_TestData/1_gt_masks"
        input_pred_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/0_TestData/Area_Under_Curve_threshold_Data/th_0.9"
        
        output_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/0_TestData/Area_Under_Curve_threshold_Data/test"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        count = 0
        list_dir = os.listdir(input_gt_masks_path)
        list_dir = sorted(list_dir)
        
        precision_list = []
        recall_list = []

        tpr_list = []
        fpr_list = []
        
        for item in list_dir:
            try:
                input_img = cv2.imread(os.path.join(input_slides_path, item))
                input_gt_mask = cv2.imread(os.path.join(input_gt_masks_path, item),0)
                input_pred_mask = cv2.imread(os.path.join(input_pred_path, item),0)

                input_gt_mask[input_gt_mask < 155] = 0
                input_gt_mask[input_gt_mask > 154] = 255

                input_pred_mask[input_pred_mask < 155] = 0
                input_pred_mask[input_pred_mask > 154] = 255

                gt_inv_mask = cv2.bitwise_not(input_gt_mask)
                pred_inv_mask = cv2.bitwise_not(input_pred_mask)

                intersection_mask = cv2.bitwise_and(input_gt_mask, input_pred_mask)
                inv_intersection_mask = cv2.bitwise_and(gt_inv_mask, pred_inv_mask)

                false_pos = input_pred_mask.copy()
                false_pos[input_gt_mask > 150] = 0

                false_neg = input_gt_mask.copy()
                false_neg[input_pred_mask > 150] = 0


                tp = np.count_nonzero(intersection_mask)
                fp = np.count_nonzero(false_pos)
                tn = np.count_nonzero(inv_intersection_mask)
                fn = np.count_nonzero(false_neg)

                tpr = (tp)/(tp+fn+1)
                fpr = (fp)/(fp+tn+1)

                tpr_list.append(tpr)
                fpr_list.append(fpr)

                precision = (tp) / (tp+fp+1)
                recall = (tp) / (tp+fn+1)

                precision_list.append(precision)
                recall_list.append(recall)
                # print(item, "\tprecision:\t", precision, "\trecall:\t",recall)

                # showImage(input_img, "input_img")
                # showImage(input_gt_mask, "gt_mask")
                # showImage(input_pred_mask, "pred_mask")
                # cv2.waitKey(0)
            except Exception as msg:
                print(item,"\t", msg)
        
        '''
        mean_prec_list  = [0.90, 0.91, 0.92, 0.93, 0.93, 0.94, 0.95, 0.96]
        mean_recl_list  = [0.96, 0.95, 0.94, 0.94, 0.93, 0.92, 0.91, 0.89]
        th_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        print("Mean Precision: ", np.mean(precision_list),"\t Recall: ", np.mean(recall_list))
        plt.figure()
        
        plt.scatter(th_list, mean_recl_list,c = 'r', label = "recall")
        plt.scatter(th_list, mean_prec_list,c = 'g', label = "precision")

        plt.xlabel("Probability Threshold")
        plt.ylabel("Metric value (Precision/ Recall)")
        plt.legend()
        plt.show()
        '''
    def copyTheImageToOriginalFolder(self):

        slides_folders_path = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_04March_S1/slides"
        imgs_path = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Model_Testing/4_TestData/or_model1_model2"

        list_dir = os.listdir(slides_folders_path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            # try:
            img_path = os.path.join(os.path.join(imgs_path,item+".png"))
            if os.path.exists(img_path):
                img = cv2.imread(img_path)

                output_path = os.path.join(slides_folders_path,item+"/loc_output_data/unet_model1_model2_or.png")
                cv2.imwrite(output_path, img)
    
    def copySpecificImagesFromLocData(self):
        input_names_list = ['whiteCorrectedInput.png','annotations_without_dark_region.png', 'total_coverslip_lines.png', 'total_foreground.png', 'nlmd_output.png']
        
        input_path = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_04March_S1/slides"
        out_path = "/home/adminspin/Slides_data/Localisation_Repository/Loc_data_04March_S1/data_11March"
        if not os.path.exists(out_path) : os.mkdir(out_path)

        list_dir = os.listdir(input_path)
        for item in list_dir:
            try:
                slide_path_input = os.path.join(input_path, item)
                slide_path_output = os.path.join(out_path, item)

                if not os.path.exists(slide_path_output) : os.mkdir(slide_path_output)
                
                src = os.path.join(slide_path_input ,"loc_output_data")
                dst = os.path.join(slide_path_output,"loc_output_data")

                if not os.path.exists(src) : continue
                if not os.path.exists(dst) : os.mkdir(dst)

                # for img_name in input_names_list:
                #     img = cv2.imread(os.path.join(src, img_name))
                #     cv2.imwrite(os.path.join(dst, img_name), img)
                
                for i in range(len(input_names_list)):
                    img = cv2.imread(os.path.join(src, input_names_list[i]))
                    cv2.imwrite(os.path.join(dst, str(i)+"_"+input_names_list[i]), img)

            except Exception as msg:
                print(item,"\t", msg)

    
    def copy_imagesWithInputNameList(self):

        # input_name_list = ['367851.png', '028295_50172.png' , '2001V401001_798.png' , '2001V402002_844.png' , '2001V401001_856.png' , '2001V401002_173.png' , \
            # '2001V401002_174.png ', 'Slide_1_2.png' , 'S3_50284.png' , 'S3_50277.png' , 'S2_50146.png ', 'S1_50071.png' , '367851.png' , '2001V501002_22039.png' ]        
        input_name_list = ['bb2d6d181c40403caa7bb80badc4752d.png' ,  'S3_50282.png' , 'M_LH18654_17_50279.png' , '362566.png' , '362511.png' , '354165.png' ]

        path1 = "/home/adminspin/Slides_data/Onex_Images_Repository/All_Slides/set3/slides"

        out_path = "/home/adminspin/Slides_data/Onex_Images_Repository/All_Slides/set3/SupportingData/model2_failurecases/slides"
        if not os.path.exists(out_path) : os.mkdir(out_path)


        for item in input_name_list:
            print(item)
            mask_path = os.path.join(path1, item)
            if not os.path.exists(mask_path): 
                print('Not Found:\t', item)
                continue
            mask_img = cv2.imread(os.path.join(path1, item))

            cv2.imwrite(os.path.join(out_path, item), mask_img)
    
    def populateFocusColorMetricValuesFromDB(self):
        grid_path = "/home/adminspin/Desktop/bubble/04r/sarvesh/H01CBA04R_8420/grid_3"
        path1 = os.path.join(grid_path, "raw_images")

        db_file_path = "/home/adminspin/Desktop/bubble/04r/sarvesh/H01CBA04R_8420/H01CBA04R_8420.db"
        _grid_id = 3
        stack_size = 5

        self.dbConn = sqlite3.connect(db_file_path)

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

            c = self.dbConn.cursor()

            c.execute("select best_idx from aoi where aoi_name == '"+aoi_name+"' and grid_id =="+str(_grid_id))
            index = c.fetchone()[0]
            # print("index: ",index)
            if index == -1 :
                cv2.putText(resized, "{}".format("Background - No Stack Captured"), (125,125),\
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,255), 2)
                cv2.imwrite(os.path.join(out_path, aoi_name+".jpeg"), resized)
                continue

            c.execute("SELECT blob_index, focus_metric, color_metric, stack_index, hue_metric from acquisition_blob_info WHERE aoi_name = '"+aoi_name+"' and grid_id =="+str(_grid_id)+" and color_metric >= 2")
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
                cv2.rectangle(resized, ((2*self.blobXCoord[blob_id])+2, (2*self.blobYCoord[blob_id])+2),(2*self.blobXCoord[blob_id]+((2*self.width)-2), \
                    2*self.blobYCoord[blob_id]+((2*self.height)-2)), (0,0,0), 1)


                cv2.putText(resized, "{}".format("fm:"+str(fm_val)), (int(2*(self.blobXCoord[blob_id])+self.width), int(2*(self.blobYCoord[blob_id])+self.height)),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                cv2.putText(resized, "{}".format("cm:"+str(cm_val)), (int(2*(self.blobXCoord[blob_id])+self.width), int(2*(self.blobYCoord[blob_id])+self.height)+25),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
                cv2.putText(resized, "{}".format("hm:"+str(hm_val)), (int(2*(self.blobXCoord[blob_id])+self.width), int(2*(self.blobYCoord[blob_id])+self.height)+50),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            cv2.imwrite(os.path.join(out_path, aoi_name+"_"+str(index)+".jpeg"), resized)
            # exit()
    
    def plot_data_from_csv(self):
        path = "/home/adminspin/Slides_data/Focus_error_metric/Thick_tissue/consolidated/test/failure_out/csv"
        output_path = "/home/adminspin/Slides_data/Focus_error_metric/Thick_tissue/consolidated/test/failure_out/csv_out"
        if not os.path.exists(output_path) : os.mkdir(output_path)

        list_dir = os.listdir(path)
        list_dir = sorted(list_dir)

        for item in list_dir:
            csv_path = os.path.join(path, item)

            with open(csv_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                _row_flag = False
                x_list = []
                y_list = []
                count = 0
                above_vals_count = 0
                below_vals_count = 0
                for row in csv_reader:
                    if not _row_flag:
                        _row_flag = True
                        continue
                    else:
                        x_list.append(count)
                        count += 1
                        cm_val = float(row[len(row)-1])
                        print("cm\t", cm_val)
                        if cm_val < 20:
                            y_list.append(-1)    
                        else:
                            _val = int(row[len(row)-2])
                            print(_val)
                            if _val < 5:
                                below_vals_count +=1
                            else:
                                above_vals_count +=1
                            y_list.append(_val)
                        # print("row: ",row)
                        # print(vals_list) 
            plt.figure("hist", figsize=(16,9))
            blobs_id = []
            
            x_list = list(x_list)
            y_list = list(y_list)

            # for i in range(35): blobs_id.append(i)
            # print("vals_list: ",vals_list)
            print("x_list: ",x_list)
            print("y_list: ",y_list)
            # plt.plot(x_list, y_list, color = 'g')
            plt.scatter(x_list,y_list, color = 'r', s= 45)
            plt.bar(x_list,y_list,width=0.4)

            plt.xlim(-1,36)
            plt.ylim(0,8)
            plt.xlabel('Blobs')
            plt.ylabel('Freq')
            # plt.hlines()

            for i in range(8):
                # print("--------",i)
                if i >= 5:
                    plt.hlines(i,0,36, colors = 'b', linestyles = '--',linewidth=2 )
                else:
                    plt.hlines(i,0,36, colors = 'b', linestyles = '--',linewidth=0.5 )
            plt.title(item+"\nless that or equal to 5--> "+str(above_vals_count)+"\nless than 5-->"+str(below_vals_count))
            plt.xticks(x_list)
            # plt.pause(0.1)

            plt.savefig(output_path+"/"+item+".png")
            # showImage(input_img, "input_img")
            # cv2.waitKey(0)
            # plt.show()
            plt.close()
    def check_table_exists(self):
        db_file_path = "/home/adminspin/Slides_data/Stitching_data/fp_in_stitching_error/JR_19_734_B10_1.db"
        conn = sqlite3.connect(db_file_path)
        c = conn.cursor()
        table_status = False                    
        #get the count of tables with the name
        try:
            c.execute("SELECT color_values FROM slide_characteristics")
            print("Tables found")

            c.execute("SELECT hue_1x FROM aoi")
            print("Tables found")
            table_status = True

        except sqlite3.OperationalError as msg:
            table_status = False
            print("Table doesnt exists", msg)

        # c.execute("SELECT count(name) FROM sqlite_master WHERE type=''+table+'' AND name=''+slide_characterist'')
        # c.execute("SELECT blob_index, focus_metric, color_metric, stack_index, hue_metric from acquisition_blob_info \
        #     WHERE aoi_name = '"+ aoi_name +"' and grid_id =="+ str(grid_id))
        #if the count is 1, then table exists
        
        # if c.fetchone()[0]==1 : 
        #     print('Table exists.')
        # else :
        #     print('Table does not exist.')
    def drawBlack(self):
        input_img = cv2.imread("/home/adminspin/Downloads/13081.250000_15881.250000_wo_black_13.jpeg")
        min_img = cv2.min(input_img[:,:,0], cv2.min(input_img[:,:,1], input_img[:,:,2]))
        max_img = cv2.max(input_img[:,:,0], cv2.max(input_img[:,:,1], input_img[:,:,2]))

        img1_temp = np.zeros(input_img[:, :, 0].shape, np.uint8)
        img2_temp = np.zeros(input_img[:, :, 0].shape, np.uint8)
        diff_img = np.zeros(input_img[:, :, 0].shape, np.uint8)
        diff_img = max_img - min_img

        # diff_img[diff_img < 25] = 255
        diff_img[diff_img < 255] = 0

        min_img[min_img < 50] = 255
        min_img[min_img < 255] = 0

        max_img[max_img < 120] = 255
        max_img[max_img < 255] = 0

        showImage(min_img, "min")
        showImage(max_img, "max")

        # diff_img = cv2.bitwise_and(diff_img, img1_temp) 

        contours, _ = cv2.findContours(np.uint8(max_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):

            if (cv2.contourArea(contours[i])< 50): continue
            cv2.drawContours(input_img, contours, i, (255,255,0), 2)
        showImage(input_img, "re")
        showImage(diff_img, "diff")
        cv2.imwrite("/home/adminspin/Downloads/13081.250000_15881.250000_wo_black_13_re.jpeg", input_img)
        cv2.waitKey(0)

    def getHueLabelsForRGBImages(self):
        path1 = "/home/adminspin/Pictures/0_For_Localization_Report/UNet/0_input_slides"
        path2 = "/home/adminspin/Pictures/0_For_Localization_Report/UNet"

        # path3 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Data_05_HSV_16Feb/1_2_biopsy_masks_from_model"

        out_path = "/home/adminspin/Pictures/0_For_Localization_Report/UNet/hue_labels"
        # out_path2 = "/home/adminspin/Slides_data/Onex_Images_Repository/DataSet_UNet/Data_05_HSV_16Feb/1_3_biopsy_masks_from_model_left_region"

        list_dir = os.listdir(path1)
        list_dir = sorted(list_dir)

        for item in list_dir[-1]:
            print(item)

            mask_path = os.path.join(path2, item)
            if os.path.exists(mask_path): mask_img = cv2.imread(os.path.join(path2, item),0)
            # mask_img2 = cv2.imread(os.path.join(path3, item),0)

            # mask_sub = mask_img2.copy()
            # mask_sub[mask_img > 150] = 0

            input_image = cv2.imread(os.path.join(path1, item))
            input_image = cv2.medianBlur(input_image, 3)        
            # Get HSV from BGR
            hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
            hue_image = np.uint8(hsv[:, :, 0])

            return_img = np.zeros(hue_image.shape)
            unique_intensities = np.unique(hue_image)
            for uni in unique_intensities:
                final_color = self.uintMax
                dist = self.uintMax
                _id = self.uintMax
                # Find the closest label from the given reference hue list
                for k in range(len(self.color_list)):
                    diff = abs(uni - self.color_list[k])
                    if diff < dist:
                        dist = diff
                        final_color = self.color_list[k]
                        _id = k
                nonzeroPoints = np.where(hue_image == uni)
                return_img[nonzeroPoints] = final_color

            return_img = cv2.medianBlur(np.uint8(return_img),3)
            
            hsv[:, :, 0] = return_img
            hsv[:, :, 1] = 150
            hsv[:, :, 2] = 255

            reverted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            if os.path.exists(mask_path): 
                reverted = cv2.bitwise_and(reverted, reverted, mask = mask_img)
            cv2.imwrite(os.path.join(out_path, item), reverted)
    def plotIntesityAcrossStack(self):
        input_path = "/datadrive/Color/Color_Metric_Analysis_only_validation/H01CBA04R_3790/validation/acquisition_debug_data/fs_stack/6818.750000_44662.500000"
        for item in os.listdir(input_path):
            img = cv2.imread(os.path.join(input_path, item),0)
            x = 1753+10
            y = 402+10
            h = 35
            w = 25
            cropped_img =  img[y:y+h, x:x+w]
            showImage(cropped_img, "cp")
            cv2.waitKey(0)
            print(item,"\t", cv2.mean(cropped_img)[0])
if __name__ == "__main__":
    Obj = Data()
    # Obj.createMask()
    # Obj.zeroPadding()
    # Obj.resizeImages()
    # Obj.split_data()
    # Obj.copy_image_withSameName()
    # Obj.copy_whiteCorrectedImages()
    # Obj.overlay_Mask_OnRGB()
    # Obj.gen_newMask_fromMasks()
    # Obj.getHueLabelsForRGBImages()
    # Obj.getHuePlots()
    # Obj.fillMasks()
    # Obj.contrastEnhancement()
    # Obj.get_IoU()
    # Obj.plotProbabilityMeanValues()
    # Obj.doThreshold()
    # Obj.getAccuracyMetrics()
    # Obj.two_images_withSameName_op()
    # Obj.copyTheImageToOriginalFolder()
    # Obj.copy_imagesWithInputNameList()
    # Obj.copySpecificImagesFromLocData()
    # Obj.image_type_conversion()
    Obj.populateFocusColorMetricValuesFromDB()
    # Obj.huePlotsForImages()
    # Obj.plot_data_from_csv()
    # Obj.separate_dark_nuclei_slides()
    # Obj.check_table_exists()
    # Obj.drawBlack()
    # Obj.copy_sampled_imgs_for_bestZ()
    #Obj.map_bubbles()
    # Obj.plotIntesityAcrossStack()