import cv2
import numpy as np
from PIL import Image
import os,glob
import sys
from unittest.mock import patch
import colour
from collections import defaultdict

def get_lab_value(_rgb):
        _rgb_image = np.zeros((1,1,3), dtype=np.uint8)
        _rgb_image[:,:,0] = _rgb[0]
        _rgb_image[:,:,1] = _rgb[1]
        _rgb_image[:,:,2] = _rgb[2]
        _lab_image = cv2.cvtColor(_rgb_image, cv2.COLOR_RGB2LAB)
        _l_mean = np.mean(_lab_image[:,:,0]) * (100/255)
        _a_mean = np.mean(_lab_image[:,:,1]) - 128
        _b_mean = np.mean(_lab_image[:,:,2]) - 128
        return [_l_mean, _a_mean, _b_mean]

def color_diff(RGB_ref,RGB_in):
    _lab_ref = get_lab_value(RGB_ref)
    _lab_in = get_lab_value(RGB_in)
    delta_E_CIE1994 = colour.difference.delta_E_CIE1994(_lab_ref, _lab_in)
    return delta_E_CIE1994



# input_path = glob.glob("/home/adminspin/Videos/basetils/trial/*.png")
def cropping(folder_path):

    for i in sorted(glob.glob(folder_path+"/*.jpeg"))[::-1]:
        dst_img = os.path.split(i)[0]+"/cropped"
        if not os.path.exists(dst_img):
            os.makedirs(dst_img)
        img_raw = cv2.imread(i)
        roi = cv2.selectROI(img_raw)
        roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        
        cv2.imwrite(dst_img+"/crop_"+i.split('/')[-1].split('.')[0]+".jpeg",roi_cropped)
        cv2.destroyAllWindows()


    list_img = os.listdir(dst_img)#iterating over dst_image to get the images as arrays

    lst = []
    for image in sorted(list_img):

        [file_name, ext] = os.path.splitext(image) #splitting file name from its extension
        arr = np.array(Image.open(os.path.join(dst_img, image))) #creating arrays for all the images
        [h, w] = np.shape(arr)[0:2]#calculating height and width for each image
        arr_dim = arr.ndim #calculating the dimension for each array
        arr_shape = arr.shape #calculating the shape for each array
        arr_mean = np.mean(arr, axis=(0,1))
        print([arr_mean[0],  arr_mean[1], arr_mean[2]])
        lst.append([arr_mean[0],  arr_mean[1], arr_mean[2]])

                
    print(lst)
    print("Color diff(22P-41) : ",color_diff(lst[0],lst[1]))
    print("Color diff(22P-42) : ",color_diff(lst[0],lst[2]))
    print("Color diff(22P-44) : ",color_diff(lst[0],lst[3]))
    print("Color diff(22P-45) : ",color_diff(lst[0],lst[4]))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    cropping(path)

    