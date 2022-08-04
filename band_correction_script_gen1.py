import cv2
import numpy as np
import glob,os
import sys
import time
import concurrent.futures

def band_correction_v2(image, level=6):
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    cond = ((b >= (230-level)) & (g >= (230-level)) & (r >= (230-level)))         #check for dip
    image[cond] = 230, 230, 230
    return image

def band_correction(image, offset_level=6):
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    cond1 = ((b >= 230) | (g >= 230) | (r >= 230))
    cond2 = ((b >= (230 - offset_level)) & (g >= (230 - offset_level)) & (r >= (230 - offset_level)))
    image[cond1 & cond2] = 230, 230, 230
    return image

def process_tile(filename, input_tiles_path, output_tiles_path):
    image = cv2.imread(filename)
    image = band_correction_v2(image, offset_level=6)

    # out_filename = filename.replace(input_tiles_path, output_tiles_path)
    out_filename = os.path.join(output_tiles_path+filename)
    cv2.imwrite(out_filename, image)



def run_correction_pyramidal(scan_path):
    input_tiles_path = scan_path + "/grid_merged/base_tiles/000_files/17"
    output_tiles_path = scan_path + "/grid_merged/bands_corrected/17"

    file_list = glob.glob(input_tiles_path + "/*.jpeg")
    t1 = cv2.getTickCount()
    for indx, filename in enumerate(file_list):
        process_tile(filename, input_tiles_path, output_tiles_path)

    t2 = cv2.getTickCount()
    t= (t2-t1)/cv2.getTickFrequency()
    print("time spent : ", t, " sec")

def run_correction_pyramidal_mp(scan_path):
    input_tiles_path = scan_path + "/grid_merged/base_tiles/000_files/17"
    output_tiles_path = scan_path + "/grid_merged/bands_corrected/17"

    file_list = glob.glob(input_tiles_path + "/*.jpeg")

    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor: #ThreadPoolExecutor #ProcessPoolExecutor
        pool = [executor.submit(process_tile, filename, input_tiles_path, output_tiles_path) for filename in file_list]
        for i in concurrent.futures.as_completed(pool):
            continue

    end = time.perf_counter()
    print(f'time spent {round(end - start, 2)} second(s)')




if __name__ == "__main__":
    #print("single process excecution :")
    #run_correction_pyramidal(scan_path="/home/adminspin/data/wsi/JR-20-3862-B5-2_H01BBB23P-56740")

    #print("multiprocess excecution :")
    run_correction_pyramidal_mp(scan_path="/home/adminspin/data/wsi/JR-20-3862-B5-2_H01BBB23P-56740")

