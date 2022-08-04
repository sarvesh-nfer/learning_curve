import concurrent.futures
import glob
import os
import cv2
import sys
import requests
from shutil import copyfile

import json
import numpy as np
from math import ceil
import time
from pydicom.encaps import decode_data_sequence
from pydicom import dcmread
from banding_detection_metric import BandingDetection


def generate_base_tiles(_input_path, slide_id, output_path, build_path):

    if not os.path.exists(output_path):
        cmd = "mkdir -p " + str(output_path)
        try:
            os.system(cmd)
        except Exception as err:
            print("exception while creating folder. Err = " + str(err))

    input_path = os.path.join(_input_path, slide_id + "_level-0.dcm")
    dicom = dcmread(input_path)

    tile_height = dicom[0x0028, 0x0010].value
    tile_width = dicom[0x0028, 0x0011].value

    print("Tile width: ", tile_width)
    print("Tile Height: ", tile_height)

    frame_count = dicom[0x0028, 0x0008].value
    print("Number of frames: ", frame_count)
    bits = dicom[0x0028, 0x0100].value
    pixels = dicom[0x7fe0, 0x0010].value
    frames = decode_data_sequence(pixels)
    rows = ceil(dicom[0x0048, 0x0007].value / tile_height)
    cols = ceil(dicom[0x0048, 0x0006].value / tile_width)
    samples_per_pixel = dicom[0x0028, 0x0002].value

    itr, r, c = 0, 0, 0

    if bits == 8:
        bits_type = np.uint8
    elif bits == 16:
        bits_type = np.uint16
    elif bits == 32:
        bits_type = np.uint32

    while (itr < frame_count):
        # just write the encoded data to the disk.
        with open(output_path + "/" + str(c) + "_" + str(r) + ".jpeg", "wb") as file:
            file.write(frames[itr])

        if (c < cols - 1):
            c += 1

        else:
            r += 1
            c = 0

        itr += 1

def run_pyramid_generation(build_path, output_path):
    try:
        os.chdir(build_path)
        os.system("./pyramid_generation " + output_path)

    except Exception as err:
        print("Error while generate pyramid. Err = " + str(err))



def pull_data_from_nas(slide_id):

    cluster_nas_path = "/mnt/clusterNas/dicom_data"
    ssd_path = "/ssd_drive/restored_data"
    # pull the data from the NAS.
    dst_path = os.path.join(ssd_path, slide_id)

    slide_path = os.path.join(cluster_nas_path, slide_id, slide_id + ".zip")

    cmd = "cp -r " + str(slide_path) + " " + str(dst_path)
    try:
        os.system(cmd)
    except Exception as err:
        print("Error while copying data from source to destination. Err = " + str(err))

    os.chdir(dst_path)
    # now unzip the data.
    cmd = "unzip " + str(os.path.join(dst_path, slide_id + ".zip"))
    try:
        os.system(cmd)
        # now delete the zip file from ssd drive.
        os.remove(os.path.join(dst_path, slide_id + ".zip"))
    except Exception as err:
        print("Error while unzipping zip file. Err = " + str(err))

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
    image = band_correction(image, offset_level=6)

    out_filename = os.path.join(output_tiles_path+filename)
    cv2.imwrite(out_filename, image)

def run_correction_pyramidal_mp(path):
    input_tiles_path = path
    output_tiles_path = path

    file_list = glob.glob(input_tiles_path + "/*.jpeg")

    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor: #ThreadPoolExecutor #ProcessPoolExecutor
        pool = [executor.submit(process_tile, filename, input_tiles_path, output_tiles_path) for filename in file_list]
        for i in concurrent.futures.as_completed(pool):
            continue

    end = time.perf_counter()
    print(f'time spent {round(end - start, 2)} second(s)')

def get_slide_ids_from_text_file(path):

    slide_list = ""
    with open(path, "r") as file:
        slide_list = file.read()
    slides = slide_list.split("\n")
    return slides

# this method will copy the banding images to the intermediate
# image folder
def copy_banding_images(slide_id):
    print("Copying banding intermediate images.")
    input_path = "/ssd_drive/restored_data"
    banding_img_path = os.path.join(input_path,
                            slide_id,
                            "banding_intermediates_corrected")

    img_list = ["concatenated_image.png", "gamma_corrected.png", "fft_image.png", "considered_maximas.png", "enhanced_image.png"]
    INTERMEDIATE_IMG_PATH = "/hdd_drive/intermediate_data/"

    img_dst_path = os.path.join(INTERMEDIATE_IMG_PATH,
                        slide_id,
                        "acq_intermediate_images")

    if not os.path.exists(img_dst_path):
        cmd = "mkdir -p " + str(img_dst_path)
        os.system(cmd)

    try:
        # now copy all files.
        for img in img_list:
            img_path = os.path.join(banding_img_path, img)
            dst_path = os.path.join(img_dst_path, img)
            if os.path.exists(dst_path):
                cmd = "rm -rf " + str(dst_path)
                os.system(cmd)

            # first check the path.
            if os.path.join(img_path):
                # if exists copy the image.
                copyfile(img_path, dst_path)
            else:
                print("Banding debug image not present. Path = " + str(img_path))

    except Exception as err:
        print("Error while copying banding images to the destination. Err = " + str(err))

def get_system_dns():
    if not os.path.exists("/wsi_app/security/dns_info.json"):
        print("System dns file is missing. Running service on localhost")
        return "localhost"

    try:
        with open("/wsi_app/security/dns_info.json", "r") as file:
            info = json.loads(file.read())
            return info["system"]["cert_path"], info["system"]["key_path"], info["system"]["dns"]

    except Exception as err:
        print("Error while getting system DNS from file. Err = " + str(err))
        return None, "localhost"

def post_request_node(slide_id, grade):
    cert, key, dns = get_system_dns()
    api_endpoint = "restore/status/" + slide_id
    url = "https://{}:{}/{}".format(dns, 1337, api_endpoint)

    data = {"status": "completed",
                    "error_info": "",
                    "error_code": "",
                    "banding_grade": grade,
                    "slide_id": slide_id,
                    "timings": ""}

    res = requests.post(url, params="", data=json.dumps(data),
                        timeout=20, verify=cert)

    copy_banding_images(slide_id)

if __name__ == "__main__":

    slides = get_slide_ids_from_text_file("slide_names")

    input_path = "/ssd_drive/restored_data"
    cluster_nas_path = "/mnt/clusterNas/dicom_data"

    m_band = BandingDetection()
    m_band.default_path = input_path
    gamma_values = [2, 4, 6]
    banding_grade = {"2": 3,
                     "4": 2,
                     "6": 1,
                     "0": 0}

    counter = 0
    for slide_id in slides:
        # check slide path exists or not.
        if not os.path.exists(os.path.join(cluster_nas_path, slide_id, slide_id + ".zip")):
            continue

        if os.path.exists(os.path.join(input_path,slide_id)):
            cmd = "rm -rf " + str(os.path.join(input_path, slide_id))
            os.system(cmd)

        if os.path.exists(os.path.join(input_path, slide_id, "grid_merged")):
            cmd = "rm -rf " + os.path.join(input_path, slide_id, "grid_merged")
            os.system(cmd)

        if slide_id == "":
            continue

        output_path = os.path.join(input_path, slide_id, "grid_merged/base_tiles/000")
        if not os.path.exists(output_path):
            cmd = "mkdir -p " + str(output_path)
            try:
                os.system(cmd)
            except Exception as err:
                print("exception while creating folder. Err = " + str(err))

        build_path = "/home/adminspin/wsi_app/libs"
        pull_data_from_nas(slide_id)
        generate_base_tiles(os.path.join(input_path, slide_id, slide_id), slide_id, output_path, build_path)
        print("running banding correction for slide = " + str(slide_id))
        start = time.time()
        try:
            run_correction_pyramidal_mp(output_path)
        except Exception as err:
            print("Error while running banding correction. Err = " + str(err))
            continue
        print("Time taken for banding correction = " + str(time.time() - start))

        # now run pyr generation
        run_pyramid_generation(build_path, output_path)

        try:
            temp_path = os.path.join(input_path,slide_id, "grid_merged/base_tiles/000_files")

            banding_presence = 0
            grade = 0
            for gamma in gamma_values:

                banding_presence = m_band.process_pipeline(temp_path,
                                                           slide_id,
                                                           gamma)
                # banding is present
                if banding_presence:
                    path = os.path.join(input_path, "result_banding_correction.json")
                    # if path doesn't exists.
                    if not os.path.exists(path):
                        with open(path, "w") as file:
                            file.write(json.dumps({}))

                    # print(path)
                    _d = {}
                    with open(path, "r") as file:
                        _d = json.loads(file.read())
                        grade = banding_grade[str(gamma)]
                        _d[slide_id] = {"banding_grade": grade, "is_banding_present": grade > 2}

                    with open(path, "w") as file:
                        file.write(json.dumps(_d))

                    break

            if not banding_presence:
                path = os.path.join(input_path, "result_banding_correction.json")
                # if path doesn't exists.
                if not os.path.exists(path):
                    with open(path, "w") as file:
                        file.write(json.dumps({}))

                # print(path)
                _d = {}
                with open(path, "r") as file:
                    _d = json.loads(file.read())
                    grade = 0
                    _d[slide_id] = {"banding_grade": grade, "is_banding_present": grade > 2}

                with open(path, "w") as file:
                    file.write(json.dumps(_d))

            cmd = "rm -rf " + os.path.join(input_path, slide_id, slide_id)
            os.system(cmd)

            if counter < 200:
                post_request_node(slide_id, grade)
            else:
                cmd = "rm -rf " + os.path.join(input_path, slide_id, "grid_merged/base_tiles/000_files")
                os.system(cmd)

            counter += 1
        except Exception as msg:
            print("Exception, ", msg)
