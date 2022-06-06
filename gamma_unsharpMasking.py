from genericpath import exists
import numpy as np
import cv2
from pylibdmtx.pylibdmtx import decode
from os.path import join
import os
import json
import tarfile
import re
import csv

def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)



def sharpness_gamma(input_image, strength = 0.65, gamma = 2.2):

    sharpness_enabled = 1
    # 1.Apply Gamma function to make the image sRGB complaint
    result = adjust_gamma(input_image, gamma)

    new_img = np.zeros(result.shape, dtype=np.uint8)

    if sharpness_enabled == 1:
       
        R_blur = cv2.medianBlur(result[:,:,2], 1)
        R = result[:,:,2] - cv2.Laplacian(R_blur, cv2.CV_32FC1)*strength

        G_blur = cv2.medianBlur(input_image[:,:,1], 1)
        G = result[:,:,1] - cv2.Laplacian(G_blur, cv2.CV_32FC1)*strength

        B_blur = cv2.medianBlur(input_image[:,:,0], 1)
        B = result[:,:,0] - cv2.Laplacian(B_blur, cv2.CV_32FC1)*strength

        R[R > 255] = 255
        G[G > 255] = 255
        B[B > 255] = 255

        R[R < 0] = 0
        G[G < 0] = 0
        B[B < 0] = 0

        new_img[:,:,0] = B
        new_img[:,:,1] = G
        new_img[:,:,2] = R
    else:

        new_img = result

    return new_img


def barcode_decoder(path):
    img = cv2.imread(path)
    img1 = sharpness_gamma(img)
    data_matrix_data = decode(img1)

    print('Raw output ', data_matrix_data)

    if len(data_matrix_data) > 0:
        data_matrix = (data_matrix_data[0].data)
        data_matrix_output = str(data_matrix)
        data_matrix_data = data_matrix_output.split("'")[1]
        print('data matrix ', data_matrix_data)
        return data_matrix_data
    else:
        return None

def check_special_characters(label_name):
    for character in label_name:
        if character.isalpha() is True or character.isdigit() is True:
            return False
    return True

def remove_special_character_from_beginning(label_name):

    special_character_count = 0

    for character in label_name:
        if character == "-" or character == "_":
            special_character_count += 1
        else:
            break
    
    return label_name[special_character_count:]

def validate_barcode(label_name):
    pattern = re.compile("[A-Za-z0-9_-]+")

    # if found match (entire string matches pattern)
    if pattern.fullmatch(label_name) is not None:
        return True
    else:
        return False

def update(barcode):
    if check_special_characters(barcode):
        return None
    
    data_matrix_output_final = ""
    for special_char in barcode:
        if special_char.isalpha() is False and \
            special_char.isdigit() is False:
            special_char = "-"
        data_matrix_output_final = data_matrix_output_final + special_char
    
    data_matrix_output_final =\
          remove_special_character_from_beginning(data_matrix_output_final)
    
    status = validate_barcode(data_matrix_output_final)

    if status == False:
        return None
    
    return data_matrix_output_final


if __name__ == "__main__":
    data_input = "/home/adminspin/wsi_app/msk"
    file_path  = "/home/adminspin/wsi_app/msk/name.txt"
    output_path = "/home/adminspin/wsi_app/msk/ouptut"
    json_out_path = join(output_path, "output.json")
    csv_path = join(output_path, "status.csv")

    lines = []
    json_output = []
    csv_output = []
    
    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split("\n")[0]
        slide_path = join(data_input, line)
        data  = ["", "", "", ""]

        try:

            output_name = join(output_path, line)
            if exists(output_name) is False:
                os.mkdir(output_name)

            names = tarfile.open(join(slide_path, "other.tar")).getnames()
            
            if "loc_output_data/barcodeImage.png" in names:
                cmd1 = "tar xvf " + join(slide_path, "other.tar") + " -C " + output_name \
                        + " " + "loc_output_data/barcodeImage.png"
                status = os.system(cmd1)

            if "loc_output_data/barcodeImage_yellow.png" in names:
                cmd2 = "tar xvf " + join(slide_path, "other.tar") + " -C " + output_name \
                        + " " + "loc_output_data/barcodeImage_yellow.png"
                status = os.system(cmd2)

            output = None
            path = join(output_name,"loc_output_data/barcodeImage_yellow.png")
            if exists(path):
                output = barcode_decoder(path)
            else:
                print("File Not availbale [YELLOW]", line)
                data[2] = "NA"

            if output is None:
                path = join(output_name,"loc_output_data/barcodeImage.png")
                if exists(path):
                    output = barcode_decoder(path)
                else:
                    print("File Not availbale [NORMAL]", line)
                    data[3] = "NA"

            if output is not None:
                barcode = update(output)
                new_slide_name = None
                if barcode is not None:
                    new_slide_name = barcode + "_" + line
                
                json_name = {
                    "prev_slide_name" : line,
                    "new_slide_name" : new_slide_name
                }

                json_output.append(json_name)
                data[0] = line
                data[1] = "decoded"
       
            else:
                print("Unable to detect the barcode: ", line)
                data[0] = line
                data[1] = "Not Decoded"

        except Exception as msg:
            print(msg)
            print("Unable to run it because of the exception ", line) 

        csv_output.append(data)   

    json_object = json.dumps(json_output, indent=2)
    with open(json_out_path, 'w') as json_file:
        json_file.write(json_object)
        json_file.close()
    
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow(csv_output)