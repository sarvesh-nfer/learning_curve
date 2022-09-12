import cv2
import sqlite3
import os, sys

# |----------------------------------------------------------------------------|
# showImage
# |----------------------------------------------------------------------------|
def showImage(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 600)
    cv2.imshow(name, image)
    cv2.waitKey(0)
# |----------------------End of showImage---------------------------|

class AoIInfo():
    '''
    Structure for storing AoI information
    '''
# |----------------------------------------------------------------------------|
# Class Variables
# |----------------------------------------------------------------------------|
#    no class variables

# |----------------------------------------------------------------------------|
# Constructor
# |----------------------------------------------------------------------------|
    def __init__(self):
        self.id = 0
        self.name = ""
        self.xPos = -1
        self.yPos = -1
        self.x_mic = -1
        self.y_mic = -1
        self.zValue = -1
        self.captureState = 0
        self.rowIdx = 0
        self.colIdx = 0
        self.slide_row_idx = 0
        self.slide_col_idx = 0
        self.focusMetric = 0
        self.regionType = -1
        self.wbcPresent = 0
        self.path = ""

        self.sampling_z = 0
        self.bg_state_fs = 0
        self.bg_state_acq = 0
        self.bg_state_nlmd = 0

        self.color_metric = 0
        self.best_idx = 0
        self.ref_z = -1
        self.best_z = -1
        self.image_path = ""
        self.stack_path = ""
        self.capture_z_stack = False
        self.best_image = None
        self.image_stack = []
        self.hue_1x = 0

        self.bg_intensity = -1
        self.bg_state_high_power = 0

        self.annotation_info = 0
        self.coverslip_info = 0

        self.aoi_class = -1
        self.replace_with_white = 0
        self.localization_blob_index = -1

        self.width = 0
        self.height = 0
        self.width_pix = 0
        self.height_pix = 0
        self.ima_row_idx = 0
        self.ima_col_idx = 0
        self.potential_annotation = 0
        self.is_out_of_focus = 0
        self.annotation_presence = 0


# |----------------------------End of Constructor-----------------------------|


def get_imaging_area_aoi_info(db_file_path):
    aoi_list = []
    dbconn_sqlite = sqlite3.connect(db_file_path)
    cursor = dbconn_sqlite.cursor()
    query = ("SELECT * FROM aoi")
    cursor.execute(query)
    records = cursor.fetchall()
    # query = "SELECT row_count, column_count FROM grid_info where grid_id =1"
    # cursor.execute(query)
    # row_count, column_count = cursor.fetchall()[0]

    for row in records:
        aoi_info = AoIInfo()

        aoi_info.id = row[0]
        aoi_info.name = row[1]
        aoi_info.xPos = row[2]
        aoi_info.yPos = row[3]
        aoi_info.x_mic = row[4]
        aoi_info.y_mic = row[5]
        aoi_info.rowIdx = row[35]
        aoi_info.colIdx = row[36]
        aoi_info.aoi_class = row[8]
        aoi_info.sampling_z = row[9]
        aoi_info.zValue = row[10]
        aoi_info.bg_state_fs = row[11]
        aoi_info.bg_state_acq = row[12]
        aoi_info.bg_state_nlmd = row[13]
        aoi_info.potential_annotation = row[55]
        aoi_info.is_out_of_focus = row[34]
        aoi_info.annotation_presence = row[40]



        aoi_list.append(aoi_info)
    return aoi_list

def read_aoi_names(white_gen_txt):
    aoi_name_lst = []
    all_aoi = []
    with open(white_gen_txt) as file:
            print(white_gen_txt)
            data_to_parse = []
            for line in file:
                data_to_parse.append(line)
            for line in data_to_parse:
                if "Aoi path:" in line:
                    all_aoi.append(line.split("//")[-1].split(".")[0].strip())

                if 'Aoi used:' in line:
                    name = line.split(" ")[-1].split(".")[-2].replace('\n','')
                    aoi_name_lst.append(name)
            data_to_parse = []
    return aoi_name_lst,all_aoi


def plot_aoi_on_image(aoi_info_list, updated_image, white_gen_txt):
    for aoi_info in aoi_info_list:
        try:

            aoi_name_lst,all_aoi = read_aoi_names(white_gen_txt)
            aoi_name = aoi_info.name
            x_pos = int(aoi_info.xPos)
            y_pos = int(aoi_info.yPos)
            width_val = int(aoi_info.width_pix)
            height_val = int(aoi_info.height_pix)
            annotation = int(aoi_info.annotation_presence)
            oof = int(aoi_info.is_out_of_focus)
            
            for j in all_aoi:
                print(j)
                if aoi_name == j :
                    cv2.rectangle(updated_image, (x_pos, y_pos), (x_pos+width_val, y_pos+height_val),(0, 0, 255), 6)
            
            for i in aoi_name_lst:
                if aoi_name == i :
                    cv2.rectangle(updated_image, (x_pos, y_pos), (x_pos+width_val, y_pos+height_val),(0, 255, 0), 6)
            # if aoi_name == "aoi5475":
            #     cv2.rectangle(updated_image, (x_pos, y_pos), (x_pos+width_val, y_pos+height_val),(0, 0, 0), 3)
            # if aoi_name == "aoi5404":
            #     cv2.rectangle(updated_image, (x_pos, y_pos), (x_pos+width_val, y_pos+height_val),(0, 0, 0), 3)
            # if aoi_name == "aoi5477":
            #     cv2.rectangle(updated_image, (x_pos, y_pos), (x_pos+width_val, y_pos+height_val),(0, 0, 0), 3)
            # if aoi_name == "aoi5500":
            #     cv2.rectangle(updated_image, (x_pos, y_pos), (x_pos+width_val, y_pos+height_val),(0, 0, 0), 3)
        except:
            continue
    return updated_image

def main():
    if len(sys.argv) == 2:
        debug_path = sys.argv[1]

        for slide in os.listdir(debug_path):
            print(slide)
            db_file_path = os.path.join(debug_path, slide, slide + ".db")
            input_image = cv2.imread(os.path.join(debug_path, slide, 'loc_output_data', "updatedInputImage.png"))
            white_gen_txt = os.path.join(debug_path, slide, "white_generation_output.txt")
            aoi_list = get_imaging_area_aoi_info(db_file_path)

            aoi_info_image = plot_aoi_on_image(aoi_list, input_image, white_gen_txt)
            cv2.imwrite(debug_path +"/" + slide + "/aoi_info_image.png", aoi_info_image)

if __name__ == '__main__':
    main()