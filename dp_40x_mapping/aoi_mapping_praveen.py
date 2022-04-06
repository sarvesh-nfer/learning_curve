import sqlite3
import os
import sys
import cv2

from os.path import exists, join

def get_aoi_info(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    query = "SELECT aoi_x, aoi_y, bg_state_acq FROM aoi;"
    print(query)
    cursor.execute(query)

    aoi_data = cursor.fetchall()

    aoi_info_list = []

    for i in range(0, len(aoi_data)):
        x_pos = aoi_data[i][0]
        y_pos = aoi_data[i][1]
        bg_state = aoi_data[i][2]

        aoi_info_list.append([x_pos, y_pos, bg_state])

    return aoi_info_list

def map_aois_to_1x(aoi_info_list, onex_img):
    fov_width = aoi_info_list[1][0] - aoi_info_list[0][0]
    fov_height = aoi_info_list[1][1] - aoi_info_list[0][1]

    for i in range(0, len(aoi_info_list)):
        x_pos = int(aoi_info_list[i][0])
        y_pos = int(aoi_info_list[i][1])
        bg_state = aoi_info_list[i][2]
        x2 = int(x_pos + fov_width)
        y2 = int(y_pos + fov_height)

        if bg_state == 0:
            cv2.rectangle(onex_img, (x_pos, y_pos),
                        (x2, y2), (255, 0, 0), 2)
        else:
            cv2.rectangle(onex_img, (x_pos, y_pos),
                        (x2, y2), (0, 0, 255), 2)

    return onex_img

if __name__ == '__main__':
    slide_path = sys.argv[1]

    db_path = join(slide_path, slide_name + ".db")

    print("DB path: ", db_path)

    # Connect to db and get aoi data.
    aoi_info_list = get_aoi_info(db_path)

    # Map the aois to 1x image.
    onex_img_path = join(slide_path, "loc_output_data", "updatedInputImage.png")
    onex_img = cv2.imread(onex_img_path)

    mapped_img = map_aois_to_1x(aoi_info_list, onex_img)

    cv2.imwrite(slide_path + '/mapped_img.png', mapped_img)

    sys.exit(0)

