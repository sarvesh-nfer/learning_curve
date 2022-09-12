#sarvesh
import cv2
import numpy as np
import os
import shutil 

lst = ["H01BBB17R-6272",
"H01BBB17R-6271",
"H-1461-22-B7--------_H01BBB17R-6270",
"H01BBB17R-6269",
"H01BBB17R-6268",
"H01BBB17R-6267",
"H-91-21-A10_H01BBB17R-6266",
"H01BBB17R-6265",
"H01BBB17R-6264",
"H-53-21-A1_H01BBB17R-6263",
"H01BBB17R-6262",
"H01BBB17R-6261",
"H01BBB17R-6260",
"H01BBB17R-6259",
"H-1461-22-A1--------_H01BBB17R-6258",
"H-1475-21-B1--------_H01BBB17R-6257",
"H01BBB17R-6256",
"H01BBB17R-6255",
"H01BBB17R-6254",
"H01BBB17R-6253",
"H01BBB17R-6252",
"H-149-21-A3_H01BBB17R-6251",
"H01BBB17R-6250",
"H01BBB17R-6249",
"H-030-21-A5--------_H01BBB17R-6248",
"H01BBB17R-6247",
"H01BBB17R-6246",
"H01BBB17R-6245",
"H-1656-21-A1_H01BBB17R-6244"]

input_path = "/mnt/clusterNas/dicom_data"

output_path = "/home/adminspin/Music/17R"

for slide_name in lst:
	print("name = ", slide_name)
	slide_path_input = os.path.join(input_path, slide_name,"other.tar")
	slide_path_output = os.path.join(output_path, slide_name)

	if not os.path.exists(slide_path_output):
		os.makedirs(slide_path_output)

		shutil.copy2(slide_path_input, slide_path_output)
	else:
		print("No white_correct Image for :", slide_name)