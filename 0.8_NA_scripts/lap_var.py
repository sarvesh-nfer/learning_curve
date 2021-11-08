# import the necessary packages
from imutils import paths
import argparse
import cv2
import os,glob
import pandas as pd

lst1 = []
lst2 = []
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

save_path = "/home/adminspin/wsi_app/acquired_data/H01FBA08R_2441/grid_2/blur"
if not os.path.exists(save_path):
	os.mkdir(save_path)
txt = glob.glob("/home/adminspin/wsi_app/acquired_data/H01FBA08R_2441/grid_2/raw_images/*.bmp")
# print(txt)
for i in txt:
    image = cv2.imread(i)
    # print(i)
    gray = image[:,:,1]
    fm = variance_of_laplacian(gray)
    print(i," : ",fm)
    lst1.append(i.split('/')[-1].split('.')[0])
    lst2.append(fm)

    if fm < 6:
        cv2.putText(image, "{:.2f}".format(fm), (250, 750),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 50)
        cv2.imshow("Image", image)
        a = os.path.join(save_path,i.split('/')[-1])
        print(a)
        cv2.imwrite(a,image)
    else:
        cv2.putText(image, "{:.2f}".format(fm), (250, 750),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 50)
        cv2.imshow("Image", image)
        a = os.path.join(save_path,i.split('/')[-1])
        cv2.imwrite(a,image)


df = pd.DataFrame(list(zip(lst1, lst2)),columns =['aoi_name', 'variance'])
df.to_csv('/home/adminspin/Desktop/aoi_vari2129.csv',index=False)