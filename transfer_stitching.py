import os, sys
from os.path import join
from multiprocessing import Pool
from os import walk
from glob import glob
import shutil
from os.path import exists
import json
import subprocess
import pandas as pd

df = pd.read_csv("/home/adminspin/Music/scripts/stitching.csv")


path1 = "/mnt/clusterNas/dicom_data"
path2 = "/datadrive/wsi_data/stitching_error"
user = "adminspin"
ip = "192.168.1.2"

# sshpass -p adminspin#123 scp -r -o StrictHostKeyChecking=no adminspin@192.168.1.2:/mnt/clusterNas/dicom_data/JR-20-1585-A3-1_H01BBB20P-23944/other.tar /home/adminspin/Music/sarvesh
for i,j in zip(df['_source.data.slide_name'],df['_source.data.cluster_name']):
    try:
        if j == "CS001":

            src_path = join(path1,i,"grid_1/grid_intermediate.tar")
            dst_path = join(path2,i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            command = "sshpass -p {} scp -r StrictHostKeyChecking=no {}@{}:{} {}".format("adminspin#123", user,"192.168.1.2",src_path, dst_path)
            print("**"*50,"CS001","**"*50)
            print(command)
            status = os.system(command)
        if j == "CS002":
            
            src_path = join(path1,i,"grid_1/grid_intermediate.tar")
            dst_path = join(path2,i)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            command = "sshpass -p {} scp -r StrictHostKeyChecking=no {}@{}:{} {}".format("adminspin#123", user,"192.168.1.3",src_path, dst_path)
            print("**"*50,"CS002","**"*50)
            print(command)
            status = os.system(command)
    except Exception as error:
        print("error :",error,"on ",i)


# if status == 0:
#     return True
# else:
#     return False