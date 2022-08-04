import shutil
import os,glob

dst = "/home/adminspin/Music/scripts/data"
for i in glob.glob("/nvme1_drive/postprocessing/banding_slides/JR-20-2847-D7-4_H01DBB35P-21354/*/111"):
    print(i)
    
    dst_path = os.path.join(dst,i.split("/")[-2])
    # if not os.path.exists(dst_path):
    #     os.makedirs(dst_path)
        
    shutil.copytree(i,dst_path)
