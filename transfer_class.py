import os,glob
import sys
import shutil

txt = glob.glob("/home/adminspin/wsi_app/acquired_data/*/*/class/*")
path = "/datadrive/wsi_data/data_for_validation"

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

for i in txt:
    info = i.split("/")[-1]
    slide = i.split("/")[-4]
    grid = i.split("/")[-3]
    try:
        output_path = os.path.join(path,slide,grid,"class",info)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print("COPYING : ",i)
        copytree(i,output_path)
        print("COPIED : ",i)
        print("**"*50)
    except Exception as msg:
        print("msg : ",msg)