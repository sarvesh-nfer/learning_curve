import os, sys
from os.path import join
from multiprocessing import Pool
from os import walk
from glob import glob
import shutil
from os.path import exists
import json
import subprocess

def transfer_other(slide,dst_path):

    c1 = ['H01BBB18P','H01BBB20P','H01BBB22P','H01BBB16P']
    c2 = ['H01BBB19P','H01BBB25P','H01BBB24P','H01BBB23P','H01JBA21P']
    c3 = ['H01BBB26P','H01BBB28P','H01BBB27P','H01BBB30P']
    c4 = ['H01DBB34P','H01DBB36P','H01BBB31P','H01DBB35P']

    op = slide.split("_")[-1].split("-")[0]
    if op in c1:
        src_path = join("/mnt/clusterNas/dicom_data",slide)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        command = "sshpass -p {} scp -r StrictHostKeyChecking=no {}@{}:{} {}".format("adminspin#123", "adminspin","192.168.1.2",src_path, dst_path)
        print("**"*50,"CS001","**"*50)
        print(command)
        status = os.system(command)
    
    if op in c2:
        src_path = join("/mnt/clusterNas/dicom_data",slide)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        command = "sshpass -p {} scp -r StrictHostKeyChecking=no {}@{}:{} {}".format("adminspin#123", "adminspin","192.168.1.3",src_path, dst_path)
        print("**"*50,"CS002","**"*50)
        print(command)
        status = os.system(command)
    
    if op in c3:
        src_path = join("/mnt/clusterNas/dicom_data",slide)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        command = "sshpass -p {} scp -r StrictHostKeyChecking=no {}@{}:{} {}".format("adminspin#123", "adminspin","192.168.1.4",src_path, dst_path)
        print("**"*50,"CS003","**"*50)
        print(command)
        status = os.system(command)
    
    if op in c4:
        src_path = join("/mnt/clusterNas/dicom_data",slide)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        command = "sshpass -p {} scp -r StrictHostKeyChecking=no {}@{}:{} {}".format("adminspin#123", "adminspin","192.168.1.5",src_path, dst_path)
        print("**"*50,"CS004","**"*50)
        print(command)
        status = os.system(command)




# def transfer_grid():

# def tranfer_both():
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        file_path = sys.argv[1]
        dst_path = sys.argv[2]
        # op = sys.argv[3]
    else:
        print("Pass correct arguments !!!")

    dst_path = sys.argv[2]
    file1 = open(file_path, "r+")

    print("Output of Readline function is ")
    slides = file1.readlines()

    for slide in slides:
        try:
            transfer_other(slide,dst_path)
        except Exception as msg:
            print("error : ",slide)

