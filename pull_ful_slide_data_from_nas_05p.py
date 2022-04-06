import os, sys
from os.path import join
from multiprocessing import Pool
from os import walk
from glob import glob
import shutil
from os.path import exists
import json
import subprocess

# |----------------------------------------------------------------------------|
# utar_data
# |----------------------------------------------------------------------------|
def utar_data(src, dst):
    status = True

    if exists(src):
        utar_cmd = "tar -xf {} -C {}".format(src, dst)

        if exists(src) is True:            
            status = os.system(utar_cmd)

            if status != 0:
                status = False

    return status

# |------------------------End of utar_data-----------------------------------|


# |----------------------------------------------------------------------------|
# read_json
# |----------------------------------------------------------------------------|
def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.loads(file.read())
        return data
    except Exception as msg:
        print("Error in reading json: {}".format(msg))
        return None

# |----------------------End of read_json-------------------------------------|


# |----------------------------------------------------------------------------|
# _get_grids
# |----------------------------------------------------------------------------|
def _get_grids(slide_path):
    # Read metadata_json and get grids.
    metadata_json_path = join(slide_path, "metadata.json") 
    metadata_json = read_json(metadata_json_path)

    grids = []
    for grid_obj in metadata_json["data"]["grid_info"]:
        if grid_obj["grid_name"] != "grid_merged":
            grids.append(grid_obj["grid_name"])
    
    print("Grids: {}".format(grids))
    
    return grids

# |---------------------------End of _get_grids-------------------------------|

# |----------------------------------------------------------------------------|
# delete_folder
# |----------------------------------------------------------------------------|
def delete_folder(folder_path):
    status = True

    if exists(folder_path):
        print("Dleteting Folder: {}".format(folder_path))

        try:
            command = "rm -rf {}".format(folder_path)
            cmd_status = os.system(command)
            print("Delete Folder status: {}".format(status))
            if cmd_status != 0:
                status = False
            print("Exiting delete folder")
        except Exception as error_msg:
            status = False
            print("Exception while deleting: {}".format(error_msg))
    else:
        status = False
        print("No such directory: {}".format(folder_path))
    
    return status

# |-------------------------End of delete_folder------------------------------|


# |----------------------------------------------------------------------------|
# create_folders
# |----------------------------------------------------------------------------|
def create_folders(path):
    if not exists(path):
        os.makedirs(path)
# |------------------------End of create_folders------------------------------|


# |----------------------------------------------------------------------------|
# run_decompression
# |----------------------------------------------------------------------------|
def run_decompression(slide_name):
    status = True
    
    grids = []
    
    try:
        dicom_genaration_path = "/home/adminspin/wsi_app/acquired_data"
        libs_path = join("/home", "adminspin", "wsi_app", "libs")
        _processes = 4

        slide_path = join(dicom_genaration_path, slide_name)
        # Untar other others.tar for metadata.json
        other_tar_path = join(slide_path, "other.tar")
        utar_data(other_tar_path, slide_path)

        grids = _get_grids(slide_path)

        for grid_name in grids: 
            grid_path = join(slide_path, grid_name)
            grid_intermediate_tar_path = join(grid_path, "grid_intermediate.tar")
            
            # untar grid data
            utar_data(grid_intermediate_tar_path, grid_path)
        
            compressed_data_folder = join(grid_path,
                                            "tmp")
            raw_data_folder = join(grid_path, "raw_images")
            print("Raw image folder: {}".\
                                            format(raw_data_folder))
            
            if exists(compressed_data_folder):
                delete_folder(compressed_data_folder)

            create_folders(compressed_data_folder)
            
            if exists(raw_data_folder):
                delete_folder(raw_data_folder)

            create_folders(raw_data_folder)
            
            blob_folder_path = join(grid_path, "blobs",
                                    "blobs.tar")
            
            # first un tar the raw_images.
            utar_status = utar_data(blob_folder_path,
                                    compressed_data_folder)
            
            if utar_status is False:
                return utar_status

            print("Raw images folder: {}".\
                                            format(raw_data_folder))

            # Now go for decompression.
            decompression_exe = join(libs_path, "ut_decompression")
            decompression_cmd = "{} {} {} {} {}".format(decompression_exe,
                                                        compressed_data_folder,
                                                        raw_data_folder,
                                                        _processes, 1)
            print("Decompression command: {}".\
                                            format(decompression_cmd))
            os.system(decompression_cmd)
#                 Now for decompression.
            # process = subprocess.Popen([decompression_cmd],
            #                             stdout=subprocess.PIPE,
            #                             shell=True)
            # process.wait()

            # delete compressed data
            delete_folder(compressed_data_folder)

        print("Decompression completed for slide: {}".format(slide_name))
    except Exception as err_msg:
        status = False
        print(err_msg)

    return grids
    
# |-----------------------End of run_decompression----------------------------|


# |----------------------------------------------------------------------------|
# move_slide_folder
# |----------------------------------------------------------------------------|
def move_slide_folder(src_path, dst_path, ip, transfer_type="rsync",
                      user="adminspin", reverse=False, pwd="adminSpin#123"):
    move_status = False

    if reverse is False:
        if transfer_type == "rsync":
            command = "sshpass -p 'adminspin#123' rsync -e 'ssh -o "\
                "StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' "\
                "-a {} {}@{}:{}".format(src_path, user, ip, dst_path)
        else:
            command = "sshpass -p '{}' scp -r {} {}@{}:{}".\
                format(pwd, src_path, user, ip, dst_path)
    else:
        if transfer_type == "rsync":
            command = "sshpass -p 'adminSpin#123' rsync -e 'ssh -o "\
                "StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' "\
                "-a {}@{}:{} {}".format(user, ip, src_path, dst_path)
        else:
            command = "sshpass -p '{}' scp -r {}@{}:{} {}".\
                format(pwd, user, ip, src_path, dst_path)
        
    print("Move command: {}".format(command))

    status = os.system(command)
    
    if status == 0:
        return True
    else:
        return False
    
# |--------------------------End of move_slide_folder------------------------|

# |----------------------------------------------------------------------------|
# tar_command
# |----------------------------------------------------------------------------|
def tar_command(name,data_list,path):
    os.chdir(path)
    status =123
    str1 = " "
    tarcmd = 'tar -cf {}.tar {}'.format(name,str1.join(data_list))
    #print('tarcmd',tarcmd)
    print("rsync command: {}".format(tarcmd))
    status = os.system(tarcmd)
    os.system('rm -rf {}'.format(str1.join(data_list)))

    return status

# |----------------------End of tar_command-------------------------------|


# |----------------------------------------------------------------------------|
# tar_grid_folder
# |----------------------------------------------------------------------------|
def tar_grid_folder(grid_path):
    #print(filelist)
    gridlist =[]
    otherlist =[]
    stacklist = []
    bloblist = []
    fusebloblist = []
    #print(grid)
    grid_path = join(slide_path,grid)
    grid_list = os.listdir(grid_path)
    if exists(join(grid_path,"blobs")) is True:
        grid_list.remove('blobs')
        bloblist = os.listdir(join(grid_path,"blobs"))
        status = tar_command('blobs',bloblist,join(grid_path,"blobs"))
    if exists(join(grid_path,"fused_blobs")) is True:
        grid_list.remove('fused_blobs')
        fusebloblist = os.listdir(join(grid_path,"fused_blobs"))
        status = tar_command('fused_blobs',fusebloblist,join(grid_path,"fused_blobs"))
    if exists(join(grid_path,"acquisition_debug_data")) is True:
        grid_list.remove('acquisition_debug_data')
        stacklist.append('acquisition_debug_data')
        status = tar_command('acquisition_debug_data',stacklist,grid_path)
    if exists(join(slide_path,"grid_merged")) is True:
        status = tar_command('grid_merged',os.path.join(slide_path,"grid_merged"))

    #print(grid_list)
    status = tar_command('grid_intermediate',grid_list,grid_path)
    #print('*****************************************')

    #print('grid -->',gridlist)

    # os.chdir(slide_path)
    # otherlist.remove(slide_name+'.jpeg')
    # #print('other-->',otherlist)
    # status = tar_command('other',otherlist,slide_path)

    return status
# |----------------------End of tar_grid_folder-------------------------------|

# |----------------------------------------------------------------------------|
# write_json
# |----------------------------------------------------------------------------|
def write_json(file_path, data):
    try:
        with open(file_path, 'w') as file:
            js = json.dumps(data, sort_keys=True, indent=2)
            file.write(js)
        file.close()
        return True
    except Exception as msg:
        print("Error in writing json: ", msg)
        return False
# |-------------------------End of write_json---------------------------------|


if __name__ == "__main__":
    if len(sys.argv) == 3:
        file_path = sys.argv[1]
        dst_path_1 = sys.argv[2]
    else:
        print("Pass correct arguments !!!")

#     nas_path = "/datadrive/wsi_data/compressed_data"
#     nas_ip = "14.140.231.202"
    nas_path = "/volume1/dicom_data"
    nas_ip = "10.20.0.10"
    # dst_path_1 = "/datadrive/wsi_data/Analysis_Acq_pipeline/System_Testing"
    file1 = open(file_path, "r+")

    print("Output of Readline function is ")
    slides = file1.readlines()

    main_json = {
            "data": []
    }

    for slide in slides:
        print("\n\n")
        try:
            print(slide.strip("\n"))
            slide = slide.strip("\n")
            slide = slide.strip(" ")
            print("slide: ",slide)
            slide_path = ""
            src_path = join(nas_path, slide)
            status = move_slide_folder(src_path, dst_path_1, nas_ip,
                            transfer_type="scp",
                            user="nasdrive", reverse=True)
            print("status: ", status)
            dicom_path = join(dst_path_1, slide, "dicom.tar")
            print("dicom_path: ", dicom_path)
            #delete_folder(dicom_path)
        except Exception as err_msg:
            print(err_msg)
            file1 = open("failed.txt", "a")  # append mode 
            file1.write(slide)
            file1.write("\n")
            file1.close()
