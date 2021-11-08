import sys
import os
import time
import json
import stat
import shutil

import requests
import subprocess

from os import walk
from glob import glob
from os.path import join, exists
from multiprocessing import Pool


# ==============================================================================
# TissueBiopsyUploadWorker
# ==============================================================================


class TissueBiopsyUploadWorker():
    '''
    Worker class for managing the uploading the of tissue biopsy data to
    the cloud or local server.
    '''
# |----------------------------------------------------------------------------|
# Class Variables
# |----------------------------------------------------------------------------|
#    no class variables

    USERNAME = "adminspin"

    PASSWORD = "adminspin#123"

# |----------------------------------------------------------------------------|
# Constructor
# |----------------------------------------------------------------------------|
    def __init__(self):
        self._cwd = os.getcwd()
        self._destDir = join("/var", "www", "html", "wsi_data",
                             "acquired_data")
        self._uploadType = "local"
        self.HOSTNAME = ""
        self.WEB_SERVER_ROOT = ""

        self._1xImgName = ""
        self._slideInfo = {}
        self._cloudSlideId = ""
        self._cloudClientId = ""
        self._cloudStudyId = ""

        self._appPath = ""
        self._slide_id = -1
        self._slide_name = ""
        self._gridList = []
        self._gridStatusHash = {}
        self._slidePath = ""
        self._modelFilePath = ""
        self._outputFilePath = ""
        self._metadataFilePath = ""
        self._acqParamFilePath = ""
        self._panParamFilePath = ""
        self._acqExePath = ""
        self._acqLibPath = ""
        self._panExePath = ""
        self._panLibPath = ""

        self._pyrGenExePath = ""
        self._baseTilesPath = ""
        self._baseTilesPath_0 = ""
        self._baseTilesPath_1 = ""

        self._blendingTime = 0
        self._uploadTime = 0
        self._pyramidGenerationTime = 0

        self._loggedInUserId = ""
        self._headerJson = None
        self.scan_type = None

# |---------------------------End of Constructor------------------------------|

# |----------------------------------------------------------------------------|
# _deleteFolder
# |----------------------------------------------------------------------------|
    def _deleteFolder(self, folderPath):
        if os.path.exists(folderPath):
            try:
                p = Pool()
                fileList = [y for x in walk(folderPath) for y in glob(join(x[0],
                                                                        '*.*'))]
                p.map_async(os.remove, fileList)
                p.close()
                p.join()
                shutil.rmtree(folderPath)
                print("Exiting delete folder")
            except Exception as msg:
                print("Exception occurred in deleting the folder: ", msg)
        else:
            print("No such directory: ", folderPath)

# |----------------------End of _deleteFolder---------------------------------|

# |----------------------------------------------------------------------------|
# _createDirectory
# |----------------------------------------------------------------------------|
    def _createDirectory(self, folderPath):
        if not exists(folderPath):
            os.mkdir(folderPath)
# |----------------------End of _createDirectory------------------------------|

# |----------------------------------------------------------------------------|
# delete_file
# |----------------------------------------------------------------------------|
    def delete_file(self, path):
        if exists(path):
            os.remove(path)
# # |----------------------End of delete_file---------------------------------|

# |----------------------------------------------------------------------------|
# _renameDirectory
# |----------------------------------------------------------------------------|
    def _renameDirectory(self, src, dst):
        if exists(src):
            os.rename(src, dst)
# |----------------------End of _renameDirectory------------------------------|

# |----------------------------------------------------------------------------|
# _grantPermission
# |----------------------------------------------------------------------------|
    def _grantPermission(self, filePath):
        try:
            os.chmod(filePath, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR |
                     stat.S_IWGRP | stat.S_IRGRP | stat.S_IXGRP |
                     stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH)
        except Exception as msg:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.join(os.getcwd(),
                                 os.path.split(exc_tb.tb_frame.f_code.co_filename)[1])
            print("fname: ", fname)
            print("Exception occured at ", exc_tb.tb_lineno, " in ", fname,
                  "Error is ", msg)

# |----------------------End of _grantPermission------------------------------|

# |----------------------------------------------------------------------------|
# _scp_to_destination
# |----------------------------------------------------------------------------|
    def _scp_to_destination(self, src, dest):

        cmd_pass = 'sshpass -p'
        cmd_scp = 'scp -r'
        dstpath = self.USERNAME + '@' + self.HOSTNAME + ':' + dest

        cmd = "{} {} {} {} {}".format(cmd_pass, self.PASSWORD, cmd_scp, src,
                                      dstpath)
        print("cmd :", cmd)

        staus = os.system(cmd)

        if staus == 0:
            print('done')
            return True
        else:
            print('not transfered')
            return False

# |----------------------End of _scp_to_destination---------------------------|

# |----------------------------------------------------------------------------|
# _writeParseJson
# |----------------------------------------------------------------------------|
    def _writeParseJson(self):
        try:
            with open(self._metadataFilePath, 'r') as jsonFile:
                jsonData = json.load(jsonFile)
            jsonFile.close()

            gridInfoList = jsonData['grid_info']

            for gridInfo in gridInfoList:
                if self._gridName == gridInfo['grid_name']:
                    gridInfoJson = {}

                    jsonParseData = {}
                    timeList = []
                    timeHash = {
                                    "blend_time": self._blendingTime,
                                    "grid_name": self._gridName
                                }
                    if exists(self._parseJsonPath):
                        with open(self._parseJsonPath, 'r+') as file:
                            jsonParseData = json.loads(file.read())
                            file.seek(0)
                            file.truncate(0)

                            timeList = jsonParseData["time"]

                            for grid in timeList:
                                if grid["grid_name"] == self._gridName:
                                    continue
                                else:
                                    grid["blend_time"] = self._blendingTime

                            parseJson = {
                                            "time": timeList
                                        }
                            json.dump(parseJson, file, indent=2)
                        file.close()
                    else:
                        timeList.append(timeHash)

                        gridInfoJson = {
                                        "time": timeList
                                    }

                        with open(self._parseJsonPath, 'w') as file:
                            json.dump(gridInfoJson, file, indent=2)
                        file.close()
        except Exception as msg:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.join(os.getcwd(), os.path.split(exc_tb.tb_frame.f_code.co_filename)[1])
            print("Exception occured at ", exc_tb.tb_lineno, " in ", fname, "Error is ", msg) 
# |----------------------End of _writeParseJson---------------------------------|

# |----------------------------------------------------------------------------|
# _parseJson
# |----------------------------------------------------------------------------|
    def _parseJson(self):
        with open(self._metadataFilePath, 'r') as jsonFile:
            jsonData = json.load(jsonFile)
        jsonFile.close()

        self._slideInfo = jsonData
# |----------------------End of _parseJson------------------------------------|

# |----------------------------------------------------------------------------|
# _getRequest
# |----------------------------------------------------------------------------|
    def _getRequest(self, url, params):
        response = requests.get(url=url, params=params,
                                headers=self._headerJson, timeout=None)
# |----------------------End of _getRequest-----------------------------------|

# |----------------------------------------------------------------------------|
# _postRequest
# |----------------------------------------------------------------------------|
    def _postRequest(self, postUrl, postData, params = {}):
        # print(postUrl)
        postData = json.dumps(postData)
        response = requests.post(postUrl, params=params, 
                                 data=postData,
                                 headers=self._headerJson,
                                 timeout=None)
        return response.text, response.status_code

# |----------------------End of _postRequest----------------------------------|

# |----------------------------------------------------------------------------|
# _startStitching
# |----------------------------------------------------------------------------|
    def _startStitching(self, zStackViewStatus=False, zStackViewSize=0):
        errMsg = ""
        try:
            if exists(self._modelFilePath) is True:
                self._createDirectory(self._baseTilesPath_mpi_base)
                startTime = time.time()
                print(["mpiexec", "--bind-to", "core:1", "-np", "1",
                       self._panExePath, self._panLibPath,
                       self._baseTilesPath, "0", "0", ":", "-np",
                       self.processes, self._panExePath, self._panLibPath,
                       self._panParamFilePath, self.gridPath, "0"])
#                 print(["mpiexec", "--bind-to", "core:1", "-np", "1",
#                        self._panExePath,    self._panLibPath,
#                        self._baseTilesPath, "0", ":", "-np",
#                        self.processes, self._panExePath, self._panLibPath,
#                        self._panParamFilePath, "0"])
#                 self._process = subprocess.Popen(["mpiexec", "--bind-to",
#                                                   "core:1", "-np", "1",
#                                                   self._panExePath,
#                                                   self._panLibPath,
#                                                   self._baseTilesPath, "0",
#                                                   ":", "-np", self.processes,
#                                                   self._panExePath,
#                                                   self._panLibPath,
#                                                   self._panParamFilePath,
#                                                   "0"])
                self._process = subprocess.Popen(["mpiexec", "-np", "1",
                                                  self._panExePath,
                                                  self._panLibPath,
                                                  self._baseTilesPath, "0",
                                                  "0", ":", "-np",
                                                  self.processes,
                                                  self._panExePath,
                                                  self._panLibPath,
                                                  self._panParamFilePath,
                                                  self.gridPath, "0"])

                self._process.wait()

                '''
                self._process =\
                    subprocess.Popen(["mpiexec", "--bind-to",
                                      "core:4", "-np", "1",
                                      self._pyrGenExePath_old,
                                      self._baseTilesPath, "1"])

                self._process.wait()
                '''

                # Generate base tiles for fused aois.
                fused_images_path = join(self._slidePath, self._gridName,
                                         "fused_images")

                if os.path.exists(fused_images_path):
                    self._baseTilesPath_fused =\
                        join(self.gridPath, "base_tiles_fused", "000")
                    self._baseTilesPath_mpi_base_fused =\
                        join(self.gridPath, "base_tiles_fused")
                    fused_model_file_path = join(self.gridPath + "/model_fused.vista")
                    fused_white_path = join(self.gridPath + "/white_fused.bmp")

                    # Rename the raw images folder to temporary name.
                    prev_folder_path = join(self.gridPath, "raw_images")
                    temp_folder_path = join(self.gridPath, "raw_images_")
                    os.rename(prev_folder_path, temp_folder_path)

                    # Rename the fused images folder to raw images.
                    os.rename(fused_images_path, prev_folder_path)

                    # Rename the model files.
                    # prev_file_path = join(self.gridPath + "/model.vista")
                    # temp_file_path = join(self.gridPath + "/model_.vista")
                    # os.rename(prev_file_path, temp_file_path)

                    # os.rename(fused_model_file_path, prev_file_path)

                    prev_white_path = join(self.gridPath + "/white.bmp")
                    temp_white_path = join(self.gridPath + "/white_.bmp")
                    os.rename(prev_white_path, temp_white_path)

                    os.rename(fused_white_path, prev_white_path)

                    # Run blending and pyramid generation.
                    self._createDirectory(self._baseTilesPath_mpi_base_fused)
                    startTime = time.time()
                    print(["mpiexec", "--bind-to",
                           "core:1", "-np", "1",
                           self._panExePath,
                           self._panLibPath,
                           self._baseTilesPath_fused, "0",
                           ":", "-np", self.processes,
                           self._panExePath,
                           self._panLibPath,
                           self._panParamFilePath,
                           "0"])
                    self._process =\
                        subprocess.Popen(["mpiexec", "--bind-to",
                                          "core:1", "-np", "1",
                                          self._panExePath,
                                          self._panLibPath,
                                          self._baseTilesPath_fused, "0", "0",
                                          ":", "-np", self.processes,
                                          self._panExePath,
                                          self._panLibPath,
                                          self._panParamFilePath,
                                          self.gridPath,
                                          "0"])

                    self._process.wait()

                    self._process =\
                        subprocess.Popen(["mpiexec", "--bind-to",
                                          "core:4", "-np", "1",
                                          self._pyrGenExePath_old,
                                          self._baseTilesPath_fused, "1"])
                    self._process.wait()

                    # Rename the folders to their original names.
                    os.rename(prev_folder_path, fused_images_path)
                    os.rename(temp_folder_path, prev_folder_path)

                    # os.rename(prev_file_path, fused_model_file_path)
                    # os.rename(temp_file_path, prev_file_path)

                    os.rename(prev_white_path, fused_white_path)
                    os.rename(temp_white_path, prev_white_path)

                if zStackViewStatus is True:
                    for stack_index in range(1, zStackViewSize+1):
                        # Rename z folder to bkp
                        z_grid_path = join(self.gridPath, "1",
                                           str(self.gridMag) + "x")
                        z_bkp_path = join(self.gridPath, "1",
                                          str(self.gridMag) + "x_bkp")
#                             print("z_grid_path : ", z_grid_path)
#                             print("z_bkp_path : ", z_bkp_path)
                        self._renameDirectory(z_grid_path, z_bkp_path)

#                             Rename z-1 folder to z
#                             print("Rename z-1 folder to z")
                        z_minus_one_path = join(self.gridPath, "1",
                                                str(self.gridMag) + "x-" +
                                                str(stack_index))
                        z_minus_one_bkp_path = join(self.gridPath, "1",
                                                    str(self.gridMag) +
                                                    "x")

                        self._renameDirectory(z_minus_one_path,
                                              z_minus_one_bkp_path)

                        self._process = subprocess.Popen(
                                 ["mpiexec",
                                  "--bind-to",
                                  "core:1",
                                  "-np", "1",
                                  self._panExePath,
                                  self._panLibPath,
                                  self._lowerBaseTilePath[stack_index - 1],
                                  "0", ":", "-np", self.processes,
                                  self._panExePath, self._panLibPath,
                                  self._panParamFilePath, "0"])
                        self._process.wait()
                        # Reverting back to z-1 from z
                        self._renameDirectory(z_minus_one_bkp_path,
                                              z_minus_one_path)

                        # Rename z+1 folder to z
                        z_plus_one_path =\
                            join(self.gridPath, "1",
                                 str(self.gridMag) + "x+" +
                                 str(stack_index))
                        z_plus_one_bkp_path = join(self.gridPath, "1",
                                                   str(self.gridMag) + "x")
                        self._renameDirectory(z_plus_one_path,
                                              z_plus_one_bkp_path)

                        self._process =\
                            subprocess.Popen(["mpiexec",
                                              "--bind-to", "core:1",
                                              "-np", "1", self._panExePath,
                                              self._panLibPath,
                                              self._upperBaseTilePath[stack_index- 1],
                                              "0", ":", "-np",
                                              self.processes,
                                              self._panExePath,
                                              self._panLibPath,
                                              self._panParamFilePath, "0"])
                        self._process.wait()
                        # Reverting back to z+1 from z
                        self._renameDirectory(z_plus_one_bkp_path, z_plus_one_path)
                        # Reverting back to z from z_bkp
                        self._renameDirectory(z_bkp_path, z_grid_path)

                endTime = time.time()
                self._blendingTime += (endTime - startTime)

                try:
                    startTime = time.time()

                    if zStackViewStatus is True:
                        for stack_index in range(1, zStackViewSize + 1):
                            self._process =\
                                subprocess.Popen([self._pyrGenExePath_old,
                                                  self._lowerBaseTilePath[stack_index - 1],
                                                  "2"])
                            self._process.wait()

                            self._process =\
                                subprocess.Popen([self._pyrGenExePath_old,
                                                  self._upperBaseTilePath[stack_index - 1],
                                                  "2"])
                            self._process.wait()

                    endTime = time.time()
                    self._pyramidGenerationTime += (endTime - startTime)
                except Exception as msg:
                    errMsg = "pyramid generation failed"
                    return False, errMsg
            else:
                errMsg = "model.vista doesn't exist"
                return False, errMsg

            return True, ""
        except Exception as msg:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.join(os.getcwd(), os.path.split(exc_tb.tb_frame.f_code.co_filename)[1])
            print("Exception occured at ", exc_tb.tb_lineno, " in ", fname, "Error is ", msg)
            errMsg = "blending failed"
            return False, errMsg

# |-------------------------End of _startStitching----------------------------|

# |----------------------------------------------------------------------------|
# _writeJsonFile
# |----------------------------------------------------------------------------|
    def _writeJsonFile(self, metadataFilePath, postData):
        with open(metadataFilePath, "w") as outfile:
            json.dump(postData, outfile, indent=2)
        outfile.close()

# |----------------------End of _writeJsonFile--------------------------------|

# |----------------------------------------------------------------------------|
# _update_slide_status
# |----------------------------------------------------------------------------|
    def _update_slide_status(self, activity_status, error_info, grid_status,
                             status_msg):
        # Update slide upload status
        print("self._slideInfo: ", self._slideInfo)
        post_cmd = "scanner/set_activity_status"
        post_url = "http://localhost:8000/{}".format(post_cmd)
        grid_info = []
        
        if self._grid_id != -1:
            grid_info = [
                    {
                        "grid_name": self._gridName,
                        "grid_id": self._grid_id,
                        "grid_status": grid_status,
                        "error_info": status_msg
                    }
            ]
            
        slide_info = {
            "slide_id": self._slide_id,
            "slide_name": self._slide_name,
            "activity": "acquisition",
            "activity_status": activity_status,
            "error_info": error_info,
            "grid_info": grid_info
            }
        post_list_data = {
                            "activity_info": [slide_info]
                         }
        ProgressrespJson, statusCode =\
            self._postRequest(post_url, post_list_data)
        print("Set activity status post url", post_url)
        print("Set activity status post data", post_list_data)
        print("Set activity status response: ", ProgressrespJson)

# |----------------------End of _update_slide_status--------------------------|

# |----------------------------------------------------------------------------|
# move_slide_folder_to_datadrive
# |----------------------------------------------------------------------------|
    def move_slide_folder_to_datadrive(self, src_path, dst_path):
        move_status = False

        try:
            if self.upload_type == "cloud":
                print("Move src_path: ", src_path, " to: ", dst_path)
                command = "sshpass -p 'adminspin#123' rsync -e 'ssh -o "\
                    "StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' "\
                    "-av {} adminspin@{}:{}".format(src_path, self.ip_address, dst_path)
            else:
                if exists(src_path) is True:
                    command = "rsync -a {} {}".format(src_path, dst_path)
                print("Move src_path: ", src_path, " to: ", dst_path)
                command = "rsync -a {} {}".format(src_path, dst_path)

            print("Move command: ", command)

            startTime = time.time()
            status = os.system(command)
            endTime = time.time()
            total_time = endTime - startTime
            print("Total time to move: ", total_time)

            if status == 0:
                move_status = True
            else:
                move_status = False
            print("Move status is: ", move_status)
        except Exception as error_msg:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.join(os.getcwd(), os.path.
                                 split(exc_tb.tb_frame.f_code.co_filename)[1])
            print("Exception occured at ", exc_tb.tb_lineno, " in ", fname,
                  "Error is ", error_msg)

        return move_status

# |----------------------End of move_slide_folder_to_datadrive----------------|

# |----------------------------------------------------------------------------|
# start
# |----------------------------------------------------------------------------|
    def start(self, zStackViewStatus=False, zStackViewSize=0, deleteAois=True,
              sync=True, blend=True, auto_grid_sync=False):
        errMsg = ""
        try:
            self.gridMag = "20x"
            self.gridPath = join(self._slidePath, self._gridName)

            self.gridDstPath = join(self._slideDestinationPath, self._gridName)

            self._modelFilePath = join(self.gridPath, "model.vista")
            self._acqParamFilePath = join(self.gridPath,
                                          "acquisition.xml")
            self._panParamFilePath = join(self.gridPath, "panorama.xml")

            self._slide_base_tiles_path =\
                join(self._slidePath, "base_tiles", "000")

            # if os.path.exists(self._slide_base_tiles_path) is False:
            #     os.mkdir(join(self._slidePath, "base_tiles"))
            #     os.mkdir(self._slide_base_tiles_path)

            self._baseTilesPath = join(self.gridPath, "base_tiles", "000")

            self._baseTilesDstPath = join(self.gridDstPath, "base_tiles",
                                          "000")

            self._baseTilesPath_mpi_base = join(self.gridPath, "base_tiles")

            self._baseTilesDstPathMpiBase = join(self.gridDstPath,
                                                 "base_tiles")

            self._upperBaseTilePath = []
            self._lowerBaseTilePath = []

            for stack_index in range(1, zStackViewSize+1):
                lowerTilePath = join(self.gridPath, "base_tiles", "000-" +
                                     str(stack_index))
                self._lowerBaseTilePath.append(lowerTilePath)
                upperTilePath = join(self.gridPath, "base_tiles", "000+" +
                                     str(stack_index))
                self._upperBaseTilePath.append(upperTilePath)
            
            grid_status = "error"
            statusMsg = ""
            
            if blend is True:
                # Generate the white reference image.
                white_gen_cmd = "python3 {} {} {} {}".\
                    format(join("/home", "adminspin", "wsi_backend", "views",
                                "utilities", "white_generation", 
                                "select_bg_inline.py"),
                                join(self._appPath, "acquired_data"),
                                self._slide_name, self._gridName)
                print("White generation command: ", white_gen_cmd)
                white_gen_proc =\
                        subprocess.Popen([white_gen_cmd], stdout=subprocess.PIPE,
                                         shell=True)
                white_gen_proc.wait()

                # Replace all the background AOIs with white reference image.
                if os.path.exists(join(self.gridPath, "bg_aoi_names.json")) is True:
                    replace_bg_cmd = "python3 {} {} {}".\
                        format(join("/home", "adminspin", "wsi_backend", "views",
                                    "utilities", "replace_bg_aois.py"),
                                    self._slidePath, self._grid_id)
                    print("Background replacing command: ", replace_bg_cmd)

                    replace_bg_proc =\
                            subprocess.Popen([replace_bg_cmd], stdout=subprocess.PIPE,
                                            shell=True)
                    replace_bg_proc.wait()

                # Start belnding and pyramid generation.
                statusVal, statusMsg = self._startStitching(zStackViewStatus,
                                                            zStackViewSize)
                
                if statusVal is False:
                    grid_status = "error"
                    # TODO: do we need to remove raw and fused images if
                    # blending fails?
                else:
                    grid_status = "completed"

                # if deleteAois is True:
                #     print("Started deleting raw images aois")
                #     # Delete 20x images.
                #     raw_img_path = join(self.gridPath, "raw_images")
                #     print("Aois raw_img_path: ", raw_img_path)
                #     self._deleteFolder(raw_img_path)

                #     print("Started deleting fused raw images aois")
                #     # Delete 20x images.
                #     fused_path = join(self.gridPath, "fused_images")

                #     if exists(fused_path):
                #         print("Aois fused_path: ", fused_path)
                #         self._deleteFolder(fused_path)

            # Move grid by grid also to data-drive.
            activity_status = "ongoing"
            error_info = ""
            if auto_grid_sync is True:
                move_status =\
                    self.move_slide_folder_to_datadrive(self._slidePath,
                                                        self._destDir)
                # Check grid got moved or not. 
                if move_status is False:
                    grid_status = "error"

                # if sync is True:
                #     activity_status = "completed"
                #     # Delete folder folder from nvme.
                #     self._deleteFolder(self._slidePath)
            elif auto_grid_sync is False and sync is True:
                # Merge the basetiles of every grid into single folder.
                script_path = join(self._appPath, "merging_grids.py")

                cmd = ["python3", script_path, self._appPath + "/acquired_data",
                       slide_name]
                print("cmd: ", cmd)

                merging_grids = subprocess.Popen(cmd, preexec_fn=os.setsid)

                merging_grids.wait()

                # Run the pyramid generation.
                self._process =\
                    subprocess.Popen(["mpiexec", "--bind-to",
                                      "core:4", "-np", "1",
                                      self._pyrGenExePath_old,
                                      self._slide_base_tiles_path, "1"])

                self._process.wait()

                # call rsync method.
                activity_status = "uploading"
                self._update_slide_status(activity_status, error_info,
                                          grid_status, statusMsg)
            #     move_status =\
            #         self.move_slide_folder_to_datadrive(self._slidePath,
            #                                             self._destDir)

            #     if move_status is True:
            #         activity_status = "completed"
            #         # Delete folder folder from nvme.
            #         self._deleteFolder(self._slidePath)
            #     else:
            #         activity_status = "error"
            #         error_info = "Failed to move slide folder"

            # self._update_slide_status(activity_status, error_info,
            #                           grid_status, statusMsg)
        except Exception as msg:
            errMsg = str(msg)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.join(os.getcwd(), os.path.
                                 split(exc_tb.tb_frame.f_code.co_filename)[1])
            print("Exception occured at ", exc_tb.tb_lineno, " in ", fname,
                  "Error is ", msg)
            f = open(self._outputFilePath, "w")
            if errMsg:
                f.write("error:{}\n".format(errMsg))
            f.close()

# |----------------------End of start-----------------------------------------|

# |----------------------------------------------------------------------------|
# init
# |----------------------------------------------------------------------------|
    def init(self, appPath, slide_name, processes, gridName, slide_id,
             folder_name, grid_id, ip_address, upload_type):
        self._slideDestinationPath = join(self._destDir, slide_name)
        self._appPath = appPath
        self._slide_name = slide_name
        self._slide_id = slide_id
        self._gridName = gridName
        self._grid_id = grid_id
        self.gridMag = 0
        self.gridPath = ""
        self.processes = processes
        self.ip_address = ip_address
        self.upload_type = upload_type
        self._slidePath = join(self._appPath, "acquired_data",
                               folder_name)
        self._metadataFilePath = join(self._slidePath, "metadata.json")
        self._parseJsonPath = join(self._slidePath, "parse.json")
        self._panExePath = join(self._appPath, "libs",
                                "ut_spinvistaPanorama_mpi")
        self._panLibPath = join(self._appPath, "libs",
                                "libspinvistaPanorama.so")
        self._pyrGenExePath = join(self._appPath, "libs",
                                   "ut_spinpyramidgeneration_mpi")
        self._pyrGenExePath_old = join(self._appPath, "libs",
                                       "ut_spinpyramidgeneration")
        self._outputFilePath = join(self._slidePath, "post_proc.txt")

# |----------------------End of init------------------------------------------|


# |----------------------------------------------------------------------------|
# transferProgress
# |----------------------------------------------------------------------------|
def transferProgress(transferred, toBeTransferred):
    progress = (transferred / toBeTransferred) * 100.0
# |----------------------End of transferProgress------------------------------|


if __name__ == "__main__":
    if len(sys.argv) == 2:
        appPath = "/home/adminspin/wsi_app"
        slide_name = sys.argv[1]
        # gridName = sys.argv[3]
        # deleteAois = False if sys.argv[4] == "0" else True
        processes = 1
        # zStackViewStatus = False if sys.argv[6] == "0" else True
        # zStackViewSize = int(sys.argv[7])
        slide_id = 1
        blend = True
        sync = False
        folder_name = slide_name
        # grid_id = int(sys.argv[12])
        auto_grid_sync = False
        ip_address = "localhost"
        upload_type = "local"
        slide_path = join(appPath, "acquired_data", slide_name)
        
        grids = []
        for folder in os.listdir(slide_path):
            if "grid" in folder and os.path.isdir(join(slide_path, folder)):
                if os.path.exists(join(slide_path, folder, "base_tiles")) is False:
                    os.mkdir(join(slide_path, folder, "base_tiles"))
                grids.append(folder)

        print("grids: ", grids)
        for itr, gridName in enumerate(grids):
            # Run displacement estimation
            # Merge the basetiles of every grid into single folder.
            script_path = "/home/adminspin/wsi_app/libs/white_generation/select_bg_inline.py"

            cmd = ["python3", script_path, join(appPath, "acquired_data"),
                    slide_name, gridName]
            print("cmd: ", cmd)

            white = subprocess.Popen(cmd, preexec_fn=os.setsid)

            white.wait()

            grid_path = join(slide_path, gridName)
            cmd = "mpiexec -np 1 ./ut_spinvistaPanorama_mpi ./libspinvistaPanorama.so {}/base_tiles/000 0 0 : -np 1 ./ut_spinvistaPanorama_mpi ./libspinvistaPanorama.so {}/panorama.xml {} 0".format(grid_path, grid_path, grid_path)
            print("cmd: ", cmd)
            os.system(cmd)

            slide_base_tiles_path = join(appPath, "acquired_data", slide_name, gridName, "base_tiles", "000")
            # Run the pyramid generation.
            print("Starting pyramid------")
            process =\
                subprocess.Popen(["mpiexec", "--bind-to",
                                    "core:4", "-np", "1",
                                    "/home/adminspin/wsi_app/libs/ut_spinpyramidgeneration",
                                    slide_base_tiles_path, "1"])

            process.wait()

        # # Merge the basetiles of every grid into single folder.
        # script_path = join(appPath, "libs", "merging_grids.py")

        # cmd = ["python3", script_path, appPath + "/acquired_data",
        #         slide_name]
        # print("cmd: ", cmd)

        # merging_grids = subprocess.Popen(cmd, preexec_fn=os.setsid)

        # merging_grids.wait()
        

        # slide_base_tiles_path = join(appPath, "acquired_data", slide_name, "grid_merged", "base_tiles", "000")
        # # Run the pyramid generation.
        # process =\
        #     subprocess.Popen(["mpiexec", "--bind-to",
        #                         "core:4", "-np", "1",
        #                         "/home/adminspin/wsi_app/libs/ut_spinpyramidgeneration",
        #                         slide_base_tiles_path, "1"])

        # process.wait()
        
            # post_proc_output = join(appPath, "acquired_data", slide_name,
            #                         gridName + "_post_proc_output.txt")
            # sys.stdout = open(post_proc_output, 'wt')
            # workerObj = TissueBiopsyUploadWorker()
            # startTime = time.time()
            # workerObj.init(appPath=appPath, slide_name=slide_name,
            #             processes=processes, gridName=gridName,
            #             slide_id=slide_id, folder_name=folder_name,
            #             grid_id=itr + 1, ip_address=ip_address,
            #             upload_type=upload_type)
            # workerObj.start(sync,
            #                 blend, auto_grid_sync)
            # endTime = time.time()
            # print("Total time (s): ", (endTime - startTime))

        # requests.post("http://localhost:8000/scanner/post_process_worker")

        sys.exit(0)
    else:
        # requests.post("http://localhost:8000/scanner/post_process_worker")

        print("Invalid arguments")
        print("<app_path> <slide_name> <gridName> <deleteAois> <processes> "
              "<zStackViewStatus-0,1> <zStackViewSize>")
        sys.exit(-1)
