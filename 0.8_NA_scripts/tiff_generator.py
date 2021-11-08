import os, subprocess, pyvips, time, sys, json, shutil, resource, pwd
from os.path import join, exists, isfile, isdir, abspath

class TiffGen():
    # |----------------------------------------------------------------------------|
    # determineSplitTerm
    # |----------------------------------------------------------------------------|
        def determineSplitTerm(self, number):
            if number % 4 == 0:
                return number
            else:
                return self.determineSplitTerm(number-1)
    # |-----------------End of determineSplitTerm-------------------------------|

    # |----------------------------------------------------------------------------|
    # create_folder
    # |----------------------------------------------------------------------------|
        def create_folder(self, path):
            if not exists(path):
                os.makedirs(path)
    # |-----------------End of create_folder-------------------------------|

    # |----------------------------------------------------------------------------|
    # generateSequence
    # |----------------------------------------------------------------------------|
        def generateSequence(self, row, column, btPath, destination, compressionType, number_of_grids, total_per):
            try:
                print("getrlimit before:", resource.getrlimit(resource.RLIMIT_NOFILE))
                resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
                print("getrlimit:", resource.getrlimit(resource.RLIMIT_NOFILE))
                print("subprocess:", subprocess.check_output("whoami; ulimit -n", shell=True))
            
                user = "adminspin"
                pwnam = pwd.getpwnam(user)
                os.setgid(pwnam.pw_gid)
                os.setuid(pwnam.pw_uid)
                print("getrlimit:", resource.getrlimit(resource.RLIMIT_NOFILE))
                print("subprocess:", subprocess.check_output("whoami; ulimit -n", shell=True))
            
                tmpRow, tmpCol = 0, 0
                total_aoi = row * column
            
                print("row*column: ", row*column)
                if row*column > 4000:
                    rowNum = self.determineSplitTerm(row)
                    colNum = self.determineSplitTerm(column)
                    tmpRow, tmpCol = rowNum, colNum
                else:
                    rowNum = row
                    colNum = column
                    tmpRow, tmpCol = rowNum, colNum
            
                print("tmpRow*tmpCol: ", tmpRow*tmpCol)

                if tmpRow*tmpCol > 57600:
                    while (tmpRow*tmpCol) > 4000:
                        print("First loop")
                        tmpRow //= 8
                        tmpCol //= 8
                elif tmpRow*tmpCol > 4000:
                    while (tmpRow*tmpCol) > 4000:
                        print("Second loop")
                        tmpRow //= 4
                        tmpCol //= 4
                elif tmpRow*tmpCol > 2000:
                    while (tmpRow*tmpCol) > 2000:
                        print("Second loop")
                        tmpRow //= 2
                        tmpCol //= 2
                else:
                    # while (tmpRow*tmpCol) > 1000:
                    #     print("Third loop")
                    #     # print("Second loop")
                    #     tmpRow //= 1
                    #     tmpCol //= 1
                    tmpRow = row
                    tmpCol = column
            
                print(row)
                print(column)
                print(tmpRow)
                print(tmpCol)
                #tmpCol = tmpCol + 1
                #tmpRow = tmpRow + 1
            
                i ,j = 0, 0
            
                rc = tmpRow
                quad = 1
                quadrow = 0
                
            
                baseTilesPath = btPath
                slideName = baseTilesPath.split("/")[-5] + '_' + baseTilesPath.split("/")[-4] 
            
                start = time.time()
                counter = 0
                per = 0
            
                while i < (rowNum):
            
                    j = 0
                    cc = tmpCol
                    quadCol = 0
                    # print("-----------------> I <--------------------")
                    while j < (colNum):       
                        startColumn = j
                        endColumn = j + cc
            
                        startRow = i 
                        endRow = i + rc
            
                        tiles = []
                        for y in range(startColumn, endColumn):
                            for x in range(startRow, endRow):
                                imgName = "{}_{}.jpeg".format(x, y)
                                counter = counter + 1
                                img = join(baseTilesPath, imgName)
                                if exists(img):
                                    # print("imagename--- : ", img)
                                    tiles.append(pyvips.Image.new_from_file(img,
                                                                        access="sequential"))
                                    per = (counter/total_aoi) * 50
                                    # print("------percentage------ ", int(per / number_of_grids + total_per))
            
                        im2 = pyvips.Image.arrayjoin(tiles, across=(abs(tmpRow)))
                        imgName = "{}/{}_{}.jpeg".format(destination, quadrow, quadCol)
                        im2.write_to_file(imgName)
                        j += tmpCol
                        quad += 1
                        quadCol += 1
                        
                    quadrow += 1
                    i += tmpRow
            
                lisOfFilesToBeDeleted = []
                tiles = []
                c = 0
                per2 = 0
                total_quad = quadCol * quadrow
                
            
                for y in range(0, quadCol):
                    for x in range(0, quadrow):
                        imgName = "{}_{}.jpeg".format(x, y)
                        img = join(destination, imgName)
                        if exists(img):
                            lisOfFilesToBeDeleted.append(img)
                            tiles.append(pyvips.Image.new_from_file(img,
                                                                    access="sequential"))
                            c = c + 1
                            per2 = (c/total_quad) * 25
                            # print("------percentage------ ", int(per2 + per / number_of_grids + total_per))
            
                im2 = pyvips.Image.arrayjoin(tiles, across=quadrow)
                imageName = "{}/{}.tif[tile,pyramid,compression={}, bigtiff = True]".format(destination, slideName, compressionType)
                # print("imageName : ", imageName)
                im2.write_to_file(imageName)
            
                time.sleep(1)
            
                total_list_del = len(lisOfFilesToBeDeleted)
                del_count = 0
                del_per = 0
            
                for images in lisOfFilesToBeDeleted:
                    cmd = "rm -rf {}".format(images)
                    os.system(cmd)
                    del_count = del_count + 1
                    del_per = (del_count/total_list_del) * 25
            
                    tmp_total_per = int(per2 + per + del_per / number_of_grids + total_per)
            
                    if tmp_total_per > 100:
                        tmp_total_per = 100
                    # print("------percentage------ ", tmp_total_per)
    #                 status["progress"] = tmp_total_per
            
                total_grid_per = (per + per2 + del_per) / number_of_grids
                # print("------percentage------ ", total_grid_per)
                # print("Total Time Taken {} secs".format(time.time() - start))
                # print()
                # print("Counter : ", counter)
                return total_grid_per
            except Exception as err:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("fname : ", fname)
                print("exc_tb.tb_lineno : ", exc_tb.tb_lineno)        
                print("Error Occured", err)  
    # |-----------------End of generateSequence-------------------------------|
    
    # |----------------------------------------------------------------------------|
    # export
    # |----------------------------------------------------------------------------|
        def export(self, slideName, compressionType):
            try:
                # print("--------percentage--------- ", 0)
                # print("slideName", slideName)
                mainSlidePath = "/home/adminspin/wsi_app/acquired_data"
                # mainSlidePath = join("/media", "adminspin", "ImageData")
                slidePath = join(mainSlidePath, slideName)
                print("--------slidePath------------ ", slidePath)
                # folderName = os.path.basename(slidePath)
                # folderName = "{}_{}".format(slideName, folderName)
                
                folders = []
                gridList = []
                grid_names = []
                total_per = 0
    
                dst = join("/home/adminspin/wsi_app/acquired_data", slide_name + "_tiff")
                print("----------dst--------- ", dst)
                self.create_folder(dst)
                
                for _, grids, files in os.walk(slidePath):
                    for grid in grids:
                        if "grid" in grid:
                            grid_names.append(grid)
                            gridList.append(join(slidePath, grid))
                    break    
                
                number_of_grids = len(gridList)
                # gridList = ["/media/adminspin/693b71f6-6abe-428d-aac2-f9298c90f3e5/wsi_data/acquired_data/2001V501002_21997/grid_2"]
                print("gridList: ", gridList)
                for itr, gridDir in enumerate(gridList):
                    print("gridDir: ", gridDir)
                    numberOfRows = 0
                    numberOfColoums = 0
                    grid_dst_name_tiff = "{}_{}.tif".format(slideName, grid_names[itr])
                    dst_grid = join(dst, grid_dst_name_tiff)

                    if not exists(dst_grid):
                        tmp_path = join(gridDir, "base_tiles")
                        print("tmp_path: ", tmp_path)

                        if exists(tmp_path):
                            tmp_folders = None
                            for _, tmp_folders, tmp_files in os.walk(tmp_path):
                                break
                            
                            print("tmp_folders : ", tmp_folders, " : ", len(tmp_folders))
                            
                            if len(tmp_folders) > 0:
                                baseTilePath = join(gridDir, "base_tiles", "000_files")
                                print("baseTilePath: ", baseTilePath)
                                if exists(baseTilePath):
                                    for _, folders, files in os.walk(baseTilePath):
                                        break
                                    maxFolder = max([int(i) for i in folders])
                                    targetbaseTilePath = join(baseTilePath, str(maxFolder))
                                    for _, folders, files in os.walk(targetbaseTilePath):
                                        break
                                    for number in files:
                                        if int(number.split("_")[0]) > numberOfRows:
                                            numberOfRows = int(number.split("_")[0]) 
                                        elif int(number.split("_")[1].split(".")[0]) > numberOfColoums:
                                            numberOfColoums = int(number.split("_")[1].split(".")[0]) 
                                    print("ROWS : COLUMNS : ", numberOfRows , " : ", numberOfColoums)
                                    tmp_per = self.generateSequence(numberOfRows, numberOfColoums,
                                                                    targetbaseTilePath, dst,
                                                                    compressionType, number_of_grids,
                                                                    total_per)
                                    total_per = tmp_per + total_per
                                    # print("total per cent = ", total_per)
                                    # print("Tiff generated successfully")
                            
                    # print("--------percentage--------- ", 100)
                        
            except Exception as err:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("fname : ", fname)
                print("exc_tb.tb_lineno : ", exc_tb.tb_lineno)        
                print("Error Occured", err)  
            # |-----------------End of export-------------------------------|

if __name__ == '__main__':
    
    # if len(sys.argv) != 1:
    slide_name = sys.argv[1]
    # print("Missing arguments, Please check !!!")
    # sys.exit(-1)
    
    # with open("/home/adminspin/Desktop/regenerate.txt", "r") as file_data:
    #     data = file_data.read().split("\n")
    # # print("data: ", type(data))
    # print("data: ", data)
    # data = list(data)
    tiff_obj = TiffGen()
    # for slide_name in data:
    # if slide_name != "":
    #     # slide_name = "2001V501002_21997"
    print(slide_name)
    tiff_obj.export(slide_name, "jpeg")
    # break
    # file_path = join("/var", "www", "html", "wsi_data", "acquired_data", "task_info.txt")
    # file_path = join("/media/adminspin/693b71f6-6abe-428d-aac2-f9298c90f3e5/wsi_data/acquired_data", "task_info.txt")
    # file1 = open(file_path, "a")  # append mode 
    # file1.write("----------Finished Tiff generation for " + slide_name + "------------\n") 
    # file1.write("------------------------------------------------------------\n") 
    # file1.close() 

