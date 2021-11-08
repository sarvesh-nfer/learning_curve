import os, subprocess, pyvips, time, sys, json, shutil
from os.path import join, exists, isfile, isdir, abspath
from tkinter import *
import tkinter, tkinter.filedialog
from functools import partial

class TiffGen():
    def __init__(self):
        self.input_val = 0
        self.status = True
        root = tkinter.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        src_dir = tkinter.filedialog.askdirectory(parent=root,initialdir="/",title='Select Source Directory')
        #print("src_dir : ", src_dir)
        src_dir = src_dir.replace("\\", '/')
        #print("src_dir 2: ", src_dir)
        slide_name = src_dir[src_dir.rfind("/") + 1 :].strip("\n")
        #print("slide_name----- ", slide_name)
        root.destroy()

        # if len(src_dir) == 0:
        #     return HttpResponse("Please Select source folder")
        # print("src_dir------ : ", src_dir)

        root = tkinter.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        destination_dir = tkinter.filedialog.askdirectory(parent=root,initialdir="/",title='Select Directory To Store Tiff images')
        root.destroy()
        # if len(destination_dir) == 0:
        #     return HttpResponse("Please Select destination directory where to generate")
        
        
        # slide_name = request.GET["slide_name"]
        destination_dir = join(destination_dir, slide_name)
        self.create_folder(destination_dir)
        #print("destination_dir------ : ", destination_dir)
        destination_dir = destination_dir.replace("\\", '/')
        #print("destination_dir------ 2: ", destination_dir)
        
        # Select Compression type
        self.get_compression()
        compression = self.input_val
        #print("compression : ", compression)

        compressionType = ""

        if compression == 1:
            compressionType = "none"
        elif compression == 2:
            compressionType = "jpeg"
        elif compression == 3:
            compressionType = "deflate"
        elif compression == 4:
            compressionType = "lzw"

        slideList = []
        imageNameList = {}
        numberOfRows = 0
        numberOfColoums = 0
        folders = []

        slideList = []
        slides = []
        try:
            for _, grids, files in os.walk(join(src_dir)):
                for grid in grids:
                    slideList.append(join(src_dir, grid))
                break    
            
            # print(slideList)

            numberOfRows = 0
            numberOfColoums = 0
            #print("slideList : ", slideList)

            for slide in slideList:
                #print("slide : ", slide)
                slide = slide.replace("\\", '/')
                #print("slide 2: ", slide)
                baseTilePath = join(src_dir, slide, "base_tiles", "000_files")
                for _, folders, files in os.walk(baseTilePath):
                    break
                # print(baseTilePath)
                maxFolder = max([int(i) for i in folders])
                targetbaseTilePath = join(baseTilePath, str(maxFolder))
                #print(targetbaseTilePath)
                for _, folders, files in os.walk(targetbaseTilePath):
                    break

                for number in files:
                    if int(number.split("_")[0]) > numberOfRows:
                        numberOfRows = int(number.split("_")[0])
                    elif int(number.split("_")[1].split(".")[0]) > numberOfColoums:
                        numberOfColoums = int(number.split("_")[1].split(".")[0])
                targetbaseTilePath = targetbaseTilePath.replace("\\", '/')
                self.generateSequence(numberOfRows, numberOfColoums, targetbaseTilePath, compressionType, destination_dir)
                #print("Tiff generated successfully")

                # root = tkinter.Tk()
                # root.title("Status")
                # tkinter.Label(root, 
                #         text="""Tiff Successfully Generated to """ + src_dir,
                #         justify = tkinter.LEFT,
                #         padx = 50,
                #         font = "Helvetica 15 bold").pack()
                # print("hgshjgdhsdgsiuiooik")
                # time.sleep(5)
                # # progress_root.destroy() 
                

        except Exception as msg:
            print("Exception------- : ", msg)


    def determineSplitTerm(self, number):
        #print("-------determineSplitTerm------")
        if number % 4 == 0:
            return number
        else:
            return self.determineSplitTerm(number-1)


    def generateSequence(self, row, column, btPath, compressionType, destination):
        #print("-------generateSequence------")
        tmpRow, tmpCol = 0, 0

        if row*column > 4000:
            rowNum = self.determineSplitTerm(row)
            colNum = self.determineSplitTerm(column)
            tmpRow, tmpCol = rowNum, colNum
        else:
            rowNum = row
            colNum = column


        if tmpRow*tmpCol > 57600:
            while (tmpRow*tmpCol) > 4000:
                tmpRow //= 8
                tmpCol //= 8
        elif tmpRow*tmpCol > 4000:
            while (tmpRow*tmpCol) > 4000:
                tmpRow //= 4
                tmpCol //= 4
        else:
            tmpRow = row
            tmpCol = column

        print(row)
        print(column)
        print(tmpRow)
        print(tmpCol)

        # exit(0)
        i ,j = 0, 0

        rc = tmpRow
        quad = 1
        quadrow = 0
        

        baseTilesPath = btPath
        #print("baseTilesPath : ", baseTilesPath)
        slideName = baseTilesPath.split("/")[-5] + '_' + baseTilesPath.split("/")[-4] 
        #print("-------slideName------------ : ", slideName)

        start = time.time()

        while i < (rowNum):

            j = 0
            cc = tmpCol
            quadCol = 0
            while j < (colNum):
                
                print("Quad {}".format(quad))
                print("Row from {} to {}".format(i, i+rc))
                print("Coloumn from {} to {}".format(j, j + cc))
                print("Number of Images across {}".format(tmpRow))
                print("Quad Row - {} Quad Col - {}".format(quadrow, quadCol))            

                startColumn = j
                endColumn = j + cc

                startRow = i 
                endRow = i + rc
                print(startColumn, endColumn, j, cc)
                
                tiles = []
                for y in range(startColumn, endColumn):
                    for x in range(startRow, endRow):
                        imgName = "{}_{}.jpeg".format(x, y)
                        img = join(baseTilesPath, imgName)
                        tiles.append(pyvips.Image.new_from_file(img,
                                                                access="sequential"))
                print("1")
                im2 = pyvips.Image.arrayjoin(tiles, across=(abs(tmpRow)))
                imgName = "{}/{}_{}.jpeg".format(destination, quadrow, quadCol)
                im2.write_to_file(imgName)
                print("2")
                j += tmpCol
                quad += 1
                quadCol += 1
                
            quadrow += 1
            i += tmpRow
        print("3")
        lisOfFilesToBeDeleted = []
        tiles = []
        for y in range(0, quadCol):
            for x in range(0, quadrow):
                imgName = "{}_{}.jpeg".format(x, y)
                img = join(destination, imgName)
                lisOfFilesToBeDeleted.append(img)
                tiles.append(pyvips.Image.new_from_file(img,
                                                        access="sequential"))
        print("4")
        im2 = pyvips.Image.arrayjoin(tiles, across=quadrow)
        try:
            imageName = "{}/{}.tif[tile,pyramid,compression={}, bigtiff = True]".format(destination, slideName, compressionType)
            print("imageName : ", imageName)
            im2.write_to_file(imageName)
        except Exception as msg:
            print("Exception 111: ", msg)

        time.sleep(1)
        print("lisOfFilesToBeDeleted : ", lisOfFilesToBeDeleted)
        for images in lisOfFilesToBeDeleted:
            cmd = "rm -rf {}".format(images)
            os.system(cmd)

        print("Total Time Taken {} secs".format(time.time() - start))
        print()

    def create_folder(self, path):
        if not exists(path):
            os.mkdir(path)

    def close_window (self, root):
        root.destroy()

    def show_choice(self, v):
        self.input_val = v.get()
        print("self.input_val : ", self.input_val)


    def get_compression(self):
        root = tkinter.Tk()
        # child = root

        v = tkinter.IntVar()
        v.set(1)
        self.input_val = 1  # initializing the choice, i.e. Python

        languages = [
            ("none"),
            ("jpeg"),
            ("deflate"),
            ("lzw")
        ]
        root.title("Compress Bar")
        tkinter.Label(root, 
                text="""Select Compression type:""",
                justify = tkinter.LEFT,
                padx = 40,
                font = "Helvetica 18 bold").pack()

        for val, language in enumerate(languages):
            c = tkinter.Radiobutton(root, 
                        text=language,
                        padx = 20, 
                        variable=v, 
                        command=partial(self.show_choice, v),
                        value=val + 1)
            c.pack(anchor = tkinter.W)
        tkinter.Button(root, text ="ok", command = partial(self.close_window, root)).pack()

        root.mainloop()
TiffGen()