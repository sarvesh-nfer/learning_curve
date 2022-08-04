import os, sys,glob

import time
import numpy as np



count = 0

txt = glob.glob("/hdd_drive/lost+found/log_all/*H01*.log")

file1 = open("/hdd_drive/lost+found/myfile.txt","w")
file1.writelines(["slide_name", ",board", ",sensor",",timestamp","\n"])
t1 = start_time = time.time()
for i in txt:
    try:
        camera_board = np.array([],dtype="float")
        camera_sensor = np.array([],dtype="float")
        start_time = time.time()
        count +=1
        with open(i) as file:
            for line in file:
                if '[DEBUG] :  ' in line:
                    timestamp = line.split(" ")[1].split("[")[-1].split("]")[0]
                if 'Acquisition started for' in line:
                    name = line.split(" ")[-2].strip()
                if 'Micro Imaging Camera Board Temperature' in line: 
                    camera_board = np.append(camera_board,line.split(" ")[-1].strip())
                if 'Micro Imaging Camera Sensor Temperature' in line:
                    camera_sensor = np.append(camera_sensor,line.split(" ")[-1].strip())

        print("%s taken to parse" % round((time.time() - start_time),2),"slide no. : ",\
              count," out of : ",len(txt))
        
        if count % 100 == 0:
            print("***Time taken for 100 slides*** :",round((time.time() - t1),2),"seconds")
        
        file1.writelines([name,",",str(np.max(camera_board.astype("float"))),\
                        ",",str(np.max(camera_sensor.astype("float"))),\
                        ",",timestamp,"\n"])

    except Exception as msg:
        print("msg",msg)