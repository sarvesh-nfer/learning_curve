import pandas as pd
import os,glob
import numpy as np

camera_board = []
camera_sensor = []

for sl in [68,69,70,71]:
    for i in glob.glob("/home/adminspin/Music/scripts/data/*"+str(sl)+"*/*.log"):
        with open(i) as file:
            for line in file:
                if 'Acquisition started for' in line:
                    name = line.split(" ")[-2].replace('\n','')
                if 'Micro Imaging Camera Board Temperature' in line:
                    board = line.split(" ")[-1].replace('\n','')
                if 'Micro Imaging Camera Sensor Temperature' in line:
                    sensor = line.split(" ")[-1].replace('\n','')

                    camera_board.append(float(board))
                    camera_sensor.append(float(sensor))

    print(sl)
    print(str(np.median(camera_sensor))+"/"+str(np.max(camera_sensor)))
# print("Median : ",np.median(camera_board))
# print("Max : ",np.max(camera_board))