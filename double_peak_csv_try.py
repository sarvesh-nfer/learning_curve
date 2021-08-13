import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import glob
import re

path = input("Enter Folder path : ")

images=os.listdir(path)
# path = '/home/adminspin/Desktop/office/double_peak/20937.5_28000.0_blob_fs.csv'
# output_path = '/home/adminspin/Desktop/office/double_peak/20937.5_28000.0_blob_fs_plot.jpeg'
height_threshold = 6
for slide_name in images:
    print("name = ", slide_name)
    slide_path_input = os.path.join(path, slide_name)
    src = glob.glob(os.path.join(slide_path_input,"grid_*","debug_data","*_fs.csv"))
    
    for j in src:
        data = pd.read_csv(j)
        #print(data)
        data = data[:35]
        output = j
        print("-"*100)
        print(output)
        print("-"*100)
        plt.style.use('ggplot')  
        fig, axs = plt.subplots(5, 7, figsize=(30,20))
        print(len(data))
        c = -1
        for i in range(len(data)):
            if i%5 == 0:
                r=0
                c+=1
                count = 0
            if count <= 4 and count!=0 :
                r+=1
            plt.style.use('ggplot')           
            axs[r,c].plot(range(1,data.shape[1]+1), 
                            np.ones((data.shape[1]))*height_threshold, color='#696969')
            axs[r,c].plot(range(1,data.shape[1]+1),data.iloc[i,:].values , marker='.', color = 'navy')
            axs[r, c].set_title('blob_index: '+str(i))
            axs[r,c].set_xticks(range(1,data.shape[1]+1,2))
            axs[r,c].set_xlabel('Stacks')
            axs[r,c].set_ylabel('Focus Metric')
            count+=1
        plt.tight_layout()
        #plt.show()
        #plt.close()
        #plt.clf()
        #output_path = output_path +slide_name+str(a)+".jpeg"
        output = os.path.split(j)[0]
        a = os.path.split(j)[-1].split('.')[0]
        print(output+"/"+a+".png")
        plt.savefig(output+"/"+a+".png")
        plt.close()
        plt.clf()
        #output_path = "/home/adminspin/Desktop/output_image/"
