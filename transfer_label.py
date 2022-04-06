import os,glob
import shutil
from sqlite3 import SQLITE_DELETE

txt = glob.glob("/home/adminspin/Music/nimhans/*/loc_output_data/input_for*.png")
print(txt)
for i in txt:
    slide = i.split('/')[-3]
    dst = "/home/adminspin/Music/nimhans_labels"+"/"+slide
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    shutil.copy2(i,dst)