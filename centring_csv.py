import os,glob
import shutil

txt = glob.glob("/hdd_drive/post_log/*/centering_check.csv")

for i in txt:
    date = i.split("/")[-2].strip()

    save_path = os.path.join("/home/adminspin/Desktop/post_logs",date)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    shutil.copy2(i,save_path)
    print("COPIED for : ", date)

