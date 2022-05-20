import glob,os
import shutil

dst = "/home/adminspin/Music"
for i in glob.glob("/hdd_drive/post_log/*/centering_check.csv"):
    try:

        folder = i.split("/")[-2]

        final_dst = os.path.join(dst,folder)
        if not os.path.exists(final_dst):
            os.makedirs(final_dst)

        shutil.copy2(i,final_dst)
    
    except Exception as msg:
        print("error in ",i)
        
