import glob,os
import tarfile

output_path = "/home/adminspin/Music/scripts/log_all"

for i in glob.glob("/mnt/clusterNas/dicom_data/*/other.tar"):
    try:
        cmd1 = "tar xvf " + i + " -C " + output_path + " /" + i.split("/")[-1]
        status = os.system(cmd1)
        print("**"*50,i.split("/")[-1],"**"*50)
    except Exception as msg:
        print("*Error*"*10,i.split("/")[-1],"*Error*"*50)
