import glob,os
import tarfile

output_path = "/hdd_drive/lost+found/log_all"
if not os.path.exists(output_path):
    os.makedirs(output_path)

for i in glob.glob("/mnt/clusterNas/dicom_data/*/other.tar"):
    try:

        slide_name = i.split("/")[-2]
        cmd1 = "tar xvf " + i + " -C " + output_path + " " + slide_name +".log"
        status = os.system(cmd1)
        print("**"*50,i.split("/")[-1],"**"*50)
    except Exception as msg:
        print("msg",msg)