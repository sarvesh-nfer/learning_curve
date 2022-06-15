import glob,os
import tarfile
import pandas as pd

output_path = "/home/adminspin/Music/scripts/log_c1"
if not os.path.exists(output_path):
    os.makedirs(output_path)

df = pd.read_csv("/home/adminspin/Music/scripts/8th_above.csv")
df = df[df['_source.data.cluster_name'] == "CS001"]

for i in df['_source.data.slide_name']:
    try:
        path = "/mnt/clusterNas/dicom_data/"+i+"/other.tar"
        cmd1 = "tar xvf " + path + " -C " + output_path + " " + i +".log"
        print(cmd1)
        status = os.system(cmd1)
        print("**"*50,path.split("/")[-1],"**"*50)
    except Exception as msg:
        print("*Error*"*10,path.split("/")[-1],"*Error*"*50)
