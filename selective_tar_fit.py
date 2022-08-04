import glob,os
import tarfile
import pandas as pd

output = "/home/adminspin/Desktop/new"
if not os.path.exists(output):
    os.makedirs(output)

# df = pd.read_csv("/home/adminspin/Music/scripts/8th_above.csv")
# #df = df[df['_source.data.cluster_name'] == "CS001"]
txt  =["JR-20-3709-B1-2_H01DBB35P-20253"]
for i in txt:#df['_source.data.slide_name']:
    try:
        output_path = os.path.join(output,i)            
        if not os.path.exists(output):
            os.makedirs(output)
        path = "/home/adminspin/Desktop/new/"+i+"/other.tar"
        cmd1 = "tar xvf " + path + " -C " + output_path + " " +"2d_fit_images"
        print(cmd1)
        status = os.system(cmd1)
        print("**"*50,path.split("/")[-1],"**"*50)
    except Exception as msg:
        print("*Error*"*10,path.split("/")[-1],"*Error*"*50)
