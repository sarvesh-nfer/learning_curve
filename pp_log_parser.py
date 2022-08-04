import glob
import pandas as pd

aoi = {}
status = {}
offset = {}

for i in glob.glob("/wsi_app/service_logs/post_processing/*.log"):
    print(i)
    with open(i) as file:
        for line in file:
            
            if 'is replace' in line:
                status[i.split("/")[-1].split(".")[0]] = line.split(" ")[-1].strip()
            if "AOI NAME =" in line:
                aoi[i.split("/")[-1].split(".")[0]] = line.split(" ")[-1].strip()
            if "Offset values :" in line:
                offset[i.split("/")[-1].split(".")[0]] = line.split(":")[-1].strip()

                
print(len(aoi),len(status),len(offset))

a = pd.DataFrame(aoi.items(),columns=['slide_name','aoi'])
s = pd.DataFrame(status.items(),columns=['slide_name','status'])
o = pd.DataFrame(offset.items(),columns=['slide_name','offset'])


df = pd.merge((pd.merge(a,s,on='slide_name')),o,on='slide_name')


df.to_csv("/home/adminspin/Music/aoi_used.csv",index=False)
