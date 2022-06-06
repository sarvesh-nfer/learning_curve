import pandas as pd
import glob

count = 0

sname = []
thickness = []

for i in glob.glob("/home/adminspin/Music/failed2acquire_2june/debug/*/*.log"):
#     print(i)
    with open(i) as file:
        for line in file:
            if 'Slide thickness is:' in line:
                print(line)
                thickness.append(line.split(":")[-1].strip())
                sname.append(i.split("/")[-1].split('.')[0].strip())
print(len(sname),len(thickness))
df = pd.DataFrame(list(zip(sname, thickness)),
               columns =['slide_name', 'thickness'])
df.to_csv("/home/adminspin/Music/scripts/thickness.csv",index=False)