import pandas as pd

txt = "/home/adminspin/Downloads/log_xyz_shift/JR-20-1304-A2-1_H01BBB24P-43166.log"

count = 0
lstx = []
lsty = []
sname = []
with open(txt) as file:
    for line in file:
        if 'Resulted XY shift(microns):' in line:
            print(line)
            lstx.append(line.split(":")[-1].split(',')[0].strip())
            lsty.append(line.split(":")[-1].split(',')[-1].strip())
            sname.append(txt.split("/")[-1].split('.')[0].strip())


print(len(sname),len(lstx),len(lsty))
df = pd.DataFrame(list(zip(sname, lstx,lsty)),
               columns =['slide_name', 'x_shift','y_shift'])
df.to_csv("/home/adminspin/Music/scripts/xyz.csv",index=False)