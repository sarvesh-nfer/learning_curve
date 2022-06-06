import pandas as pd
import glob


count = 0
lstx = []
lsty = []
sname = []
for i in glob.glob("/datadrive/wsi_data/xyz_1june/*/*.log"):
    with open(i) as file:
        for line in file:
            if 'Resulted XY shift(microns):' in line:
                print(line)
                lstx.append(line.split(":")[-1].split(',')[0].strip())
                lsty.append(line.split(":")[-1].split(',')[-1].strip())
                sname.append(i.split("/")[-1].split('.')[0].strip())


print(len(sname),len(lstx),len(lsty))
df = pd.DataFrame(list(zip(sname, lstx,lsty)),
               columns =['slide_name', 'x_shift','y_shift'])
df.to_csv("/home/adminspin/Music/scripts/xyz_1june.csv",index=False)
