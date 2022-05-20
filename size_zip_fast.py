import pandas as pd
import os,glob

sar ={}

for i in glob.glob("/mnt/clusterNas/dicom_data/*/*.zip"):
    sar[i.split("/")[-2]] = os.path.getsize(i)

    break

print(sar)
