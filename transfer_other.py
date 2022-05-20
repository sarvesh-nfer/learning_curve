import pandas as pd
import shutil,os

path = "/mnt/clusterNas/dicom_data"

output = "/home/adminpin/Music"

df = pd.read_csv("")

for i in df['slide_name']:
    output_path = output + "/" +i

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = path + "/" + i +"/other.tar"
    
    shutil.copy2(input_path, output_path)

