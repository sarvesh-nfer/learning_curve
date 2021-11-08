import pandas as pd
import os
import glob

input_path = glob.glob("/datadrive/wsi_data/onex_images/*/onex_white_ref.png")
print(len(input_path))
lst1=[]
for slide in input_path:
    lst1.append(slide.split("/")[-2])
print(len(lst1))
print(lst1)
# slide_list = os.listdir(input_path)
# count = 0
# lst1 = []
# for slide in slide_list[:100]:
#     white_p = os.path.join(input_path,slide,'onex_white_ref.png')
#     if not os.path.exists(white_p):
#         print("NO REF error : ",slide)
#         lst1.append(slide)
#     else:
#         print("existing ref : ",slide)

df = pd.DataFrame(lst1)
df.to_csv('/home/adminspin/Desktop/df_yes.csv',index=False)