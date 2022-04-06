import os,glob

txt =  glob.glob("/datadrive/wsi_data/data_for_validation/*/*/class/*/*.jpeg")
print(txt)
for i in txt:
    if os.path.exists(i):
        os.remove(i)
    else:
        print("The file does not exist")