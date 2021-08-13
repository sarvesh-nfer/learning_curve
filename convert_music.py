import os 
directory = '/datadrive/wsi_data/compressed_data/H01CBA07P_13051/grid_1/raw_images/'
for filename in os.listdir(directory):
    prefix = filename.split(".bmp")[0]
    os.rename(filename, prefix+".jpeg")
