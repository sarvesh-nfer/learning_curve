#python GIF MAKER

import glob
import imageio

filenames =  glob.glob("/home/adminspin/Desktop/validation/H01FBA08R_2441/grid_2/FM_6_7/*.jpeg")

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/home/adminspin/Desktop/validation/H01FBA08R_2441/grid_2/FM_6_7.gif', images)