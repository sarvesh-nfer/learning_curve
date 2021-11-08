from PIL import Image
import os
import glob

def imgcrop(input_p, xPieces, yPieces):
    filename, file_extension = os.path.splitext(input_p)
    im = Image.open(input_p)
    aoi = input_p.split('/')[-1].split('.')[0]
    imgwidth, imgheight = im.size
    height = imgheight // yPieces
    width = imgwidth // xPieces
    count = 0
    count2 = 0
    count3 = 0
    save_path = os.path.split(os.path.split(input_p)[0])[0] + "/cropped"
    if not os.path.exists(save_path+ "/"+ aoi):
        os.makedirs(save_path+ "/"+ aoi)
    
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            try:
                print(save_path+"/"+ aoi + "/"+aoi+"_blob_" + str(count2) + file_extension)
                a.save(save_path+"/"+ aoi + "/"+aoi+"_blob_" + str(count2) + file_extension)
                count +=1
                count2 += 5
                if count == 7:
                    print('changing')
                    count = 0
                    print("count : ",count)
                    count3 += 1
                    count2 = 0
                    count2 += count3
                    print("count2 : ",count2)
            except:
                pass


lst = glob.glob("/home/adminspin/Music/sarvesh/validation/*/grid_*/BI_bg/*.jpeg")

for sar in lst:
    imgcrop(sar,7,5)
    print("**"*50)
    print(f"AOI :{sar.split('/')[-1].split('.')[0]} Saved successfully ")
    print("**"*50)

