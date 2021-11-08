import pandas as pd
import sqlite3
import os,glob
import shutil
from PIL import Image

def imgcrop(input_p, xPieces, yPieces):
    filename, file_extension = os.path.splitext(input_p)
    im = Image.open(input_p)
    aoi = input_p.split('/')[-1].split('.')[0]
    imgwidth, imgheight = im.size
    height = imgheight // yPieces
    width = imgwidth // xPieces
    count = 0
    save_path = os.path.split(os.path.split(input_p)[0])[0] + "/cropped"
    if not os.path.exists(save_path+ "/"+ aoi):
        os.makedirs(save_path+ "/"+ aoi)
    
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            try:
                print(save_path+"/"+ aoi+ "/-" + str(count) + file_extension)
                a.save(save_path+"/"+ aoi + "/"+aoi+"_blob_" + str(count) + file_extension)
                count += 1
            except:
                pass

def database_glob(slide_path):

    db_path = glob.glob(slide_path+'/*db')[0]
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    df1 = pd.read_sql_query("select * from acquisition_blob_info ;", conn)
    df2 = pd.read_sql_query("select * from aoi ;",conn)

    df2 = df2[['aoi_name','grid_id','best_idx']]

    df3 = pd.merge(df2,df1,on=['aoi_name','grid_id'])

    df = df3[df3['stack_index'] == df3['best_idx']]

    df['actual'] = df['aoi_name'] + "_blob_" + df['blob_index']+".jpeg"

    path = glob.glob(slide_path+"/grid_*/cropped/*/*.jpeg")
    for i in path:
        try:
            a = os.path.split(i)[-1]
            g = i.split('/')[-4].split('_')[-1]
            s = i.split('/')[-5]
            if df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] > 6:
                dst = slide_path +"/grid_"+g+"/FM_l6"
                print("grid : ",g,"\t a : ",a)
                print(dst)
                if not os.path.exists(dst):
                    os.mkdir(dst)
                shutil.copy2(i,dst)
            
            else if df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] > 6 and \
            df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] < 7:
                
                dst = slide_path +"/grid_"+g+"/FM_6_7"
                print("grid : ",g,"\t a : ",a)
                print(dst)
                if not os.path.exists(dst):
                    os.mkdir(dst)
                shutil.copy2(i,dst)
            
            else if df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] > 7 and \
            df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] < 8:
                
                dst = slide_path +"/grid_"+g+"/FM_7_8"
                print("grid : ",g,"\t a : ",a)
                print(dst)
                if not os.path.exists(dst):
                    os.mkdir(dst)
                shutil.copy2(i,dst)

            else if df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] > 8 and \
            df[(df['grid_id'] == int(g)) & (df['actual'] == a)].iloc[0][5] <= 9:
                
                dst = slide_path +"/grid_"+g+"/FM_8_9"
                print("grid : ",g,"\t a : ",a)
                print(dst)
                if not os.path.exists(dst):
                    os.mkdir(dst)
                shutil.copy2(i,dst)
            else:
                (" More than 9 : ",i )
        except Exception as msg:
            print(a," has a problem","\t :",msg)


if __name__ == "__main__":
    
    # lst = glob.glob("/home/adminspin/Desktop/validation/*/grid_*/BI_bg/*.jpeg")
    # for sar in lst:
    # imgcrop(sar, 7, 5)
