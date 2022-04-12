import tarfile
import os,glob

txt = glob.glob("/home/adminspin/Music/stitching_test_slides/*/*.tar")

count = 1
for i in txt:
    try:
        my_tar = tarfile.open(i)
        my_tar.extractall(os.path.split(i)[0])
        my_tar.close()
        print("**"*50)
        print("Slide Successfully Extracted : ",i.split('/')[-2],"\t COUNT :",count)
        count +=1
        print("**"*50)
    except Exception as msg:
        print("--"*50)
        print("Couldn't Extract Slide : ",i.split('/')[-2],"/t msg :",msg)
        print("--"*50)
