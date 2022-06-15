import tarfile
import os,glob
count = 1

for i in glob.glob("/home/adminspin/Downloads/7th_shipment/*/*.tar"):
    my_tar = tarfile.open(i,'r')
    for member in my_tar.getmembers():
        if not "raw_images" in member.name:
            print(member.name)
            my_tar.extract(member, os.path.split(i)[0])
            #my_tar.close()
            print("**"*50)
            print("Slide Successfully Extracted : ",i.split('/')[-2],"\t COUNT :",count)
            count +=1
            print("**"*50)
