import glob,os,sys
import shutil
import tarfile


def xtract_tar(slide_path):

    my_tar = tarfile.open(slide_path+"/other.tar",'r')
    for member in my_tar.getmembers():
        if not "raw_images" in member.name:
            print(member.name)
            my_tar.extract(member, slide_path+"/other")
            #my_tar.close()
    dst = "/home/adminspin/Music/scripts/snap/"+os.path.split(slide_path)[-1]
    # if not os.path.exists(dst):
    #     os.makedirs(dst)

    shutil.copytree(slide_path+"/other",dst)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]

    lst = ["JR-20-3960-D1-2_H01BBB30P-6361",
    "JR-20-3838-A23-1_H01BBB26P-6721",
    "JR-20-3953-A3-1_H01BBB30P-6362",
    "JR-20-3848-B1-1_H01BBB26P-6722",
    "JR-20-3848-A9-2_H01BBB26P-6723",
    "JR-20-5953-A1-10_H01EBB39P-22507",
    "JR-20-613-F1-3_H01EBB42P-16211",
    "JR-20-5349-B1-2_H01EBB42P-16207",
    "JR-20-497-B24-2_H01EBB42P-16205",
    "JR-20-342-A8-6_H01EBB42P-16204",
    "JR-20-497-B24-2_H01EBB42P-15937",
    "JR-20-5349-B1-2_H01EBB41P-15792",
    "JR-20-180-A1-5_H01EBB44P-16006",
    "JR-20-613-F1-3_H01EBB41P-15751",
    "JR-20-342-A8-6_H01EBB42P-15845",
    "JR-20-5518-C1-1_H01EBB45P-15302",
    "JR-20-5558-D18-3_H01BBB16P-47804",
    "JR-20-5349-B1-2_H01EBB38P-20553",
    "JR-20-5019-B5-6_H01BBB31P-27716",
    "JR-20-4349-A2-3_H01EBB44P-13218",
    "JR-20-4506-A12-1_H01EBB44P-13217",
    "JR-20-4349-A2-3_H01BBB16P-46022",
    "JR-20-4506-A12-1_H01DBB34P-25843",]

    for i in lst: #os.listdir(path):
        try:
            slide_path = path+"/"+i
            xtract_tar(slide_path)
        except Exception as msg:
            print(msg)
