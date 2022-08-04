import os
import sys

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("*.py <path_to_slide_folder> <path_to_white_imgs>")
        exit(0)

    white_img_folder = sys.argv[2]
    slide_folder = sys.argv[1]

    # first check whether white folder exists or not.
    if not os.path.exists(white_img_folder):
        print("White image folder doesn't exists.")
        exit(0)

    if not os.path.exists(slide_folder):
        print("Slide folder doesn't exists")
        exit(0)

    # now get all images from white folder.
    white_images = os.listdir(white_img_folder)

    os.rename(os.path.join(slide_folder, "white.bmp"), os.path.join(slide_folder, "white_used.bmp"))
    for white in white_images:
        # now first copy white image to white.bmp
        white_img_name = white[:white.find(".bmp")]

        # first create a folder
        tile_folder = os.path.join(slide_folder, white_img_name, "000")
        cmd = "mkdir -p " + tile_folder
        os.system(cmd)

        cmd = "cp " + os.path.join(white_img_folder, white) + " " + slide_folder
        os.system(cmd)

        # rename the white now.
        os.rename(os.path.join(slide_folder, white), os.path.join(slide_folder, "white.bmp"))

        # now run blending and tiling
        os.chdir("/home/adminspin/wsi_app/libs")

        cmd = "mpiexec -np 1 ./ut_spinvistaPanorama_mpi ./libspinvistaPanorama.so " + tile_folder + " 0 0 : -np 2 ./ut_spinvistaPanorama_mpi ./libspinvistaPanorama.so " + os.path.join(slide_folder, "panorama.xml") + " " + slide_folder + " 0"
        os.system(cmd)

        cmd = "./pyramid_generation " + os.path.join(tile_folder)
        os.system(cmd)

        cmd = "rm -rf " + os.path.join(slide_folder, "white.bmp")
        os.system(cmd)
