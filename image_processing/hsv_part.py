# import cv2

# def showImage(image, name):
#     cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(name, 600, 600)
#     cv2.imshow(name, image)
#     cv2.waitKey(0)

# img_hsv = cv2.imread("/home/adminspin/Desktop/QC_X4417_2_hsv.png")

# mask = cv2.inRange(img_hsv, (0, 0, 4), (255, 5, 255))

# showImage(mask,"hsv_mask")

# showImage(img_hsv,"hsv_og")

import cv2
import numpy as np
from skimage import segmentation

# read image
img = cv2.imread('/home/adminspin/Desktop/QC_X4417_2_hsv.png')

# convert image to hsv colorspace
hsv = cv2.imread('/home/adminspin/Desktop/QC_X4417_2_hsv.png')
h, s, v = cv2.split(hsv)

# threshold saturation image
s2 = cv2.GaussianBlur(s,(5,5),0)
thresh1 = cv2.threshold(s2, 0, 255, cv2.THRESH_OTSU)[1]

segmented_img1 = segmentation.mark_boundaries(img, thresh1)
segmented_nlmd = cv2.normalize(segmented_img1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

# threshold value image and invert
thresh2 = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)[1]
thresh2 = 255 - thresh2

# combine the two threshold images as a mask
mask = cv2.add(thresh1,thresh2)

# use mask to remove lines in background of input
result = img.copy()
result[mask==0] = (255,255,255)

# display IN and OUT images
# cv2.imshow('IMAGE', img)
# cv2.imshow('SAT', s)
# cv2.imshow('VAL', v)
cv2.imshow('THRESH1', thresh1)
cv2.imshow('THRESH2', thresh2)
cv2.imshow('MASK', mask)
cv2.imshow('segmented_nlmd', segmented_nlmd)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save output image
cv2.imwrite('symbols_thresh1.png', thresh1)
cv2.imwrite('symbols_thresh2.png', thresh2)
cv2.imwrite('symbols_mask.png', mask)
cv2.imwrite('symbols_cleaned.png', result)
