# from skimage.data import camera
# from skimage.util import img_as_float
import cv2
#import function for marking boundaries
from skimage.segmentation import mark_boundaries
from skimage import segmentation
#importing needed libraries
import skimage.segmentation
from matplotlib import pyplot as plt

img = cv2.imread("/home/adminspin/Downloads/IP_report_15thnov/QC_X4418_12/whiteCorrectedInput.png")
res3 = skimage.segmentation.felzenszwalb(img, scale=100,sigma=1)
res4 = skimage.segmentation.felzenszwalb(img, scale=500,sigma=1)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(mark_boundaries(img, res3)); ax1.set_xlabel("With k=100")
ax2.imshow(mark_boundaries(img, res4)); ax2.set_xlabel("With k=500")
fig.suptitle("Plotting the boundaries")