import cv2

import cv2
import numpy as np

img = cv2.imread("/home/adminspin/Music/sarvesh/newly/aoi0803.bmp")
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

cv2.imshow('Original',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Red',r)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('green',g)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('blue',b)

cv2.waitKey(0)
cv2.destroyAllWindows()


print("variance for Original image : ",cv2.Laplacian(img, cv2.CV_64F).var())
print("variance for Red channel : ",cv2.Laplacian(r, cv2.CV_64F).var())
print("variance for Green channel : ",cv2.Laplacian(g, cv2.CV_64F).var())
print("variance for Blue channel : ",cv2.Laplacian(b, cv2.CV_64F).var())