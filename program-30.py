import cv2
import numpy as np

img = cv2.imread("image-4.png",0)
kernel = np.ones((5,5),np.uint8)

dilation = cv2.dilate(img,kernel)

cv2.imshow("Dilation",dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
