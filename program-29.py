import cv2
import numpy as np

img = cv2.imread("image-2.png",0)
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img,kernel)

cv2.imshow("Erosion",erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
