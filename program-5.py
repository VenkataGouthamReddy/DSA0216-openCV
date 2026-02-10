import cv2
import numpy as np

img = cv2.imread("image-5.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)

eroded_img = cv2.erode(gray, kernel, iterations=1)

cv2.imshow("Original Image", img)
cv2.imshow("Eroded Image", eroded_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
