import cv2
import numpy as np

img = cv2.imread("image-4.png")
h, w = img.shape[:2]

src = np.float32([[10,10], [w-10,10], [10,h-10], [w-10,h-10]])
dst = np.float32([[100,50], [w-100,20], [80,h-80], [w-50,h-50]])

H, _ = cv2.findHomography(src, dst, 0)   # DLT method
out = cv2.warpPerspective(img, H, (w, h))

cv2.imshow("Original", img)
cv2.imshow("DLT Image", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
