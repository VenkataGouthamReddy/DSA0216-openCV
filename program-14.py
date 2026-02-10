import cv2
import numpy as np

img = cv2.imread("image-2.png")
h, w = img.shape[:2]

src = np.float32([[0,0], [w,0], [0,h], [w,h]])
dst = np.float32([[50,50], [w-100,30], [80,h-50], [w-30,h-30]])

H, status = cv2.findHomography(src, dst)
out = cv2.warpPerspective(img, H, (w, h))

cv2.imshow("Original", img)
cv2.imshow("Homography Image", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
