import cv2
import numpy as np

img = cv2.imread("image-3.png")
h, w = img.shape[:2]

pts1 = np.float32([[50,50], [w-50,50], [50,h-50], [w-50,h-50]])
pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])

M = cv2.getPerspectiveTransform(pts1, pts2)
out = cv2.warpPerspective(img, M, (w, h))

cv2.imshow("Original", img)
cv2.imshow("Perspective Image", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
