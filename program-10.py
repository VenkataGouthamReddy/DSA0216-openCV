import cv2
import numpy as np

img = cv2.imread("image-1.png")
h, w = img.shape[:2]

M = np.float32([[1, 0, 50],
                [0, 1, 30]])

move = cv2.warpAffine(img, M, (w, h))

cv2.imshow("Original", img)
cv2.imshow("Moved Image", move)

cv2.waitKey(0)
cv2.destroyAllWindows()
