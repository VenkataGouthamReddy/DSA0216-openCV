import cv2
import numpy as np

img = cv2.imread("image-2.png", 0)

kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

laplacian = cv2.filter2D(img, -1, kernel)

sharpened = img - laplacian

cv2.imshow("Original", img)
cv2.imshow("Sharpened Image", sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
