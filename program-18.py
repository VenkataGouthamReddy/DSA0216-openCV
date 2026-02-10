import cv2

img = cv2.imread("image-2.png", 0)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

cv2.imshow("Original", img)
cv2.imshow("Sobel Y", sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()
