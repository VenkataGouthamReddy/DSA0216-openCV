import cv2

img = cv2.imread("image-1.png", 0)

edges = cv2.Canny(img, 100, 200)

cv2.imshow("Original", img)
cv2.imshow("Canny Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
