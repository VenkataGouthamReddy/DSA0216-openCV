import cv2

img = cv2.imread("image-3.png")
blur = cv2.GaussianBlur(img,(5,5),0)
sharp = cv2.addWeighted(img,1.5,blur,-0.5,0)

cv2.imshow("Unsharp", sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()
