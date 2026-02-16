import cv2

img = cv2.imread("image-4.png")
blur = cv2.GaussianBlur(img,(5,5),0)
highboost = cv2.addWeighted(img,2,blur,-1,0)

cv2.imshow("High Boost", highboost)
cv2.waitKey(0)
cv2.destroyAllWindows()
