import cv2

img = cv2.imread("image-6.png")

cv2.rectangle(img,(100,100),(300,300),(0,255,0),2)

roi = img[100:300,100:300]

cv2.imshow("Rectangle",img)
cv2.imshow("Extracted Object",roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
