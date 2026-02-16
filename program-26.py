import cv2

img = cv2.imread("image-1.png")
cv2.putText(img,"WATERMARK",(50,50),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow("Watermark",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
