import cv2

img = cv2.imread("image-3.png")

if img is None:
    print("Image not found")
else:
    big = cv2.resize(img, None, fx=2, fy=2)
    small = cv2.resize(img, None, fx=0.5, fy=0.5)

    cv2.imshow("Big Image", big)
    cv2.imshow("Small Image", small)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
