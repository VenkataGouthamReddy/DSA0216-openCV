import cv2

img = cv2.imread("image-2.png")

if img is None:
    print("Error: Image not found or path is incorrect")
else:
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imshow("Original Image", img)
    cv2.imshow("Gaussian Blur Image", blur_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
