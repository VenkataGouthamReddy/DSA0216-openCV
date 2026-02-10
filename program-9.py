import cv2

# Read the image
img = cv2.imread("image-1.png")

# Rotate clockwise (90 degrees)
clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Rotate counter-clockwise (90 degrees)
counter_clockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Clockwise Rotation", clockwise)
cv2.imshow("Counter Clockwise Rotation", counter_clockwise)

cv2.waitKey(0)
cv2.destroyAllWindows()
