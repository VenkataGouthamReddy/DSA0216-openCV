import cv2

img = cv2.imread("image-4.png")

h, w, c = img.shape   # get image size

# Crop region
crop = img[50:200, 50:200]

ch, cw, cc = crop.shape   # get crop size

# Safe paste location (inside image boundary)
x = 200
y = 200

# Check boundary before pasting
if y+ch <= h and x+cw <= w:
    img[y:y+ch, x:x+cw] = crop
else:
    print("Image too small for this paste location!")

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
