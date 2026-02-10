import cv2
import numpy as np

# Read image
img = cv2.imread("image-1.png")
rows, cols = img.shape[:2]

# Original points
pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])

# New points
pts2 = np.float32([[10, 100],
                   [200, 50],
                   [100, 250]])

# Get affine matrix
M = cv2.getAffineTransform(pts1, pts2)

# Apply affine transformation
affine = cv2.warpAffine(img, M, (cols, rows))

# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Affine Transformed Image", affine)

cv2.waitKey(0)
cv2.destroyAllWindows()
