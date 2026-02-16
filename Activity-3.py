import cv2
import matplotlib.pyplot as plt

# 1. Load image
img = cv2.imread("image-2.png")   # Replace with your image path
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Generate Gaussian Pyramid
level0 = img
level1 = cv2.pyrDown(level0)
level2 = cv2.pyrDown(level1)
level3 = cv2.pyrDown(level2)

# 3. Apply Canny Edge Detection at each level
edges0 = cv2.Canny(level0, 100, 200)
edges1 = cv2.Canny(level1, 100, 200)
edges2 = cv2.Canny(level2, 100, 200)
edges3 = cv2.Canny(level3, 100, 200)

# 4. Display results
titles = ['Level 0 - Original', 'Level 1 - Half',
          'Level 2 - Quarter', 'Level 3 - One-Eighth']

images = [edges0, edges1, edges2, edges3]

plt.figure(figsize=(10,8))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
