import cv2
import numpy as np

# Step 1: Read image
image = cv2.imread("image-1.png")

if image is None:
    print("Error: Image not found")
else:
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Step 4: Find contours
    contours, _ = cv2.findContours(edges,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    print("Total Objects Detected:", len(contours))

    for contour in contours:
        
        area = cv2.contourArea(contour)

        # Ignore very small objects
        if area < 1000:
            continue
        
        # Approximate contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(approx)

        # Shape classification
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Unknown"

        # Draw contour
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
        
        # Put label
        cv2.putText(image, shape,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    # Show results
    cv2.imshow("Original Image", image)
    cv2.imshow("Edge Image", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
