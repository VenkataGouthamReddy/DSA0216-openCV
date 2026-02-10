import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    pts1 = np.float32([[50,50], [w-50,50], [50,h-50], [w-50,h-50]])
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, M, (w, h))

    cv2.imshow("Original Video", frame)
    cv2.imshow("Perspective Video", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
