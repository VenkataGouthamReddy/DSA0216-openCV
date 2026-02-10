import cv2

cap = cv2.VideoCapture("video-1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)

    # Change value to control speed
    # Slow motion: 100
    # Fast motion: 5
    cv2.waitKey(5)

cap.release()
cv2.destroyAllWindows()
