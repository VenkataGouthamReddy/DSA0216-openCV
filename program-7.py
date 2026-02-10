import cv2

cap = cv2.VideoCapture(0)   # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam Video", frame)

    # Change delay value to control speed
    # Slow motion → 100
    # Normal speed → 25
    # Fast motion → 5q
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
