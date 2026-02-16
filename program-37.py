import cv2

cap = cv2.VideoCapture("video-1.mp4")
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

for frame in reversed(frames):
    cv2.imshow("Reverse",frame)
    if cv2.waitKey(30)==27:
        break

cap.release()
cv2.destroyAllWindows()
