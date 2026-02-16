import cv2

# Load video
cap = cv2.VideoCapture("video-1.mp4")

# Load built-in cascade (no download required)
car_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

if car_cascade.empty():
    print("Error loading cascade file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Draw rectangles
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
