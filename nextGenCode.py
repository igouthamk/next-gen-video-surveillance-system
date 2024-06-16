import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

# Load pre-trained Haar cascade classifiers for detecting faces and full bodies
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Variables for motion detection
prev_frame = None
motion_detected = False

# Variables for video recording
detection = False
detection_stopped_time = None
timer_started = False
SECONDS_AFTER_DETECTION = 5
f_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Function to detect motion in the frame
def detect_motion(frame):
    global prev_frame, motion_detected
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Apply Gaussian blur to reduce noise
    
    if prev_frame is None:
        prev_frame = gray
        return False
    
    # Compute absolute difference between current frame and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours of the moving objects
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    
    # Check if any contour has area greater than a threshold
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            motion_detected = True
            break
    
    prev_frame = gray
    return motion_detected

# Main loop
while True:
    _, frame = cap.read()
    
    # Check for motion
    motion_detected = detect_motion(frame)
    
    # Detect faces and bodies in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) + len(bodies) > 0 or motion_detected:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, f_size)
            print("Started Recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Recording Stopped!")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)
    
    cv2.imshow("Smart Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
