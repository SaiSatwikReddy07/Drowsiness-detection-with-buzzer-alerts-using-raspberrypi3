import cv2
import numpy as np
import dlib
from imutils import face_utils
import RPi.GPIO as GPIO
import time

capture = cv2.VideoCapture(0)
detector_ = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# States initialization
sleep = 0
drowsy = 0
active = 0
# Displaying the status in the specified color
status = ""
color = (0, 0, 0)

# Set up GPIO pin for the buzzer
buzzer_pin = 14
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

def euclidean_distance(point1, point2):
    distance = np.linalg.norm(point1 - point2)
    return distance

def blink_detection(l1, l2, l3, l4, l5, l6):
    ver = euclidean_distance(l2, l4) + euclidean_distance(l3, l5)
    hor = euclidean_distance(l1, l6)
    eye_aspect_ratio = ver / (2.0 * hor)

    if eye_aspect_ratio > 0.25:
        return 2
    elif eye_aspect_ratio > 0.21 and eye_aspect_ratio <= 0.25:
        return 1
    else:
        return 0

def activate_buzzer():
    GPIO.output(buzzer_pin, GPIO.HIGH)
    time.sleep(0.5)  # Adjust the delay as needed
    GPIO.output(buzzer_pin, GPIO.LOW)

while True:
    _, frame = capture.read()
    gray_scale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_frame = frame.copy()

    faces = detector(gray_scale_image)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray_scale_image, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye_blink = blink_detection(landmarks[36], landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_eye_blink = blink_detection(landmarks[42], landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        if left_eye_blink == 0 or right_eye_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "Sleepy"
                color = (255, 0, 0)
                activate_buzzer()

        elif left_eye_blink == 1 or right_eye_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy"
                color = (0, 0, 255)
                activate_buzzer()

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active"
                color = (0, 255, 0)
        	
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("State of the person", frame)
    cv2.imshow("Face detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27: #ASCII value for 'esc' key
        break

# Clean up GPIO on exit
GPIO.cleanup()
cv2.destroyAllWindows()