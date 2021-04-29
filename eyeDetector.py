import cv2
from random import randrange
# Real Time Image
video = cv2.VideoCapture(0)
# Eye Classifier
faceClassifier = cv2.CascadeClassifier('frontalFaceDetector.xml')
eyeClassifier = cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
    readSuccessful, frame = video.read()
    if not readSuccessful:
        break
    # Converting Image to Gray Scale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Getting Frame Coordinates
    faceCoordinates = faceClassifier.detectMultiScale(grayFrame)
    eyesCoordinates = eyeClassifier.detectMultiScale(grayFrame)
    print('Face Coordinates - ',faceCoordinates)
    print('Eyes Coordinates - ',eyesCoordinates)
    #Drawing Rectangles
    for (x,y,w,h) in faceCoordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)),3)
        for (x,y,w,h) in eyesCoordinates:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)),1)
    cv2.imshow('Eye Detection', frame)
    key = cv2.waitKey(1)
    # Stop if Q is pressed
    if key==81 or key==113:
        break
video.release()
print('Code Completed')