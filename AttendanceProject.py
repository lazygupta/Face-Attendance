import cv2
import numpy as np
import face_recognition
import os
import pyttsx3 as textspeech
from datetime import datetime

aniarya=textspeech.init() # object creation
voices = aniarya.getProperty('voices') # getting details of current voice

rate = aniarya.getProperty('rate')  # getting details of current speaking rate
aniarya.setProperty('rate', 150) # setting up new voice rate

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) # Just remove .jpg from the images names
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #To convert our imported image from BGR into RGB colour format
        encode = face_recognition.face_encodings(img)[0] # Encoding our images in ImagesAttendance Folder
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f: # open Attendance.csv as read file
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%d/%m/%y,%I:%M %p')
            Present = str('Yes')
            f.writelines(f'\n{name},{dtString},{Present}')
            aniarya.setProperty('voice', voices[0].id)
            statement = str('Welcome to Class' + name)
            aniarya.say(statement)
            aniarya.runAndWait()


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read() # Read the captured image
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # Resizing our captured images
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS) #to store bounding box coordinates
    #print(facesCurFrame)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #Used to compare encodings of previous image with captured image
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #Used to find Face Distance
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.60: #checking that the distance is less than 0.6 or not for accuracy
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
            aniarya.say('Please next')
            aniarya.runAndWait()

        # For making bounding box in live detection
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 300, 0) , 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 300, 0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)