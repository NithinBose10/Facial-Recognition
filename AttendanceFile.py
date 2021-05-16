import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path1 = 'Images'
images = []
classNames = []
myList = os.listdir(path1)
# print(myList)

for cls in myList:
    curImg = cv2.imread("{}/{}".format(path1, cls))
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
# print(classNames)


def findEncodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markPeople(name):
    with open('people.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now1 = datetime.now()
            dateStr = now1.strftime("%H:%M:%S")
            f.writelines("\n {} \n {}".format(name, dateStr))


encodeListKnown = findEncodings(images)
print('Encoding Done!')

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceFrame = face_recognition.face_locations(imgS)
    encode = face_recognition.face_encodings(imgS, faceFrame)

    for encodeFace, faceLoc in zip(encode, faceFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markPeople(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeElonTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
#
# result = face_recognition.compare_faces([encodeElon], encodeElonTest)


