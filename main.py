import cv2
import face_recognition
import os

path = 'images'
images = []
classNames = []

for file in os.listdir(path):
    img = face_recognition.load_image_file(f'{path}/{file}')
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

def encodeImages(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = encodeImages(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = faceDis.argmin()

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = [v*4 for v in faceLoc]

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, name, (x1, y2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()