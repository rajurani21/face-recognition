import face_recognition
import cv2
import numpy as np  
import os
from datetime import datetime   
video_capture = cv2.VideoCapture(0)

se_images = face_recognition.load_image_file("images/home.jpg")
se_encoding = face_recognition.face_encodings(se_images)[0]
known_face_encodings = [se_encoding,
                        ]

known_face_names = ["GR",
                    ]
students = known_face_names.copy()
face_locations = []
face_encodings = []
face_names = []
s = True

now= datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+")
Inwrite=csv.writer(f) 

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            if name in known_face_names:
                font=cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText=(10,100)
                fontScale=1.5
                fontColor=(0,255,0) 
                lineType=3
                thickness=3
            cv2.putText(frame,name + " Present", 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                Inwrite.writerow([name, current_time])

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()
