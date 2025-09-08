import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load single known image
image_path = r"C:\Users\ganta\Desktop\home.jpg"
image = face_recognition.load_image_file(image_path)
encoding = face_recognition.face_encodings(image)[0]

known_face_encodings = [encoding]
known_face_names = ["Gowtham"]  # Change to your name

students = known_face_names.copy()

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Open CSV file
current_date = datetime.now().strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
writer = csv.writer(f)
writer.writerow(["Name", "Time"])

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.putText(frame, name + " Present", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, 3)

        if name in students:
            students.remove(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            writer.writerow([name, current_time])
            print(f"[ATTENDANCE] {name} at {current_time}")

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
