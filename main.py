import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
path = 'images'
attendance_file = 'attendance.csv'
# Keep the image name as same as of the person's name
images = []
classNames = []

# =========================
# LOAD IMAGES
# =========================
for file in os.listdir(path):
    file_path = os.path.join(path, file)

    # Skip non-image files
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        continue

    img = face_recognition.load_image_file(file_path)
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

# =========================
# ENCODE KNOWN FACES
# =========================
def encodeImages(images):
    encodeList = []
    validClassNames = []

    for img, name in zip(images, classNames):
        try:
            encodings = face_recognition.face_encodings(img)

            if len(encodings) > 0:
                encodeList.append(encodings[0])
                validClassNames.append(name)
            else:
                print(f"[WARNING] No face found in image: {name}")
        except Exception as e:
            print(f"[ERROR] Could not encode {name}: {e}")

    return encodeList, validClassNames

encodeListKnown, classNames = encodeImages(images)
print("Encoding Complete")

# =========================
# ATTENDANCE FUNCTIONS
# =========================
def initialize_attendance_file():
    """
    Create the attendance CSV file if it doesn't exist.
    """
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Attendance"])
        df = df.astype(str)
        df.to_csv(attendance_file, index=False)

def load_attendance():
    """
    Load attendance CSV into DataFrame as strings only.
    """
    if os.path.exists(attendance_file):
        return pd.read_csv(attendance_file, dtype=str).fillna("")
    else:
        initialize_attendance_file()
        return pd.read_csv(attendance_file, dtype=str).fillna("")

def save_attendance(df):
    """
    Save attendance DataFrame safely to CSV.
    """
    df = df.fillna("")
    df = df.astype(str)
    df.to_csv(attendance_file, index=False)

def prepare_today_sheet():
    """
    Ensure all students have a row for today.
    If a student's row for today doesn't exist, create it as Absent.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    df = load_attendance()

    for name in classNames:
        # Check if this student already has an entry for today
        exists = ((df["Name"] == name) & (df["Date"] == today_date)).any()

        if not exists:
            new_row = pd.DataFrame([{
                "Name": name,
                "Date": today_date,
                "Time": "",
                "Attendance": "Absent"
            }])
            df = pd.concat([df, new_row], ignore_index=True)

    save_attendance(df)

def markAttendance(name):
    """
    Mark a student present only once per day.
    If already present today, do nothing.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    df = load_attendance()

    # Find today's row for the student
    mask = (df["Name"] == name) & (df["Date"] == today_date)

    if mask.any():
        row_index = df[mask].index[0]

        # Only update if currently absent
        if df.at[row_index, "Attendance"] != "Present":
            df.at[row_index, "Attendance"] = "Present"
            df.at[row_index, "Time"] = current_time
            print(f"[ATTENDANCE] {name} marked Present at {current_time}")
        else:
            # Already present today -> ignore repeated detection
            pass
    else:
        # Safety fallback (in case today's row wasn't created somehow)
        new_row = pd.DataFrame([{
            "Name": name,
            "Date": today_date,
            "Time": current_time,
            "Attendance": "Present"
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        print(f"[ATTENDANCE] {name} marked Present at {current_time}")

    save_attendance(df)

# =========================
# INITIALIZE TODAY'S ATTENDANCE
# =========================
initialize_attendance_file()
prepare_today_sheet()

# =========================
# START WEBCAM
# =========================
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("[ERROR] Could not read from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        name = "UNKNOWN"

        if len(faceDis) > 0:
            matchIndex = faceDis.argmin()

            # Stricter threshold for better reliability
            if matches[matchIndex] and faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex]
                markAttendance(name)

        y1, x2, y2, x1 = [v * 4 for v in faceLoc]

        # Green for known, red for unknown
        color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name.upper(), (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Webcam Attendance System', img)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()