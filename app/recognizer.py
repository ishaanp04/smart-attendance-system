import os
from datetime import datetime

import cv2
import pandas as pd
from deepface import DeepFace

ATTENDANCE_FILE = "attendance.xlsx"


def log_attendance(student_name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # If file exists, load it. Otherwise, create new DataFrame
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Date", "Time", "Student", "Status"])

    # Avoid duplicate entries (same student, same date)
    if not ((df["Date"] == date) & (df["Student"] == student_name)).any():
        new_entry = pd.DataFrame(
            [[date, time, student_name, "Present"]],
            columns=["Date", "Time", "Student", "Status"],
        )
        df = pd.concat([df, new_entry], ignore_index=True)

        # Save back to Excel
        df.to_excel(ATTENDANCE_FILE, index=False)
        print(f"[LOGGED] {student_name} marked present at {time}")
    else:
        print(f"[SKIP] {student_name} already marked for {date}")


# Path to dataset
DATASET_PATH = "dataset"
PLACEHOLDER_IMG = "dataset/2022UCB6037/2022UCB6037_0.jpg"

print("[INFO] Building face database...")
# DeepFace can create embeddings database automatically
DeepFace.find(
    img_path=PLACEHOLDER_IMG,
    db_path=DATASET_PATH,
    enforce_detection=False,
    # detector_backend="opencv",
)
print("[INFO] Database built successfully!")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save current frame temporarily
    cv2.imwrite("temp.jpg", frame)

    try:
        # Search for the face in dataset
        result = DeepFace.find(
            img_path="temp.jpg",
            db_path=DATASET_PATH,
            enforce_detection=False,
            # detector_backend="opencv",
        )

        if len(result) > 0 and not result[0].empty:
            # Get folder name (student name) from file path
            identity_path = result[0]["identity"][0]
            student_name = os.path.basename(os.path.dirname(identity_path))
            cv2.putText(
                frame,
                f"Recognized: {student_name}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            log_attendance(student_name)
        else:
            cv2.putText(
                frame,
                "Unknown",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    except Exception as e:
        cv2.putText(
            frame,
            "No face detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
