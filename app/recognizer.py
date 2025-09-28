import os

import cv2
from deepface import DeepFace

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
