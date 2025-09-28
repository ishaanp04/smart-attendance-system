import os

import cv2


def capture_faces(student_name, num_samples=20):
    # Create dataset folder if not exists
    dataset_path = "dataset"
    student_path = os.path.join(dataset_path, student_name)
    os.makedirs(student_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"[INFO] Capturing {num_samples} images for {student_name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Faces", frame)

        # Save every frame as image until count reaches num_samples
        if cv2.waitKey(1) & 0xFF == ord("c"):  # press 'c' to capture
            img_name = f"{student_name}_{count}.jpg"
            img_path = os.path.join(student_path, img_name)
            cv2.imwrite(img_path, frame)
            print(f"[SAVED] {img_path}")
            count += 1

        # Exit if enough images are taken
        if count >= num_samples:
            break

        # Quit anytime with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Capture complete.")


if __name__ == "__main__":
    student = input("Enter student name/ID: ")
    capture_faces(student, num_samples=20)
