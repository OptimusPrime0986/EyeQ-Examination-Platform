import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Start webcam
cap = cv2.VideoCapture(0)

no_face_start_time = None

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        face_count = 0

        if results.detections:
            face_count = len(results.detections)

            # Draw bounding boxes
            for detection in results.detections:
                mp.solutions.drawing_utils.draw_detection(frame, detection)

            no_face_start_time = None  # Reset timer

        else:
            # Start timer if no face detected
            if no_face_start_time is None:
                no_face_start_time = time.time()
            else:
                elapsed_time = time.time() - no_face_start_time
                if elapsed_time > 5:
                    cv2.putText(frame,
                                "NO FACE DETECTED - SUSPICIOUS",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2)

        # Multiple face alert
        if face_count > 1:
            cv2.putText(frame,
                        "MULTIPLE FACES DETECTED",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

        # Display face count
        cv2.putText(frame,
                    f"Faces Detected: {face_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("EyeQ Face Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()