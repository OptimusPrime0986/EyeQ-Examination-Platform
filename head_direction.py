import cv2
import mediapipe as mp
import numpy as np
import time

direction_start_time = None
current_tracked_direction = None

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# Variables
previous_direction = "Looking Straight"
head_turn_count = 0
look_down_count = 0
risk_score = 0

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        head_direction = "No Face"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                nose = face_landmarks.landmark[1]
                nose_x = int(nose.x * frame_width)
                nose_y = int(nose.y * frame_height)

                cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

                # Determine direction
                if nose_x < frame_width * 0.4:
                    head_direction = "Looking Left"
                elif nose_x > frame_width * 0.6:
                    head_direction = "Looking Right"
                elif nose_y > frame_height * 0.65:
                    head_direction = "Looking Down"
                else:
                    head_direction = "Looking Straight"

                # Count head turns (only when direction changes)
                # Detect prolonged suspicious direction
                if head_direction in ["Looking Left", "Looking Right", "Looking Down"]:

                    if current_tracked_direction != head_direction:
                        current_tracked_direction = head_direction
                        direction_start_time = time.time()

                    else:
                        duration = time.time() - direction_start_time

                        if duration > 3:   # 3 seconds threshold
                            risk_score += 5
                            cv2.putText(frame,
                                        "PROLONGED LOOK DETECTED",
                                        (20, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (0, 0, 255),
                                        2)

                else:
                    current_tracked_direction = None
                    direction_start_time = None

        # Calculate risk score
        risk_score = (head_turn_count * 2) + (look_down_count * 3)

        # Display Info
        cv2.putText(frame,
                    f"Direction: {head_direction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.putText(frame,
                    f"Head Turns: {head_turn_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Look Down Count: {look_down_count}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Risk Score: {risk_score}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        cv2.imshow("EyeQ Head Behavior Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()