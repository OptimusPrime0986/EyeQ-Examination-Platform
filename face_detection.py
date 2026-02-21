import cv2
import mediapipe as mp

#Initialize Mediapipe Face_detection
mp_face_detection = mp.solutions.face_detection

#Start webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        #Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Detect faces
        results = face_detection.process(rgb_frame)
        
        face_count = 0
        
        if results.detections:
            face_count = len(results.detections)
            
            for detection in results.detections:
                mp.solutions.drawing_utils.draw_detection(frame, detection)
                
        cv2.putText(frame,
                    f"Faces Detected : {face_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)
        
        cv2.imshow("EyeQ Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
        
                