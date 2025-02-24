import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

# CUDA-Gerät setzen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Modell laden
model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\runs\detect\train13\weights\best.pt')
model.to(device)

# Zielklassen: Jetzt ALLE wieder tracken!
target_classes = {key: value for key, value in model.names.items() if value in ('plate', 'racket', 'ttball')}
print(f"Zielklassen: {target_classes}")

# Kalman-Filter für den Ball
kalman = cv2.KalmanFilter(4, 2)  # 4 Zustände (x, y, dx, dy) | 2 Messwerte (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Video öffnen
video_path = r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\newData\videoTTBall\video_2025-02-12_13-12-37.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Fehler beim Öffnen der Videodatei.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)

confidence_threshold = 0.5
kalman_initialized = False

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Ende des Videos erreicht oder Fehler beim Lesen des Frames.")
        break

    results = model(frame, device=device)
    
    ball_detected = False
    measurement = None

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            cls = int(box.cls[0])
            label_name = model.names[cls]

            if conf > confidence_threshold and label_name in target_classes.values():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                if label_name == "ttball":  # Nur den Ball mit Kalman tracken
                    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                    ball_detected = True
                    
                    if not kalman_initialized:
                        kalman.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                        kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                        kalman_initialized = True

                    kalman.correct(measurement)

                # Zeichne Bounding Boxen für ALLE Objekte
                color = (0, 255, 0) if label_name == "ttball" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label_name} {conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Falls der Ball nicht erkannt wurde, Kalman-Schätzung anzeigen
    if not ball_detected and kalman_initialized:
        prediction = kalman.predict()
        predicted_x, predicted_y = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (predicted_x, predicted_y), 10, (0, 0, 255), -1)
        cv2.putText(frame, "Predicted", (predicted_x, predicted_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('YOLO + Kalman Tracking', frame)

    elapsed_time = (time.time() - start_time) * 1000
    delay = max(1, frame_delay - int(elapsed_time))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
