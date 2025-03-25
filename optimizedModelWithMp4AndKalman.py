import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np

# Gerät einstellen (auf dem Raspberry Pi ist CUDA in der Regel nicht verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Modell laden, auf das Gerät übertragen und in den Evaluationsmodus versetzen
model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLOrepo\TableTennisTracker\runs\detect\train\weights\best.pt')  # Pfad anpassen
model.to(device)
model.eval()

# Zielklassen definieren (Namen überprüfen)
target_classes = [int(key) for key, value in model.names.items() if value in ('plate', 'racket', 'ttball')]
print(f"Zielklassen-IDs: {target_classes}")

# Ermittlung der Klasse-ID für "plate"
plate_class_id = None
for key, value in model.names.items():
    if value == "plate":
        plate_class_id = int(key)
        break

if plate_class_id is None:
    print("Klasse 'plate' nicht im Modell gefunden.")
    exit()

# Kalman-Filter für "plate" initialisieren
kalman_plate = cv2.KalmanFilter(4, 2)
kalman_plate.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
kalman_plate.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
kalman_plate.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman_initialized = False  # Wird auf True gesetzt, wenn erstmalig eine "plate" gemessen wurde
plate_box_size = None       # Speichert die zuletzt gemessene Boxgröße von "plate"

# Videodatei öffnen (Pfad anpassen)
video_path = r'C:\Users\LetsP\Desktop\Ausbildung\YOLOrepo\TableTennisTracker\TestVid\newVid.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Fehler beim Öffnen der Videodatei.")
    exit()

# Framerate des Videos auslesen
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Verzögerung pro Frame in Millisekunden

confidence_threshold = 0.5  # Mindestkonfidenz für eine Erkennung

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Ende des Videos erreicht oder Fehler beim Lesen des Frames.")
        break

    # Optional: Bildauflösung reduzieren (z. B. um 50 % verkleinern)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # Flag, ob "plate" in diesem Frame erkannt wurde
    plate_measurement = None

    # Inferenz ohne Gradientenberechnung
    with torch.no_grad():
        results = model(frame, device=device)

    # Ergebnisse filtern und zeichnen
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Für "plate" nur den Kalman-Filter updaten (keine grüne Box zeichnen)
            if cls == plate_class_id and conf > confidence_threshold:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                plate_measurement = np.array([[np.float32(center_x)],
                                              [np.float32(center_y)]])
                plate_box_size = (x2 - x1, y2 - y1)
            # Andere Klassen normal zeichnen
            elif cls in target_classes and conf > confidence_threshold:
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Kalman-Filter-Aktualisierung bzw. Vorhersage für "plate"
    if plate_measurement is not None:
        if not kalman_initialized:
            # Initialisierung des Zustands: [x, y, 0, 0]
            kalman_plate.statePre = np.array([[plate_measurement[0][0]],
                                              [plate_measurement[1][0]],
                                              [0],
                                              [0]], np.float32)
            kalman_initialized = True
        kalman_plate.correct(plate_measurement)
    else:
        # Falls "plate" nicht erkannt wurde, erfolgt kein Update – Kalman liefert die Vorhersage
        pass

    # Zeichne für "plate" nur die Kalman-Vorhersage (blaue Box)
    if kalman_initialized:
        prediction = kalman_plate.predict()
        pred_x, pred_y = prediction[0][0], prediction[1][0]
        if plate_box_size is not None:
            w, h = plate_box_size
            top_left = (int(pred_x - w / 2), int(pred_y - h / 2))
            bottom_right = (int(pred_x + w / 2), int(pred_y + h / 2))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(frame, "Kalman Plate", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('YOLO Video Object Detection', frame)

    # Berechnung der Restverzögerung pro Frame
    elapsed_time = (time.time() - start_time) * 1000  # in Millisekunden
    delay = max(1, frame_delay - int(elapsed_time))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
