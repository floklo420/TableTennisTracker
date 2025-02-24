import cv2
import torch
from ultralytics import YOLO
import time

# Überprüfen, ob CUDA verfügbar ist und das Gerät entsprechend einstellen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Modell laden und auf das gewählte Gerät verlagern
model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\runs\detect\train13\weights\best.pt')  # Pfad anpassen
model.to(device)

# Zielklassen definieren (Namen überprüfen)
target_classes = [key for key, value in model.names.items() if value in ('plate', 'racket', 'ttball')]

print(f"Zielklassen-IDs: {target_classes}")

# Videodatei öffnen (Pfad anpassen)
video_path = r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\newData\videoTTBall\video_2025-02-12_13-12-37.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Fehler beim Öffnen der Videodatei.")
    exit()

# Framerate des Videos auslesen
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Verzögerung pro Frame in Millisekunden


confidence_threshold = 0.5  # Mindestkonfidenz für eine Erkennung


# Video analysieren
while True:
    start_time = time.time()  # Startzeit des Frames
    
    ret, frame = cap.read()
    if not ret:
        print("Ende des Videos erreicht oder Fehler beim Lesen des Frames.")
        break

    # YOLO-Erkennung (Frame auf das richtige Gerät senden)
    results = model(frame, device=device)

    # Ergebnisse filtern und zeichnen
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Klasse des Objekts
            if cls in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Ergebnisse filtern und zeichnen
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        conf = box.conf[0]  # Konfidenzwert
                        cls = int(box.cls[0])  # Klasse des Objekts
                        if cls in target_classes and conf > confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = f'{model.names[cls]} {conf:.2f}'
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Frame anzeigen
    cv2.imshow('YOLOv11 Video Object Detection', frame)

    # Verzögerung berechnen und hinzufügen
    elapsed_time = (time.time() - start_time) * 1000  # Verstrichene Zeit in ms
    delay = max(1, frame_delay - int(elapsed_time))  # Restverzögerung
    if cv2.waitKey(delay) & 0xFF == ord('q'):  # Drücke 'q', um die Analyse zu beenden
        break

cap.release()
cv2.destroyAllWindows()
