
import cv2
import torch
from ultralytics import YOLO
import time

# Gerät einstellen (auf dem Raspberry Pi ist CUDA in der Regel nicht verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Modell laden, auf das Gerät übertragen und in den Evaluationsmodus versetzen
model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLOrepo\TableTennisTracker\runs\detect\train\weights\best.pt')  # Pfad anpassen
model.to(device)
model.eval()

# Zielklassen definieren (Namen überprüfen)
target_classes = [key for key, value in model.names.items() if value in ('plate', 'racket', 'ttball')]
print(f"Zielklassen-IDs: {target_classes}")

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

    # Inferenz ohne Gradientenberechnung
    with torch.no_grad():
        results = model(frame, device=device)

    # Ergebnisse filtern und zeichnen
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls in target_classes and conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLO Video Object Detection', frame)

    # Berechnung der Restverzögerung pro Frame
    elapsed_time = (time.time() - start_time) * 1000  # in Millisekunden
    delay = max(1, frame_delay - int(elapsed_time))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()