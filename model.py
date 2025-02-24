import cv2
import torch
from ultralytics import YOLO

# Überprüfen, ob CUDA verfügbar ist und das Gerät entsprechend einstellen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Modell laden und auf das gewählte Gerät verlagern
#model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\krassesGitRepo\TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch\checkpoints\ttnet\ttnet_best.pth')
model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\runs\detect\train6\weights\best.pt')  # Pfad anpassen
model.to(device)

# Zielklassen definieren (Namen überprüfen)
#target_classes = [key for key, value in model.names.items() if value == 'ttball']  # Hier 'ttball' eintragen
target_classes = [key for key, value in model.names.items() if value in ('ttball', 'racket', 'plate')]


print(f"Zielklassen-IDs: {target_classes}")

# Webcam öffnen
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fehler beim Öffnen der Kamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Frames.")
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

    # Frame anzeigen
    cv2.imshow('YOLOv11 Live Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
