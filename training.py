from ultralytics import YOLO
import torch


if __name__ == '__main__':

    gimmeDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf: {gimmeDevice}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    
    model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\yolo11n.pt').to(device)

    # Training starten
    model.train(
        data=r"C:\Users\LetsP\Desktop\Ausbildung\YOLO\dataset.yaml",  # Pfad zu deiner Dataset-Datei
        epochs=100,  # Anzahl der Trainingsepochen
        imgsz=640,  # Bildgröße
        batch=16,  # Batch-Größe
        workers=6,  # Anzahl der Worker für Datenvorbereitung (CPU Kerne)
        device=device,  # Training auf der GPU
        # project='runs/train',  # Basisordner für das Training (Standard: 'runs/train')
        # name='train1'  # Der Name für das Trainingsverzeichnis, z.B. 'train1', 'train2' usw.
    )



# # Lade das Modell (vorgefertigte YOLOv11-Weights)
# model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\yolo11n.pt').cuda()  # Pfad zum Pretrained YOLO-Modell

# gimmeDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Training auf: {gimmeDevice}")

# # Training starten
# model.train(
#     data=r"C:\Users\LetsP\Desktop\Ausbildung\YOLO\dataset.yaml",  # Pfad zu deiner Dataset-Datei
#     epochs=100,  # Anzahl der Trainingsepochen
#     imgsz=640,  # Bildgröße
#     batch=16,  # Batch-Größe
#     workers=4,  # Anzahl der Worker für Datenvorbereitung
#     device='cuda:0',  # Training auf der GPU
#     #project='runs/train',  # Basisordner für das Training (Standard: 'runs/train')
#     #name='train1'  # Der Name für das Trainingsverzeichnis, z.B. 'train1', 'train2' usw.
# )
