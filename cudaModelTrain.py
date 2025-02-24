import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO


# Überprüfen, ob GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training auf: {device}")

# Beispiel-Dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)  # 10 Features
        self.labels = torch.randint(0, 2, (size,))  # Binäre Labels (0 oder 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Modell-Definition
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # 10 Input-Features -> 2 Output-Klassen
    
    def forward(self, x):
        return self.fc(x)

# Dataset und DataLoader
dataset = DummyDataset(size=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # batch_size auf 100 gesetzt

# Modell, Loss und Optimizer initialisieren
model = YOLO(r'C:\Users\LetsP\Desktop\Ausbildung\YOLO\yolo11n.pt')  # Modell auf GPU übertragen
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Verzeichnis für Trainingsergebnisse erstellen
save_dir = 'runs/train6'  # Hier kannst du den Ordnernamen nach Belieben anpassen
os.makedirs(save_dir, exist_ok=True)

# Trainingsschleife
epochs = 1000
for epoch in range(epochs):
    model.train()  # Setzt das Modell in den Trainingsmodus
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Daten auf GPU übertragen
        
        # Vorwärtsdurchlauf
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Rückwärtsdurchlauf und Optimierung
        optimizer.zero_grad()  # Gradient zurücksetzen
        loss.backward()        # Backpropagation
        optimizer.step()       # Parameter aktualisieren
        
        total_loss += loss.item()
    
    # Nach jeder Epoche das Modell speichern
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print(f"Training abgeschlossen. Modelle werden unter {save_dir} gespeichert.")
