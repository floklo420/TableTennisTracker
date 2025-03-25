import glob

label_files = glob.glob(r"C:\Users\LetsP\Desktop\Ausbildung\YOLOrepo\TableTennisTracker\train\labels\*.txt")

for file in label_files:
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            if values:
                class_id = int(values[0])
                if class_id < 0 or class_id >= 3:
                    print(f"Fehler in Datei {file}: Ungültige class_id {class_id}")
