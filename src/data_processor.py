import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

def get_kaggle_instructions():
    """Zwraca instrukcję konfiguracji Kaggle API."""
    return """
Instrukcja konfiguracji Kaggle API:
1. Zaloguj się na https://www.kaggle.com/.
2. Przejdź do ustawień swojego konta (Settings).
3. W sekcji API kliknij 'Create New Token'.
4. Plik 'kaggle.json' zostanie pobrany na Twój komputer.
5. Umieść go w folderze:
   - Windows: C:\\Users\\<TwojaNazwaUzytkownika>\\.kaggle\\kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json
6. Upewnij się, że masz zainstalowaną bibliotekę kaggle: pip install kaggle
"""

def download_dataset(dest_path="data/raw", dataset_name="meowwwmyya/gtsrb-german-traffic-sign"):
    """Pobiera zbiór danych z Kaggle."""
    dest_path = Path(dest_path)
    if (dest_path / "Train").exists():
        print(f"Dane już istnieją w {dest_path}. Pomijam pobieranie.")
        return True

    print(f"Próba pobrania zbioru {dataset_name}...")
    try:
        import kaggle
        dest_path.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_name, path=dest_path, unzip=True)
        print("Pobieranie zakończone pomyślnie.")
        return True
    except Exception as e:
        print(f"Błąd podczas pobierania: {e}")
        print(get_kaggle_instructions())
        return False

def merge_train_test(raw_path="data/raw", combined_path="data/combined"):
    """
    Zadanie 1: Łączy zbiory Train i Test w jeden folder strukturyzowany klasami.
    Cel edukacyjny: ponowne ręczne podzielenie danych w przyszłości.
    """
    raw_path = Path(raw_path)
    combined_path = Path(combined_path)
    combined_path.mkdir(parents=True, exist_ok=True)

    train_path = raw_path / "Train"
    test_path = raw_path / "Test"
    test_csv = raw_path / "Test.csv"

    # 1. Kopiowanie Train (już strukturyzowany)
    if train_path.exists():
        print("Kopiowanie folderu Train do zbioru połączonego...")
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                dest_class_dir = combined_path / class_dir.name
                if dest_class_dir.exists():
                    shutil.rmtree(dest_class_dir)
                shutil.copytree(class_dir, dest_class_dir)
    
    # 2. Kopiowanie Test (wymaga mapowania z CSV do odpowiednich folderów klas)
    if test_path.exists():
        if test_csv.exists():
            print("Mapowanie i kopiowanie folderu Test przy użyciu Test.csv...")
            df = pd.read_csv(test_csv)
            for _, row in df.iterrows():
                img_path = raw_path / row['Path']
                class_id = str(row['ClassId'])
                dest_dir = combined_path / class_id
                dest_dir.mkdir(exist_ok=True)
                shutil.copy(img_path, dest_dir / img_path.name)
        else:
            print("Folder Test istnieje, ale brak Test.csv. Próba kopiowania bezpośredniego...")
            for class_dir in test_path.iterdir():
                if class_dir.is_dir():
                    dest_class_dir = combined_path / class_dir.name
                    dest_class_dir.mkdir(exist_ok=True)
                    for img in class_dir.iterdir():
                        shutil.copy(img, dest_class_dir / img.name)
    
    print(f"Zadanie 1 zakończone. Dane połączone w: {combined_path}")

def balance_dataset(data_dir="data/combined", target_dir="data/balanced", method="oversample"):
    """
    Zadanie 2: Wyrównuje liczebność klas, aby zmniejszyć nierówność reprezentacji.
    """
    data_dir = Path(data_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    class_counts = {}
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            class_counts[class_dir.name] = len(list(class_dir.glob("*")))

    if not class_counts:
        print("Brak danych do zbalansowania. Uruchom najpierw merge_train_test().")
        return

    max_samples = max(class_counts.values())
    print(f"Najliczniejsza klasa ma {max_samples} próbek. Metoda wyrównywania: {method}")

    for class_name, count in class_counts.items():
        src_class_dir = data_dir / class_name
        dest_class_dir = target_dir / class_name
        dest_class_dir.mkdir(exist_ok=True)

        images = list(src_class_dir.glob("*"))
        
        if method == "oversample":
            # Kopiujemy oryginały
            for img in images:
                shutil.copy(img, dest_class_dir / img.name)
            
            # Wyrównujemy do max_samples
            num_to_add = max_samples - count
            if num_to_add > 0:
                print(f"Klasa {class_name}: balansowanie (+{num_to_add} próbek).")
                for i in range(num_to_add):
                    img_to_copy = np.random.choice(images)
                    new_name = f"bal_{i}_{img_to_copy.name}"
                    shutil.copy(img_to_copy, dest_class_dir / new_name)

    print(f"Zadanie 2 zakończone. Dane zbalansowane w: {target_dir}")

def apply_oversampling(dataset, num_classes=43, seed=42):
    """
    Oversampling w pamięci na tf.data.Dataset.
    
    Strategia:
    1. Grupuje próbki według klas
    2. Znajduje najliczniejszą klasę
    3. Powtarza próbki z mniejszych klas 
    
    Args:
        dataset:     tf.data.Dataset (niebatchowany!)
        num_classes: liczba klas
        seed:        dla reprodukowalności
    
    Returns:
        tf.data.Dataset — zbalansowany
    """
    tf.random.set_seed(seed)
    
    # --- 1. Rozpakuj dataset do list per klasa ---
    class_datasets = {}
    class_counts = {}
    
    # Konwertuj do numpy (dla małych/średnich datasetów OK)
    images_list = [[] for _ in range(num_classes)]
    
    for image, label in dataset:
        label_int = int(label.numpy())
        images_list[label_int].append(image)
    
    # Policz próbki per klasa
    for i in range(num_classes):
        class_counts[i] = len(images_list[i])
    
    max_count = max(class_counts.values())
    print(f"Oversampling: max_count = {max_count}")
    
    # --- 2. Stwórz zbalansowane datasety per klasa ---
    balanced_datasets = []
    
    for class_id in range(num_classes):
        if class_counts[class_id] == 0:
            continue
            
        # Stack do tensora
        class_images = tf.stack(images_list[class_id])
        class_labels = tf.fill([len(images_list[class_id])], class_id)
        
        class_ds = tf.data.Dataset.from_tensor_slices((class_images, class_labels))
        
        # Repeat + take dla oversamplingu
        if class_counts[class_id] < max_count:
            # Powtarzaj w nieskończoność, weź max_count
            class_ds = class_ds.repeat().take(max_count)
        
        balanced_datasets.append(class_ds)
    
    # --- 3. Połącz wszystkie klasy ---
    balanced_ds = balanced_datasets[0]
    for ds in balanced_datasets[1:]:
        balanced_ds = balanced_ds.concatenate(ds)
    
    # --- 4. Shuffle ---
    total_samples = max_count * num_classes
    balanced_ds = balanced_ds.shuffle(buffer_size=min(total_samples, 50000), seed=seed)
    
    print(f"Oversampling zakończony: {total_samples} próbek")
    
    return balanced_ds

if __name__ == "__main__":
    # Scenariusz użycia dla zadań 1 i 2
    if download_dataset():
        merge_train_test(raw_path="data/raw", combined_path="data/combined")
        balance_dataset(data_dir="data/combined", target_dir="data/balanced")
