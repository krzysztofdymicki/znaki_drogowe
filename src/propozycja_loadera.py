# PROPOZYCJA ROZSZERZENIA DATA_LOADER.PY
# Poniższe funkcje wykorzystują teraz zbalansowany zbiór i dodają normalizację obrazów w locie.

import os
import tensorflow as tf
import matplotlib.pyplot as plt

def get_loaders(
    base_dir="data/balanced", # Domyślnie korzystamy z nowej, wyrównanej bazy danych
    img_size=(32, 32),
    batch_size=32,
    seed=42
):
    """
    Przygotowuje zestawy danych do treningu i walidacji.
    Rozszerza oryginalną logikę o podział i normalizację (Zadania 3 i 4).
    """
    
    # 1. Ładowanie zbioru treningowego (wykorzystujemy wbudowany validation_split)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    # 2. Ładowanie zbioru walidacyjnego (20% z pozostałych po balansowaniu)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    # Zadanie 3: Normalizacja (Rescaling)
    # Możemy to zrobić tutaj lub bezpośrednio jako pierwszą warstwę w modelu Sequential.
    # Robienie tego tutaj optymalizuje proces, jeśli używamy cache().
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Optymalizacja wydajności (z Twojego oryginalnego loadera)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

def show_sample(dataset, n=9):
    """Podgląd próbek po normalizacji 0-1."""
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(n):
            plt.subplot(3, 3, i + 1)
            # images[i] jest teraz float [0,1], plt.imshow poradzi sobie z tym idealnie
            plt.imshow(images[i].numpy())
            plt.title(f"Klasa: {int(labels[i])}")
            plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Test propozycji po uruchomieniu procesora danych
    if os.path.exists("data/balanced"):
        print("Ładowanie zbalansowanego i znormalizowanego zbioru...")
        train, val = get_loaders()
        print("Gotowe! Można zacząć trenować model.")
        show_sample(train)
    else:
        print("BŁĄD: Folder data/balanced nie istnieje. Uruchom najpierw src/data_processor.py!")
