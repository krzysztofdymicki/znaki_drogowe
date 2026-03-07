import numpy as np 
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# import danych
def create_dataset(
    base_dir: str,
    img_size=(32, 32),
    batch_size=32,
    seed=42,
    shuffle=True,
    cache=False,
    normalize=False
):
    # Foldery w kolejności alfabetycznej (tak robi image_dataset_from_directory)
    class_names_alphabetical = sorted(os.listdir(base_dir))
    # Tworzymy mapowanie alfabetyczny idx -> numer folderu
    class_to_numeric = {i: int(name) for i, name in enumerate(class_names_alphabetical)}

    # Tworzymy lookup table
    keys = tf.constant(list(class_to_numeric.keys()), dtype=tf.int32)
    values = tf.constant(list(class_to_numeric.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1
    )

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )
   # Mapowanie labeli alfabetycznych -> numeryczne
    dataset = dataset.map(lambda x, y: (x, table.lookup(y)))

    #zapis danych do RAMu (szybsze uczenie w kolejnych epokach)
    if cache:
        dataset = dataset.cache()
    
    #optymalizacja wczytywania paczki danych 
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# Sprawdzenie czy dane zostały dobrze wczytane i jak wyglądają. 
def show_sample_images(dataset, class_names=None, n=9):
    plt.figure(figsize=(8,8))
    for images, labels in dataset.take(1):  # pobieramy jeden batch
        for i in range(n):
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if class_names:
                plt.title(class_names[labels[i].numpy()])
            else:
                plt.title(int(labels[i].numpy()))
            plt.axis("off")
    plt.show()

#wczytanie danych
# Budujemy ścieżkę relatywną do folderu 'data/raw/Train'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "raw", "Train")

if not os.path.exists(data_path):
    print(f"BŁĄD: Nie znaleziono danych w: {data_path}")
else:
    dataset = create_dataset(data_path)
# test poprawności wczytania
show_sample_images(dataset, class_names=None, n=9)