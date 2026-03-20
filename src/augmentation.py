# augmentation.py
"""
Augmentacja danych dla TensorFlow/Keras.
Kompatybilne z data_loader.py (dane znormalizowane 0-1).
"""

import tensorflow as tf


def get_augmentation_layer():
    """
    Zwraca warstwę augmentacji dla danych ZNORMALIZOWANYCH (0-1).
    
    WAŻNE: Twój loader już normalizuje obrazy do [0,1],
    więc augmentacja musi działać na tym zakresie.
    
    Działa TYLKO podczas treningu (training=True).
    Podczas ewaluacji/predykcji obrazy przechodzą bez zmian.
    
    Returns:
        tf.keras.Sequential z warstwami augmentacji
    
    Example:
        >>> train_ds, val_ds = get_loaders()  # dane już 0-1
        >>> train_ds = apply_augmentation(train_ds)
        >>> 
        >>> # LUB w modelu:
        >>> model = tf.keras.Sequential([
        ...     get_augmentation_layer(),
        ...     tf.keras.layers.Conv2D(32, 3, activation='relu'),
        ...     ...
        ... ])
    """
    return tf.keras.Sequential([
        # === Geometryczne ===
        tf.keras.layers.RandomRotation(
            factor=0.04,           # ±15 stopni
            fill_mode='nearest'
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15),
            fill_mode='nearest'
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
            fill_mode='nearest'
        ),
        
        # === Fotometryczne ===
        # RandomBrightness i RandomContrast działają poprawnie na [0,1]
        tf.keras.layers.RandomBrightness(
            factor=0.2,            # ±20% jasności
            value_range=(0.0, 1.0) # zakres danych wejściowych
        ),
        tf.keras.layers.RandomContrast(
            factor=0.2             # ±20% kontrastu
        ),
        
    ], name='augmentation')


def apply_augmentation(dataset):
    """
    Aplikuje augmentację do tf.data.Dataset.
    
    UWAGA: Stosuj TYLKO do danych treningowych!
    Dane walidacyjne/testowe NIE powinny być augmentowane.
    
    Args:
        dataset: tf.data.Dataset z (images, labels), 
                 obrazy znormalizowane do [0,1]
    
    Returns:
        Dataset z augmentacją
    
    Example:
        >>> train_ds, val_ds = get_loaders()
        >>> train_ds = apply_augmentation(train_ds)  # ✓ augmentacja
        >>> # val_ds bez zmian                       # ✓ bez augmentacji
    """
    augmentation_layer = get_augmentation_layer()
    
    return dataset.map(
        lambda images, labels: (augmentation_layer(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE
    )


def visualize_augmentation(dataset, n_samples=5, n_augmentations=4):
    """
    Wizualizuje efekt augmentacji na przykładowych obrazach.
    
    Args:
        dataset: Dataset z obrazami (znormalizowane 0-1)
        n_samples: Liczba przykładowych obrazów
        n_augmentations: Liczba augmentacji na obraz
    """
    import matplotlib.pyplot as plt
    
    augmentation = get_augmentation_layer()
    
    for images, labels in dataset.take(1):
        images = images[:n_samples]
        labels = labels[:n_samples]
        
        fig, axes = plt.subplots(
            n_samples,
            n_augmentations + 1,
            figsize=(3 * (n_augmentations + 1), 3 * n_samples)
        )
        
        for i in range(n_samples):
            # Oryginalny obraz
            axes[i, 0].imshow(images[i].numpy())  # [0,1] - matplotlib OK
            axes[i, 0].set_title(f'Oryginał\nKlasa: {labels[i].numpy()}')
            axes[i, 0].axis('off')
            
            # Augmentacje
            for j in range(n_augmentations):
                augmented = augmentation(
                    tf.expand_dims(images[i], 0),
                    training=True
                )[0]
                
                # Clip do [0,1] na wszelki wypadek
                augmented = tf.clip_by_value(augmented, 0.0, 1.0)
                
                axes[i, j + 1].imshow(augmented.numpy())
                axes[i, j + 1].set_title(f'Aug {j + 1}')
                axes[i, j + 1].axis('off')
        
        plt.suptitle('Wizualizacja augmentacji', fontsize=14)
        plt.tight_layout()
        plt.show()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import os
    import sys
    
    # Dodaj ścieżkę do katalogu src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from data_loader import get_loaders
    
    if os.path.exists("data/balanced"):
        print("Test augmentacji...")
        
        train_ds, val_ds = get_loaders()
        
        print("\n1. Wizualizacja augmentacji:")
        visualize_augmentation(train_ds, n_samples=3, n_augmentations=4)
        
        print("\n2. Test aplikowania augmentacji do datasetu:")
        train_ds_aug = apply_augmentation(train_ds)
        
        for images, labels in train_ds_aug.take(1):
            print(f"   Batch shape: {images.shape}")
            print(f"   Value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        
        print("\n✅ Test zakończony pomyślnie!")
    else:
        print("BŁĄD: Folder data/balanced nie istnieje!")