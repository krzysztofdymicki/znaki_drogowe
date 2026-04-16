"""
Budowa modelu CNN do klasyfikacji znakow drogowych GTSRB.

Architektura:
  [Augmentacja]  <- opcjonalna, aktywna TYLKO podczas treningu
  [Rescaling]    <- normalizacja 0-255 -> 0-1, zawsze aktywna
  Conv2D(32) x2 -> MaxPool -> Dropout
  Conv2D(64) x2 -> MaxPool -> Dropout
  Flatten -> Dense(256) -> Dropout -> Dense(43, softmax)

Uzycie:
    from cnn_builder import build_model
    model = build_model(use_augmentation=True)
    model = build_model(use_augmentation=False)  # bez augmentacji
"""

import tensorflow as tf


def build_model(input_shape=(32, 32, 3), num_classes=43, use_augmentation=True):
    """
    Buduje sekwencyjny model CNN z opcjonalna warstwa augmentacji.

    Warstwy augmentacji (RandomRotation, RandomZoom, RandomBrightness,
    RandomContrast) sa aktywne TYLKO podczas model.fit() - przy
    model.evaluate() i model.predict() sa automatycznie wylaczane.

    Args:
        input_shape:      rozmiar obrazu wejsciowego (H, W, C)
        num_classes:       liczba klas (43 dla GTSRB)
        use_augmentation:  True = dodaj warstwy augmentacji

    Returns:
        tf.keras.Sequential model (nieskompilowany)
    """
    model = tf.keras.Sequential(name="gtsrb_cnn")

    # --- Input ---
    model.add(tf.keras.layers.Input(shape=input_shape))

    # --- Augmentacja (opcjonalna, aktywna tylko w trybie treningowym) ---
    if use_augmentation:
        model.add(tf.keras.layers.RandomRotation(
            0.1, fill_mode="nearest", name="aug_rotation"
        ))
        model.add(tf.keras.layers.RandomZoom(
            0.1, fill_mode="nearest", name="aug_zoom"
        ))
        model.add(tf.keras.layers.RandomBrightness(
            0.2, name="aug_brightness"
        ))
        model.add(tf.keras.layers.RandomContrast(
            0.2, name="aug_contrast"
        ))

    # --- Normalizacja (zawsze) ---
    model.add(tf.keras.layers.Rescaling(1.0 / 255, name="normalization"))

    # --- Blok konwolucyjny 1 ---
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1a"))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1b"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), name="pool1"))
    model.add(tf.keras.layers.Dropout(0.25, name="drop1"))

    # --- Blok konwolucyjny 2 ---
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2a"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2b"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), name="pool2"))
    model.add(tf.keras.layers.Dropout(0.25, name="drop2"))

    # --- Klasyfikator ---
    model.add(tf.keras.layers.Flatten(name="flatten"))
    model.add(tf.keras.layers.Dense(256, activation="relu", name="dense1"))
    model.add(tf.keras.layers.Dropout(0.5, name="drop3"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="output"))

    return model


if __name__ == "__main__":
    print("=== Model Z augmentacja ===")
    m1 = build_model(use_augmentation=True)
    m1.summary()

    print("\n=== Model BEZ augmentacji ===")
    m2 = build_model(use_augmentation=False)
    m2.summary()
