# train.py

import os
import tensorflow as tf

# Twoje moduły
from propozycja_loadera import get_loaders, show_sample
from augmentation import apply_augmentation, get_augmentation_layer
from balancing import get_class_weights, print_class_distribution, analyze_balance


# ============================================================
# 1. WCZYTAJ DANE
# ============================================================
print("Wczytywanie danych...")
train_ds, val_ds = get_loaders(base_dir="data/balanced")

# ============================================================
# 2. ANALIZA
# ============================================================
print("\nAnaliza danych...")
print_class_distribution(train_ds)

is_balanced, ratio = analyze_balance(train_ds)
print(f"Zbalansowane: {'TAK' if is_balanced else 'NIE'} (ratio: {ratio:.2f})")

# ============================================================
# 3. OPCJE BALANSOWANIA
# ============================================================

# Jeśli dane JUŻ zbalansowane (data/balanced):
if is_balanced:
    print("\nDane zbalansowane - lekkie wagi lub brak")
    class_weights = get_class_weights(train_ds, smoothing=0.3)
    # LUB: class_weights = None

# Jeśli dane NIEzbalansowane (data/combined):
else:
    print("\nDane niezbalansowane - pełne wagi")
    class_weights = get_class_weights(train_ds, smoothing=1.0)

# ============================================================
# 4. AUGMENTACJA (OPCJA A - na datasecie)
# ============================================================
print("\nAplikowanie augmentacji...")
train_ds_aug = apply_augmentation(train_ds)
# val_ds BEZ augmentacji!
"""
# ============================================================
# 5. MODEL (OPCJA B - augmentacja w modelu)
# ============================================================
model = tf.keras.Sequential([
    # Augmentacja jako warstwa (alternatywa do apply_augmentation)
    # get_augmentation_layer(),  # odkomentuj jeśli NIE używasz apply_augmentation
    
    # CNN
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(43, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 6. TRENING
# ============================================================
print("\nTrening...")

history = model.fit(
    train_ds_aug,          # z augmentacją
    validation_data=val_ds, # BEZ augmentacji
    epochs=30,
    class_weight=class_weights,  # balansowanie
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]
)

print("\n✅ Trening zakończony!")
"""