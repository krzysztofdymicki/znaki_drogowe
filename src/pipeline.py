"""
Glowny pipeline treningowy GTSRB.

Prawidlowy podzial danych:
  - Trening/walidacja: data/raw/Train -> data/train_balanced (oversampling)
  - Test (held-out):   data/raw/Test + Test.csv -> data/test/{klasa}/

Uzycie:
    python src/pipeline.py                          # trening + ewaluacja, augmentacja ON
    python src/pipeline.py --no-augmentation        # trening + ewaluacja, augmentacja OFF
    python src/pipeline.py --eval-only MODEL_PATH   # sama ewaluacja zapisanego modelu
    python src/pipeline.py --epochs 50 --lr 0.0005  # custom hiperparametry
"""

import os
import sys
import json
import shutil
import argparse
import datetime

import numpy as np
import pandas as pd

# Sciezka do roota projektu (dla importow i sciezek do danych)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import tensorflow as tf
from cnn_builder import build_model
from class_weights import load_class_weights, save_class_weights


# ---------------------------------------------------------------------------
# Sciezki
# ---------------------------------------------------------------------------
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
TRAIN_RAW = os.path.join(RAW_PATH, "Train")
TEST_RAW = os.path.join(RAW_PATH, "Test")
TEST_CSV = os.path.join(RAW_PATH, "Test.csv")

#TRAIN_BALANCED = os.path.join(PROJECT_ROOT, "data", "train_balanced")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "config", "class_weights.json")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# Konfiguracja GPU / DirectML
# ---------------------------------------------------------------------------

def setup_device():
    """Wykrywa i konfiguruje dostepne urzadzenie (GPU/DirectML/CPU)."""
    print("\n--- Konfiguracja urzadzenia ---")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Znaleziono GPU: {[g.name for g in gpus]}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return "GPU"

    print("Brak GPU — trening na CPU.")
    print("Tip: 'pip install tensorflow-directml-plugin' dla AMD Radeon")
    return "CPU"


# ---------------------------------------------------------------------------
# Przygotowanie danych — prawidlowy podzial Train / Test
# ---------------------------------------------------------------------------

def prepare_test_set():
    """
    Organizuje data/raw/Test w strukture folderow klas na podstawie Test.csv.
    Wynik: data/test/{ClassId}/{obrazek}.png — gotowe do image_dataset_from_directory.
    """
    if os.path.exists(TEST_DIR):
        num_classes = len([d for d in os.listdir(TEST_DIR)
                          if os.path.isdir(os.path.join(TEST_DIR, d))])
        if num_classes > 0:
            return

    print("Przygotowuje held-out test set z data/raw/Test + Test.csv...")
    os.makedirs(TEST_DIR, exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    copied = 0
    for _, row in df.iterrows():
        class_id = str(row["ClassId"])
        img_src = os.path.join(RAW_PATH, row["Path"])

        class_dir = os.path.join(TEST_DIR, class_id)
        os.makedirs(class_dir, exist_ok=True)

        img_name = os.path.basename(img_src)
        shutil.copy2(img_src, os.path.join(class_dir, img_name))
        copied += 1

    print(f"Test set: {copied} obrazow w {TEST_DIR}")

"""
def prepare_train_balanced():
    
    Balansuje TYLKO dane treningowe (data/raw/Train) przez oversampling.
    Wynik: data/train_balanced/{ClassId}/{obrazki}
    Test set NIE jest mieszany z danymi treningowymi.
    
    if os.path.exists(TRAIN_BALANCED):
        return

    print("Balansuje dane treningowe (tylko Train, bez Test)...")
    from data_processor import balance_dataset
    balance_dataset(data_dir=TRAIN_RAW, target_dir=TRAIN_BALANCED)
"""

def ensure_data_ready():
    """Sprawdza czy dane sa przygotowane, jesli nie — przygotowuje."""
    if not os.path.exists(TRAIN_RAW):
        print(f"BLAD: Brak danych w {TRAIN_RAW}")
        print("Pobierz dataset GTSRB z Kaggle i rozpakuj do data/raw/")
        return False

    if not os.path.exists(TEST_CSV):
        print(f"BLAD: Brak pliku {TEST_CSV}")
        return False

    
    prepare_test_set()
    return True


# ---------------------------------------------------------------------------
# Ladowanie danych
# ---------------------------------------------------------------------------

def load_train_val(data_dir, img_size, batch_size, seed):
    """Laduje zbiory train/val z katalogu (80/20 split)."""
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=None,
        label_mode="int",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
      # --- 3. Oversampling TYLKO na train ---
    print("\n--- Oversampling zbioru treningowego ---")
    from data_processor import apply_oversampling
    train_ds = apply_oversampling(train_ds_raw, num_classes=43, seed=seed)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    num_train = train_ds.cardinality().numpy() * batch_size
    num_val = val_ds.cardinality().numpy() * batch_size
    print(f"Train: ~{num_train} obrazow | Val: ~{num_val} obrazow")

    return train_ds, val_ds


def load_test(data_dir, img_size, batch_size):
    """Laduje held-out test set."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    return test_ds


# ---------------------------------------------------------------------------
# Ewaluacja
# ---------------------------------------------------------------------------

def evaluate_model(model_path, img_size, batch_size, log_dir=None):
    """
    Ewaluacja modelu na held-out test set (data/test/).
    Wyswietla overall accuracy i per-class accuracy.
    Zapisuje wyniki do JSON.
    """
    print(f"\n{'='*60}")
    print("EWALUACJA NA HELD-OUT TEST SET")
    print(f"{'='*60}")
    print(f"Model: {model_path}")

    model = tf.keras.models.load_model(model_path)
    test_ds = load_test(TEST_DIR, img_size, batch_size)

    # Overall metrics
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class accuracy
    class_names = sorted(os.listdir(TEST_DIR))

    all_labels = []
    all_preds = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy())
        all_preds.extend(np.argmax(preds, axis=1))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print(f"\nPer-class accuracy (najslabsze 10 klas):")
    print(f"{'Klasa':<10} {'Nazwa':>8} {'Acc':>8} {'Probek':>8}")
    print("-" * 38)

    per_class = {}
    for idx, name in enumerate(class_names):
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == idx).mean()
        per_class[name] = {"accuracy": round(float(cls_acc), 4), "samples": int(mask.sum())}

    # Sortuj od najslabszych
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["accuracy"])
    for name, info in sorted_classes[:10]:
        print(f"  {name:<10} {info['accuracy']:>7.2%} {info['samples']:>8}")

    print(f"\nSrednia per-class accuracy: "
          f"{np.mean([v['accuracy'] for v in per_class.values()]):.4f}")

    # Zapis wynikow
    results = {
        "model_path": model_path,
        "test_dir": TEST_DIR,
        "test_loss": round(loss, 4),
        "test_accuracy": round(accuracy, 4),
        "total_test_samples": len(all_labels),
        "correct_predictions": int((all_labels == all_preds).sum()),
        "per_class": per_class,
    }

    out_dir = log_dir or MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nWyniki zapisane do: {results_path}")
    return results


# ---------------------------------------------------------------------------
# Trening
# ---------------------------------------------------------------------------

def run_training(args):
    """Glowna funkcja treningowa."""

    device = setup_device()

    # -- 1. Dane --
    print("\n--- 1. Przygotowanie danych ---")
    if not ensure_data_ready():
        sys.exit(1)

    print(f"Trening:    {TRAIN_RAW} ( Train)")
    print(f"Test:       {TEST_DIR} (held-out, nie widziany przez model)")

    # -- 2. Wagi klas --
    print("\n--- 2. Wagi klas ---")
    if not os.path.exists(WEIGHTS_PATH):
        print("Generuje wagi klas z data/raw/Train (oryginalny rozklad)...")
        save_class_weights(TRAIN_RAW, WEIGHTS_PATH)
    else:
        print(f"Wczytuje wagi z {WEIGHTS_PATH}")

    class_weights = load_class_weights(WEIGHTS_PATH)
    print(f"Zaladowano wagi dla {len(class_weights)} klas")
    print("-> class_weight pominiete (dane zbalansowane przez oversampling)")

    # -- 3. Ladowanie danych --
    print("\n--- 3. Ladowanie danych treningowych ---")
    train_ds, val_ds = load_train_val(
        TRAIN_RAW, args.img_size, args.batch_size, args.seed
    )

    # -- 4. Budowa modelu --
    aug_label = "ON" if args.augmentation else "OFF"
    print(f"\n--- 4. Budowa modelu (augmentacja: {aug_label}) ---")

    model = build_model(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=43,
        use_augmentation=args.augmentation,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # -- 5. Callbacks i logowanie --
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_aug{aug_label}"

    log_dir = os.path.join(PROJECT_ROOT, "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_model_path = os.path.join(MODEL_DIR, f"best_{run_name}.keras")

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, "training_log.csv"),
        ),
    ]

    # -- 6. Trening --
    print(f"\n--- 5. Trening ({args.epochs} epok, patience={args.patience}) ---")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # -- 7. Zapis modelu koncowego --
    final_model_path = os.path.join(MODEL_DIR, f"final_{run_name}.keras")
    model.save(final_model_path)

    # -- 8. Podsumowanie treningu --
    best_val_acc = max(history.history["val_accuracy"])
    final_val_acc = history.history["val_accuracy"][-1]
    epochs_trained = len(history.history["loss"])

    run_config = {
        "run_name": run_name,
        "device": device,
        "augmentation": args.augmentation,
        "epochs_requested": args.epochs,
        "epochs_trained": epochs_trained,
        "best_val_accuracy": round(best_val_acc, 4),
        "final_val_accuracy": round(final_val_acc, 4),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "img_size": args.img_size,
        "seed": args.seed,
        "patience": args.patience,
        "data_source":"data/raw/Train (oversampling w pamięci)",
        "model_path": final_model_path,
        "best_model_path": best_model_path,
    }

    config_path = os.path.join(log_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("TRENING ZAKONCZONY")
    print("=" * 60)
    print(f"  Epoki:              {epochs_trained}/{args.epochs}")
    print(f"  Najlepsza val_acc:  {best_val_acc:.4f}")
    print(f"  Koncowa val_acc:    {final_val_acc:.4f}")
    print(f"  Najlepszy model:    {best_model_path}")
    print(f"  Logi:               {log_dir}")
    print("=" * 60)

    # -- 9. Ewaluacja na held-out test set --
    print("\n--- 6. Ewaluacja na held-out test set ---")
    evaluate_model(best_model_path, args.img_size, args.batch_size, log_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline treningowy GTSRB — znaki drogowe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Augmentacja
    parser.add_argument(
        "--augmentation", action="store_true", default=True,
        help="Uzyj warstw augmentacji w modelu",
    )
    parser.add_argument(
        "--no-augmentation", dest="augmentation", action="store_false",
        help="Wylacz augmentacje",
    )

    # Ewaluacja
    parser.add_argument(
        "--eval-only", type=str, default=None, metavar="MODEL_PATH",
        help="Tylko ewaluacja (bez treningu) — podaj sciezke do .keras",
    )

    # Hiperparametry
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=32, help="Rozmiar obrazu (px)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=7, help="EarlyStopping patience")

    args = parser.parse_args()

    if args.eval_only:
        setup_device()
        if not os.path.exists(TEST_DIR):
            ensure_data_ready()
        evaluate_model(args.eval_only, args.img_size, args.batch_size)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
