"""
GUI do trenowania i testowania modelu rozpoznawania znakow drogowych.

Uruchomienie:
    python src/gui.py

Otwiera interfejs w przegladarce z trzema zakladkami:
  1. Trening    — ustawienie parametrow i uruchomienie treningu z podgladem na zywo
  2. Klasyfikacja — wrzuc zdjecie znaku i sprawdz co model powie
  3. Przeglad    — ewaluacja na zbiorze testowym, podglad blednie sklasyfikowanych
"""

import os
import sys
import random
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import tensorflow as tf
from cnn_builder import build_model

# ---------------------------------------------------------------------------
# Sciezki
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
TRAIN_BALANCED = os.path.join(PROJECT_ROOT, "data", "train_balanced")

# ---------------------------------------------------------------------------
# Nazwy znakow GTSRB (niemieckie znaki drogowe)
# ---------------------------------------------------------------------------
SIGN_NAMES = {
    0: "Ograniczenie 20 km/h",
    1: "Ograniczenie 30 km/h",
    2: "Ograniczenie 50 km/h",
    3: "Ograniczenie 60 km/h",
    4: "Ograniczenie 70 km/h",
    5: "Ograniczenie 80 km/h",
    6: "Koniec ogr. 80 km/h",
    7: "Ograniczenie 100 km/h",
    8: "Ograniczenie 120 km/h",
    9: "Zakaz wyprzedzania",
    10: "Zakaz wyprz. pow. 3.5t",
    11: "Pierwszenstwo na skrzyzowaniu",
    12: "Droga z pierwszenstwem",
    13: "Ustap pierwszenstwa",
    14: "Stop",
    15: "Zakaz ruchu",
    16: "Zakaz wjazdu pow. 3.5t",
    17: "Zakaz wjazdu",
    18: "Uwaga niebezpieczenstwo",
    19: "Niebezpieczny zakret lewy",
    20: "Niebezpieczny zakret prawy",
    21: "Podwojny zakret",
    22: "Nierowna droga",
    23: "Sliska nawierzchnia",
    24: "Zwezenie prawe",
    25: "Roboty drogowe",
    26: "Sygnalizacja swietlna",
    27: "Przejscie dla pieszych",
    28: "Uwaga dzieci",
    29: "Uwaga rowerzysci",
    30: "Uwaga oblodzenie",
    31: "Dzikie zwierzeta",
    32: "Koniec ograniczen",
    33: "Nakaz jazdy w prawo",
    34: "Nakaz jazdy w lewo",
    35: "Nakaz jazdy prosto",
    36: "Prosto lub prawo",
    37: "Prosto lub lewo",
    38: "Trzymaj sie prawej",
    39: "Trzymaj sie lewej",
    40: "Rondo",
    41: "Koniec zakazu wyprzedzania",
    42: "Koniec zakazu wyprz. 3.5t",
}


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def get_class_names():
    """Zwraca posortowana liste nazw folderow klas (jak robi to TF)."""
    if not os.path.exists(TEST_DIR):
        return [str(i) for i in range(43)]
    return sorted([d for d in os.listdir(TEST_DIR)
                    if os.path.isdir(os.path.join(TEST_DIR, d))])


def sign_name(class_folder):
    """Zwraca czytelna nazwe znaku na podstawie nazwy folderu."""
    try:
        return SIGN_NAMES.get(int(class_folder), f"Klasa {class_folder}")
    except (ValueError, TypeError):
        return f"Klasa {class_folder}"


def list_models():
    """Lista plikow .keras w folderze models/."""
    if not os.path.exists(MODELS_DIR):
        return []
    return sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")])


def make_history_plot(history):
    """Tworzy wykres accuracy i loss z historii treningu."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(history["accuracy"], label="trening", marker=".")
    ax1.plot(history["val_accuracy"], label="walidacja", marker=".")
    ax1.set_title("Dokladnosc")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["loss"], label="trening", marker=".")
    ax2.plot(history["val_loss"], label="walidacja", marker=".")
    ax2.set_title("Strata (loss)")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Zakladka 1: Trening
# ---------------------------------------------------------------------------

def run_training(epochs, lr, batch_size, use_augmentation, patience):
    """Generator — trenuje epoka po epoce i zwraca postep na zywo."""

    if not os.path.exists(TRAIN_BALANCED):
        yield "BLAD: Brak danych treningowych. Uruchom najpierw:\n  python src/pipeline.py", None
        return

    yield "Ladowanie danych...", None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_BALANCED, validation_split=0.2, subset="training",
        seed=42, image_size=(32, 32), batch_size=int(batch_size), label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_BALANCED, validation_split=0.2, subset="validation",
        seed=42, image_size=(32, 32), batch_size=int(batch_size), label_mode="int",
    )
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    yield "Budowa modelu...", None

    model = build_model(use_augmentation=use_augmentation)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Przygotowanie zapisu
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    aug_label = "ON" if use_augmentation else "OFF"
    run_name = f"run_{timestamp}_aug{aug_label}"
    best_path = os.path.join(MODELS_DIR, f"best_{run_name}.keras")
    os.makedirs(MODELS_DIR, exist_ok=True)

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val_acc = 0
    no_improve = 0

    for epoch in range(int(epochs)):
        h = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)

        for key in history:
            history[key].append(h.history[key][0])

        val_acc = history["val_accuracy"][-1]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(best_path)
            no_improve = 0
        else:
            no_improve += 1

        # Tekst postepu
        lines = [f"Epoka {i+1:2d}:  acc={history['accuracy'][i]:.4f}"
                 f"  val_acc={history['val_accuracy'][i]:.4f}"
                 for i in range(len(history["accuracy"]))]
        status = "\n".join(lines)
        status += f"\n\nNajlepsza val_acc: {best_val_acc:.4f}"

        if no_improve >= int(patience):
            status += f"\nEarly stopping po {epoch+1} epokach."
            yield status, make_history_plot(history)
            return

        yield status, make_history_plot(history)

    # Zapis koncowy
    final_path = os.path.join(MODELS_DIR, f"final_{run_name}.keras")
    model.save(final_path)
    status += f"\n\nModel zapisany: {os.path.basename(final_path)}"
    yield status, make_history_plot(history)


# ---------------------------------------------------------------------------
# Zakladka 2: Klasyfikacja
# ---------------------------------------------------------------------------

def classify_image(model_name, image):
    """Klasyfikuje wrzucony obrazek — zwraca top-5 predykcji."""
    if image is None:
        return "Wrzuc zdjecie znaku drogowego."
    if not model_name:
        return "Wybierz model z listy."

    model_path = os.path.join(MODELS_DIR, model_name)
    model = tf.keras.models.load_model(model_path)
    class_names = get_class_names()

    img = tf.image.resize(image, (32, 32))
    img = tf.expand_dims(img, 0)
    preds = model.predict(img, verbose=0)[0]

    top5 = np.argsort(preds)[-5:][::-1]
    return {sign_name(class_names[i]): float(preds[i]) for i in top5}


def classify_random():
    """Losuje obrazek z test setu i zwraca go + prawdziwa klase."""
    if not os.path.exists(TEST_DIR):
        return None, "Brak danych testowych."

    class_names = get_class_names()
    cls = random.choice(class_names)
    cls_dir = os.path.join(TEST_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if not f.startswith(".")]
    img_name = random.choice(images)
    img_path = os.path.join(cls_dir, img_name)
    true_label = f"Prawdziwa klasa: {cls} — {sign_name(cls)}"

    return img_path, true_label


# ---------------------------------------------------------------------------
# Zakladka 3: Przeglad testu
# ---------------------------------------------------------------------------

def evaluate_on_test(model_name):
    """Ewaluacja modelu na test secie — zwraca statystyki i bledne obrazki."""
    if not model_name:
        return "Wybierz model.", []
    if not os.path.exists(TEST_DIR):
        return "Brak danych testowych.", []

    model_path = os.path.join(MODELS_DIR, model_name)
    model = tf.keras.models.load_model(model_path)
    class_names = get_class_names()

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=(32, 32), batch_size=32,
        label_mode="int", shuffle=False,
    )

    # Zbierz predykcje
    all_images, all_labels, all_preds = [], [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        all_images.append(images.numpy())
        all_labels.extend(labels.numpy())
        all_preds.extend(pred_classes)

    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    correct = (all_labels == all_preds).sum()
    total = len(all_labels)
    accuracy = correct / total

    # Per-class
    lines = [f"Dokladnosc ogolna: {accuracy:.2%} ({correct}/{total})\n"]
    lines.append(f"{'Klasa':<6} {'Nazwa':<30} {'Acc':>7} {'Probek':>7}")
    lines.append("-" * 55)

    per_class = []
    for idx, name in enumerate(class_names):
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == idx).mean()
        per_class.append((name, cls_acc, int(mask.sum())))

    per_class.sort(key=lambda x: x[1])
    for name, acc, count in per_class:
        lines.append(f"{name:<6} {sign_name(name):<30} {acc:>6.2%} {count:>7}")

    stats_text = "\n".join(lines)

    # Galeria blednych (maks 30)
    wrong_mask = all_labels != all_preds
    wrong_indices = np.where(wrong_mask)[0]
    if len(wrong_indices) > 30:
        wrong_indices = np.random.choice(wrong_indices, 30, replace=False)

    gallery = []
    for i in wrong_indices:
        img = all_images[i].astype("uint8")
        true_name = sign_name(class_names[all_labels[i]])
        pred_name = sign_name(class_names[all_preds[i]])
        caption = f"Prawda: {true_name} | Model: {pred_name}"
        gallery.append((img, caption))

    return stats_text, gallery


# ---------------------------------------------------------------------------
# Budowa interfejsu
# ---------------------------------------------------------------------------

def create_gui():
    available_models = list_models()
    default_model = available_models[0] if available_models else None

    with gr.Blocks(title="GTSRB — Znaki Drogowe") as app:
        gr.Markdown("# Rozpoznawanie znakow drogowych (GTSRB)")

        # === ZAKLADKA 1: TRENING ===
        with gr.Tab("Trening"):
            gr.Markdown("Ustaw parametry i uruchom trening. Postep widoczny na zywo.")

            with gr.Row():
                with gr.Column(scale=1):
                    epochs_sl = gr.Slider(1, 100, value=30, step=1, label="Epoki")
                    lr_sl = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning rate")
                    batch_sl = gr.Slider(16, 128, value=32, step=16, label="Batch size")
                    patience_sl = gr.Slider(1, 20, value=7, step=1, label="Patience (early stopping)")
                    aug_cb = gr.Checkbox(value=True, label="Augmentacja (obroty, zoom, jasnosc)")
                    train_btn = gr.Button("Rozpocznij trening", variant="primary")

                with gr.Column(scale=2):
                    train_log = gr.Textbox(label="Postep treningu", lines=15, interactive=False)
                    train_plot = gr.Plot(label="Wykresy")

            train_btn.click(
                fn=run_training,
                inputs=[epochs_sl, lr_sl, batch_sl, aug_cb, patience_sl],
                outputs=[train_log, train_plot],
            )

        # === ZAKLADKA 2: KLASYFIKACJA ===
        with gr.Tab("Klasyfikacja"):
            gr.Markdown("Wrzuc zdjecie znaku drogowego albo wylosuj z testu.")

            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=available_models, value=default_model,
                    label="Model", interactive=True,
                )
                refresh_btn = gr.Button("Odswiez liste modeli")

            refresh_btn.click(
                fn=list_models,
                outputs=model_dd,
            )

            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="Zdjecie znaku")
                    random_btn = gr.Button("Losowy obraz z testu")
                    true_label = gr.Textbox(label="Prawdziwa klasa", interactive=False)

                with gr.Column():
                    prediction = gr.Label(num_top_classes=5, label="Predykcja modelu")

            img_input.change(
                fn=classify_image,
                inputs=[model_dd, img_input],
                outputs=prediction,
            )

            random_btn.click(
                fn=classify_random,
                outputs=[img_input, true_label],
            )

        # === ZAKLADKA 3: PRZEGLAD TESTU ===
        with gr.Tab("Przeglad testu"):
            gr.Markdown("Ewaluacja modelu na calym zbiorze testowym.")

            with gr.Row():
                eval_model_dd = gr.Dropdown(
                    choices=available_models, value=default_model,
                    label="Model", interactive=True,
                )
                eval_btn = gr.Button("Ewaluuj", variant="primary")

            eval_stats = gr.Textbox(label="Wyniki", lines=20, interactive=False)
            eval_gallery = gr.Gallery(
                label="Blednie sklasyfikowane obrazy", columns=5, height="auto",
            )

            eval_btn.click(
                fn=evaluate_on_test,
                inputs=eval_model_dd,
                outputs=[eval_stats, eval_gallery],
            )

    return app


# ---------------------------------------------------------------------------
# Uruchomienie
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_gui()
    app.launch(theme=gr.themes.Soft())
