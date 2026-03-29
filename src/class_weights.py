"""
Modul do obliczania i zarzadzania wagami klas dla zbioru GTSRB.

Wagi klas kompensuja nierownowage w zbiorze danych - klasy z mniejsza
liczba probek dostaja wieksza wage w funkcji straty (loss function).

Uzycie:
    python src/class_weights.py                           # z data/combined
    python src/class_weights.py --data-dir data/balanced  # z innego zrodla
"""

import os
import json
from pathlib import Path


def compute_class_weights(data_dir):
    """
    Oblicza wagi klas na podstawie rozkladu probek w katalogu.

    Foldery sa sortowane alfabetycznie (tak jak robi to
    tf.keras.utils.image_dataset_from_directory), wiec indeksy
    labeli odpowiadaja kolejnosci alfabetycznej nazw folderow.

    Wzor: weight_i = total_samples / (num_classes * count_i)

    Returns:
        weights     - dict {label_index: weight}
        counts      - dict {label_index: count}
        label_map   - dict {label_index: folder_name}
    """
    data_dir = Path(data_dir)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        raise FileNotFoundError(f"Brak folderow klas w {data_dir}")

    counts = {}
    label_map = {}
    for idx, class_dir in enumerate(class_dirs):
        count = len([f for f in class_dir.iterdir() if f.is_file()])
        counts[idx] = count
        label_map[idx] = class_dir.name

    total = sum(counts.values())
    num_classes = len(counts)

    weights = {}
    for idx, count in counts.items():
        weights[idx] = round(total / (num_classes * count), 4) if count > 0 else 1.0

    return weights, counts, label_map


def save_class_weights(data_dir, output_path="config/class_weights.json"):
    """Oblicza wagi klas i zapisuje do pliku JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights, counts, label_map = compute_class_weights(data_dir)

    result = {
        "source_dir": str(data_dir),
        "num_classes": len(weights),
        "total_samples": sum(counts.values()),
        "class_weights": {str(k): v for k, v in weights.items()},
        "class_counts": {str(k): v for k, v in counts.items()},
        "label_to_classname": {str(k): v for k, v in label_map.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wagi klas zapisane do {output_path} ({len(weights)} klas, {sum(counts.values())} probek)")
    return weights


def load_class_weights(path="config/class_weights.json"):
    """
    Wczytuje wagi klas z pliku JSON.
    Zwraca dict {int: float} gotowy do uzycia w model.fit(class_weight=...).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {int(k): v for k, v in data["class_weights"].items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generacja wag klas GTSRB")
    parser.add_argument("--data-dir", default="data/combined",
                        help="Katalog z danymi (domyslnie: data/combined)")
    parser.add_argument("--output", default="config/class_weights.json",
                        help="Sciezka wyjsciowa JSON")
    args = parser.parse_args()

    save_class_weights(args.data_dir, args.output)
