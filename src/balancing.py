# balancing.py
"""
Balansowanie klas dla TensorFlow/Keras.
Kompatybilne z data_loader.py

UWAGA: Jeśli używasz data/balanced (już zbalansowane przez kopiowanie),
       możesz pominąć class_weights lub użyć lekkich wag.
       
       Jeśli używasz data/combined (oryginalne proporcje),
       class_weights są KLUCZOWE!
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple


def get_labels_from_dataset(dataset) -> np.ndarray:
    """
    Wyciąga etykiety z tf.data.Dataset.
    
    Args:
        dataset: tf.data.Dataset z (images, labels)
    
    Returns:
        numpy array z etykietami
    """
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy())
    return np.array(labels)


def compute_class_weights(
    labels: np.ndarray,
    smoothing: float = 1.0
) -> Dict[int, float]:
    """
    Oblicza wagi klas.
    
    Args:
        labels: Tablica etykiet
        smoothing: Wygładzanie wag
                   1.0 = pełne wagi (dla niezbalansowanych danych)
                   0.5 = sqrt wag (łagodniejsze)
                   0.0 = brak wag (wszystkie = 1.0)
    
    Returns:
        Słownik {class_id: weight}
    """
    classes = np.unique(labels)
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    # Wygładzanie
    if smoothing != 1.0:
        weights = np.power(weights, smoothing)
    
    return dict(zip(classes.astype(int), weights))


def get_class_weights(
    dataset,
    smoothing: float = 1.0,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Oblicza wagi klas z tf.data.Dataset.
    
    Użyj w model.fit(class_weight=...).
    
    Args:
        dataset: tf.data.Dataset z (images, labels)
        smoothing: Wygładzanie wag (1.0 = pełne, 0.5 = łagodne)
        verbose: Czy wyświetlać informacje
    
    Returns:
        Słownik {class_id: weight}
    
    Example:
        >>> train_ds, val_ds = get_loaders()
        >>> class_weights = get_class_weights(train_ds)
        >>> model.fit(train_ds, class_weight=class_weights, epochs=50)
    """
    labels = get_labels_from_dataset(dataset)
    weights = compute_class_weights(labels, smoothing)
    
    if verbose:
        print(f"Obliczono wagi dla {len(weights)} klas (smoothing={smoothing}):")
        print(f"  Min: {min(weights.values()):.4f}")
        print(f"  Max: {max(weights.values()):.4f}")
        print(f"  Stosunek max/min: {max(weights.values())/min(weights.values()):.2f}x")
    
    return weights


def print_class_distribution(dataset, title: str = "ROZKŁAD KLAS"):
    """
    Wyświetla rozkład klas w datasecie.
    
    Args:
        dataset: tf.data.Dataset z (images, labels)
        title: Tytuł wyświetlany
    """
    labels = get_labels_from_dataset(dataset)
    
    unique, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    min_count = counts.min()
    total = len(labels)
    
    print(f"\n{'='*55}")
    print(f"{title:^55}")
    print(f"{'='*55}")
    
    for class_id, count in zip(unique, counts):
        bar_len = int(35 * count / max_count)
        bar = '█' * bar_len
        pct = 100 * count / total
        print(f"Klasa {class_id:2d}: {count:5d} ({pct:4.1f}%) {bar}")
    
    print(f"{'='*55}")
    print(f"Razem: {total} | Klasy: {len(unique)}")
    print(f"Min: {min_count} | Max: {max_count} | Stosunek: 1:{max_count/min_count:.1f}")
    print(f"{'='*55}\n")


def analyze_balance(dataset) -> Tuple[bool, float]:
    """
    Sprawdza czy dataset jest zbalansowany.
    
    Args:
        dataset: tf.data.Dataset
    
    Returns:
        Tuple (is_balanced, imbalance_ratio)
        is_balanced: True jeśli stosunek max/min < 2
        imbalance_ratio: stosunek max/min
    """
    labels = get_labels_from_dataset(dataset)
    _, counts = np.unique(labels, return_counts=True)
    
    ratio = counts.max() / counts.min()
    is_balanced = ratio < 2.0
    
    return is_balanced, ratio


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import os
    import sys
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from data_loader import get_loaders
    
    if os.path.exists("data/balanced"):
        print("Test modułu balansowania...\n")
        
        train_ds, val_ds = get_loaders()
        
        # Analiza
        print("1. Rozkład klas w zbiorze treningowym:")
        print_class_distribution(train_ds, "TRAIN")
        
        print("2. Rozkład klas w zbiorze walidacyjnym:")
        print_class_distribution(val_ds, "VALIDATION")
        
        # Sprawdzenie balansu
        is_balanced, ratio = analyze_balance(train_ds)
        print(f"3. Czy zbalansowane: {'TAK' if is_balanced else 'NIE'} (ratio: {ratio:.2f})")
        
        # Wagi
        print("\n4. Wagi klas:")
        if is_balanced:
            print("   Dataset już zbalansowany - wagi opcjonalne")
            weights = get_class_weights(train_ds, smoothing=0.5)
        else:
            print("   Dataset niezbalansowany - wagi REKOMENDOWANE")
            weights = get_class_weights(train_ds, smoothing=1.0)
        
        print("\n✅ Test zakończony!")
    else:
        print("BŁĄD: Folder data/balanced nie istnieje!")