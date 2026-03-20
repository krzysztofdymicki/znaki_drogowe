augmentation.py — Augmentacja danych
Cel
Zapobieganie overfittingowi poprzez losowe przekształcanie obrazów podczas treningu. Model widzi w każdej epoce inne warianty tych samych zdjęć, dzięki czemu uczy się ogólnych cech znaków zamiast zapamiętywać konkretne piksele.

Zastosowane transformacje
Typ	      Transformacja	Zakres	Uzasadnienie
Geometryczne	Rotacja	±15°	Znaki bywają przechylone
Zoom	85–115%	Różna odległość od kamery
Przesunięcie	±10%	Znak nie zawsze w centrum
Fotometryczne	Jasność	±20%	Różne oświetlenie
Kontrast	±20%	Mgła, słońce, cień
Pominięte: flip poziomy/pionowy, duże zmiany koloru — zmieniają znaczenie znaku.

Główne funkcje
Python

get_augmentation_layer()      # warstwa do modelu
apply_augmentation(dataset)   # augmentacja datasetu (tylko train!)
balancing.py — Balansowanie klas
Cel
Wyrównanie ważności klas w zbiorze niezbalansowanym. GTSRB zawiera klasy liczące od ~200 do ~2250 próbek (stosunek 1:10). Bez balansowania model faworyzuje klasy liczne.

Mechanizm
Oblicza wagi klas odwrotnie proporcjonalne do ich liczebności. Wagi przekazywane do model.fit(class_weight=...) modyfikują funkcję straty — błędy na klasach rzadkich są bardziej karane.

Główne funkcje
Python

get_class_weights(dataset)        # zwraca {class_id: weight}
print_class_distribution(dataset) # wizualizacja rozkładu klas
Użycie w pipeline
Python

from augmentation import apply_augmentation
from balancing import get_class_weights

train_ds, val_ds = get_loaders()
train_ds = apply_augmentation(train_ds)     # augmentacja (tylko train)
class_weights = get_class_weights(train_ds) # wagi klas

model.fit(
    train_ds,
    validation_data=val_ds,
    class_weight=class_weights,
    epochs=50
)
Podsumowanie
Moduł       	Problem	            Rozwiązanie
augmentation.py	Overfitting	        Losowe transformacje obrazów
balancing.py	Nierównowaga klas	Wagi w funkcji straty