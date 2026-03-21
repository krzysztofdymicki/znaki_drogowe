augmentation.py
Cel: Zapobieganie overfittingowi poprzez losowe przekształcanie obrazów podczas treningu.

Transformacje geometryczne:

Rotacja ±15°
Zoom 85–115%
Przesunięcie ±10%
Transformacje fotometryczne:

Jasność ±20%
Kontrast ±20%
Świadomie pominięte: flip poziomy/pionowy, duże zmiany koloru — zmieniają znaczenie znaku drogowego.

Główne funkcje:

get_augmentation_layer() — zwraca warstwę augmentacji
apply_augmentation(dataset) — aplikuje augmentację do datasetu

balancing w data_processor 
Zmniejszono oversampling do max/2 i dodano tworzenie pliku json wag w zależności od liczebności klasy.
Do wykożystania w poprawnym uczeniu modelu.