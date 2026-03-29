# Raport zmian: Pipeline treningowy

## Co zostalo zrobione

Stworzone zostaly 4 nowe pliki, ktore lacza i rozszerzaja funkcjonalnosc wczesniejszych skryptow.
Zadne istniejace pliki nie zostaly zmodyfikowane ani usuniete.

### Nowe pliki

- **`src/pipeline.py`** — glowny skrypt do uruchamiania treningu i ewaluacji modelu.
  Obsluguje caly proces: przygotowanie danych, budowe modelu, trening z logowaniem wynikow,
  oraz testowanie na oddzielnym zbiorze testowym.

- **`src/cnn_builder.py`** — budowa sieci neuronowej z mozliwoscia wlaczenia lub wylaczenia
  augmentacji (losowe obroty, przyblizenia, zmiany jasnosci i kontrastu).

- **`src/class_weights.py`** — obliczanie i zapis wag klas do pliku JSON, dzieki czemu
  model moze lepiej radzic sobie z klasami, ktore maja mniej probek. (To już było, ale teraz jest dostosowane do nowego pipeline)

- **`src/gui.py`** — interfejs graficzny (otwiera sie w przegladarce). Trzy zakladki:
  - *Trening* — ustawienie parametrow (epoki, learning rate, batch size, augmentacja on/off)
    i uruchomienie treningu z podgladem postepu na zywo (tekst + wykresy accuracy/loss).
  - *Klasyfikacja* — wybor modelu z listy, wrzucenie wlasnego zdjecia znaku lub wylosowanie
    z testu. Pokazuje top-5 predykcji z paskami pewnosci i prawdziwa klase.
  - *Przeglad testu* — ewaluacja modelu na calym zbiorze testowym. Wyswietla dokladnosc
    per klasa i galerie blednie sklasyfikowanych obrazkow z opisem (co model powiedzial
    vs jaka byla prawda).

  Uruchomienie: `python src/gui.py` (otwiera sie na http://127.0.0.1:7860)

---

## Co sie zmienilo i dlaczego

### Rozdzielenie danych treningowych i testowych

**Wczesniej:** dane z folderu Train i Test byly laczone w jeden zbior, a potem dzielone losowo
na trening i walidacje. Przez to obrazy testowe mogly trafic do treningu — model "widzial"
odpowiedzi przed egzaminem.

**Teraz:** Train jest balansowany i uzywany do nauki, a Test jest trzymany osobno i uzywany
tylko do koncowej oceny. Dzieki temu wyniki sa rzetelne.

### Preprocessing i augmentacja wbudowane w model

**Teraz:** normalizacja i augmentacja sa warstwami wewnatrz modelu. Augmentacja wlacza sie
automatycznie tylko podczas treningu, a przy testowaniu jest nieaktywna. Mozna ja tez
calkowicie wylaczyc flaga `--no-augmentation` jesli puszczamy z command line.

### Pelny pipeline zamiast szkieletow

**Wczesniej:** `train.py` i `model.py` zawieraly tylko szkielety z komentarzami TODO.

**Teraz:** pipeline dziala od poczatku do konca — wystarczy jedno polecenie:
```
python src/pipeline.py
python src/pipeline.py --no-augmentation
python src/pipeline.py --eval-only models/best_model.keras
```

---

## Status plikow

### Aktualne (uzywane):
| Plik | Rola |
|------|------|
| `src/pipeline.py` | Trening i ewaluacja (command line) |
| `src/cnn_builder.py` | Budowa modelu |
| `src/class_weights.py` | Wagi klas |
| `src/gui.py` | Interfejs graficzny (przeglądarka) |
| `src/data_processor.py` | Balansowanie danych (funkcja `balance_dataset()` wywolywana przez pipeline) |

### Zastapione (zostaja do podgladu):
| Plik | Zastapiony przez |
|------|-----------------|
| `src/train.py` | `pipeline.py` |
| `src/model.py` | `cnn_builder.py` |
| `src/augmentation.py` | warstwy augmentacji w `cnn_builder.py` |
| `src/data_loader.py` | `pipeline.py` |
| `src/preprocessing.py` | `pipeline.py` + `cnn_builder.py` |
| `docs/propozycja_loadera.py` | `pipeline.py` |

---

## Wyniki treningow

### Wariant 1 — stare podejscie (Train+Test zmieszane)
| Augmentacja | Dokladnosc walidacji | Epoki |
|-------------|---------------------|-------|
| Wlaczona    | 99.91%              | 20/30 |
| Wylaczona   | 99.93%              | 23/30 |

*Wynik zawyony — dane testowe wyciekaly do treningu.*

### Wariant 2 — nowe podejscie (osobny zbior testowy)
| Augmentacja | Dokladnosc walidacji | Dokladnosc na tescie | Epoki |
|-------------|---------------------|---------------------|-------|
| Wlaczona    | 99.81%              | 97.13%              | 21/30 |
| Wylaczona   | 99.94%              | 98.29%              | 30/30 |

*Wynik rzetelny — zbior testowy nigdy nie widziany przez model.*
