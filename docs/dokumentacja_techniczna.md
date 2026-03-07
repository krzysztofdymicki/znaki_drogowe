# Dokumentacja techniczna - System wizyjny klasyfikacji znaków (GTSRB)

## 1. Wprowadzenie
Celem projektu jest utworzenie konwolucyjnej sieci neuronowej (CNN) do automatycznego rozpoznawania znaków drogowych na podstawie zbioru kilkudziesięciu tysięcy obrazów GTSRB (German Traffic Sign Recognition Benchmark).

## 2. Architektura systemu
> [Tutaj Krzysztof opiszesz jak wygląda cały workflow, począwszy od pobrania danych, po preprocessing obrazków, budowę archtetkury modelu aż do treningu.]

## 3. Preprocessing
Proces przygotowania pobranych danych został podzielony na trzy etapy:

### 3.1. Preprocessing Wstępny (Initial Preprocessing)
Odpowiada za ujednolicenie danych wejściowych.
- **Skalowanie:** Standaryzacja wszystkich zdjęć do rozmiaru 32x32 pikseli przy użyciu `cv2.resize`.
- **Konwersja kolorów:** Opcjonalnie przygotowano skrypt do konwersji na skalę szarości.

### 3.2. Augmentacja (Augmentation)
*Implementacja w toku - Lubomir Kwiatkowski*

### 3.3. Preprocessing Finalny (Final Preprocessing)
Przygotowanie danych bezpośrednio dla warstw wejściowych sieci neuronowej.
- **Normalizacja:** Przeskalowanie wartości pikseli z zakresu [0, 255] do [0.0, 1.0] (typ `float32`).

## 4. Wyświetlanie wyników
*(Sekcja zostawiona dla Kacpra i Adama z podzespołu 2)*
