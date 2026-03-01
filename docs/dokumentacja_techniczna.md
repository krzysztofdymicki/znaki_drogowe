# Dokumentacja techniczna - System wizyjny klasyfikacji znaków (GTSRB)

## 1. Wprowadzenie
Celem projektu jest utworzenie konwolucyjnej sieci neuronowej (CNN) do automatycznego rozpoznawania znaków drogowych na podstawie zbioru kilkudziesięciu tysięcy obrazów GTSRB (German Traffic Sign Recognition Benchmark).

## 2. Architektura systemu
> [Tutaj Krzysztof opiszesz jak wygląda cały workflow, począwszy od pobrania danych, po preprocessing obrazków, budowę archtetkury modelu aż do treningu.]

## 3. Preprocessing
Obrazy poddawane są następującym operacjom przed wejściem do sieci:
- **Skalowanie:** 32x32 piksele.
- **Normalizacja:** [0, 1].
- **Augmentacja:** Rotacje, zmiany jasności (i ew. zoom).
- **Konwersja:** (Zostawiona jako opcja, np. skala szarości lub praca na kanałach RGB).

## 4. Wyświetlanie wyników
*(Sekcja zostawiona dla Kacpra i Adama z podzespołu 2)*
