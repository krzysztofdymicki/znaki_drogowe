# Raport z Realizacji Zadań 1 i 2 - Data Processor

Niniejszy dokument opisuje zmiany wprowadzone w module `src/data_processor.py`, dotyczące automatyzacji pobierania danych, ich konsolidacji oraz balansowania.

## 1. Automatyzacja Pobierania Danych
Zaimplementowano funkcję `download_dataset()`, która zapewnia ciągłość pracy bez względu na stan lokalnego środowiska.

*   **Co robi:** Sprawdza, czy dane już istnieją w `data/raw`. Jeśli nie, próbuje wykorzystać API Kaggle do automatycznego pobrania zbioru GTSRB.
*   **Dlaczego tak:** W przypadku braku skonfigurowanego klucza API (`kaggle.json`), skrypt nie wyrzuca błędu "w próżnię", lecz wyświetla **kompletną instrukcję krok po kroku** dla użytkownika, wyjaśniając jak uzyskać klucz i gdzie go umieścić.

## 2. Zadanie 1: Konsolidacja Zbiorów (Merge)
Funkcja `merge_train_test()` odpowiada za zebranie rozproszonych danych w jeden spójny zbiór.

*   **Co robi:** Łączy obrazy z folderu `Train` (już podzielonego na klasy) z obrazami z folderu `Test`. W przypadku zbioru testowego, wykorzystuje plik `Test.csv`, aby przypisać każdy obraz do właściwego folderu klasy (`ClassId`).
*   **Dlaczego tak:** Fabryczny podział 80/20 jest często nieoptymalny dla specyficznych celów edukacyjnych lub walidacji krzyżowej. Połączenie danych w jeden folder `data/combined` pozwala na:
    *   Całkowitą kontrolę nad procesem losowania nowych podziałów.
    *   Łatwiejsze przeprowadzenie balansowania całej populacji znaków.

## 3. Zadanie 2: Wyrównywanie Reprezentacji (Balancing)
Funkcja `balance_dataset()` rozwiązuje problem nadreprezentacji niektórych znaków drogowych (np. "Ograniczenie do 50 km/h" vs rzadsze znaki).

*   **Co robi:** Wykorzystuje metodę **Oversampling**. Wyznacza najliczniejszą klasę w zbiorze, a następnie w rzadszych klasach losuje istniejące obrazy i tworzy ich duplikaty (z prefiksem `bal_`), aż do wyrównania poziomu.
*   **Dlaczego tak:**
    *   **Uniknięcie Biasu:** Model trenowany na niezbalansowanych danych "nauczy się", że częste znaki są bezpieczniejszym wyborem przy niepewności, co drastycznie obniża skuteczność na rzadkich znakach.
    *   **Bezpieczeństwo danych:** Operacja odbywa się w nowym folderze `data/balanced`, dzięki czemu dane źródłowe pozostają nienaruszone.
    *   **Prostota i skuteczność:** Oversampling jest bezpieczniejszy od Undersamplingu, ponieważ nie tracimy cennych informacji o rzadkich klasach poprzez ich usuwanie.

## Integracja
Wszystkie funkcje są odseparowane od głównego `data_loader.py`, co pozwala na ich niezależne wywołanie jako proces "pre-processingowy" przed właściwym trenowaniem modelu.
