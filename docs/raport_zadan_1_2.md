# Raport z Realizacji Zadań 1 i 2 - Data Processor

Niniejszy dokument opisuje zmiany wprowadzone w module `src/data_processor.py`, dotyczące automatyzacji pobierania danych, ich konsolidacji oraz balansowania.

## 1. Automatyzacja Pobierania Danych
Zaimplementowano funkcję `download_dataset()`, która zapewnia ciągłość pracy bez względu na stan lokalnego środowiska.

*   **Co robi:** Sprawdza, czy dane już istnieją w `data/raw`. Jeśli nie, próbuje wykorzystać API Kaggle do automatycznego pobrania zbioru GTSRB.
*   **Dlaczego tak:** W przypadku braku skonfigurowanego klucza API (`kaggle.json`), skrypt nie wyrzuca błędu "w próżnię", lecz wyświetla **kompletną instrukcję krok po kroku** dla użytkownika, wyjaśniając jak uzyskać klucz i gdzie go umieścić.




