import os
# import kaggle # Jeśli używacie dedykowanego API

def download_dataset():
    """
    Krzysztof: Tutaj zaimplementuj ewentualne środowisko do 
    automatycznego pobierania danych (np wget, API Kaggle)
    lub samą instrukcję z wypisaniem ścieżek gdzie paczka na dysku.
    """
    print("TODO: Pobieranie datasetu..")

def load_images_and_labels(data_path):
    """
    Krzysztof: Ta funkcja powinna iterować przez foldery w zbiorze 
    i wczytywać zdjęcia (do tablic numpy, albo tf.data.Dataset) 
    z przypisanymi im numerami folderu (klasami od 0 do 42).
    """
    print("TODO: Wczytywanie zdjęć do pamięci.")
    pass

if __name__ == "__main__":
    download_dataset()
    # load_images_and_labels('data/raw/...')
