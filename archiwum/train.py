import os
from data_loader import load_images_and_labels
from model import build_transfer_learning_model

def main():
    """
    Lubomir / Krzysztof: Ten plik to sterownik całego kodu.
    Będzie korzystał z waszych funkcji, a finalnie na instancji
    modelu wywoływał .fit() żeby uczyć parametry na zbiorze danych.
    """
    print("--- START SYSTEMU WIZYJNEGO ---")
    
    # 1. Załaduj paczki uzywając data_loader.py (Krzysztof)
    print("1. Ładowanie danych z GTSRB...")
    
    # 2. Przetwórz uzywając preprocessing.py / augmentation.py (Krzysztof, Lubomir)
    print("2. Preprocessing / Augmentacja...")
    
    # 3. Zbuduj sieć uzywając model.py (Lubomir)
    print("3. Inicjalizacja modelu MobileNet...")

    # 4. Trening 
    print("4. Rozpoczęcie nauki (model.fit)...")

    # 5. Eksport wag np. model.save('models/model_gtsrb.h5')
    print("5. Zapisanie gotowego modelu.")

if __name__ == '__main__':
    main()
