import cv2
import numpy as np

def initial_preprocess(image, size=(32, 32)):
    """
    KROK 1: Preprocessing wstępny (Krzysztof)
    Zadanie: Ujednolicenie formatu wszystkich zdjęć.
    
    1. Zmiana rozmiaru do 32x32.
    2. Opcjonalnie: zmiana na skalę szarości.
    """
    # 1. Resize
    img = cv2.resize(image, size)
    
    # 2. Opcjonalnie: grayscale (odkomentuj jeśli model ma być czarno-biały)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def final_preprocess(image):
    """
    KROK 3: Preprocessing finalny (Krzysztof)
    Zadanie: Przygotowanie danych bezpośrednio pod sieć neuronową.
    
    1. Normalizacja (0.0 - 1.0).
    """
    # Sprowadzenie wartości pikseli do zakresu [0, 1]
    img_normalized = image.astype('float32') / 255.0
    
    return img_normalized

# Przykład użycia:
# 1. raw_img = load(...)
# 2. img = initial_preprocess(raw_img)
# 3. img = apply_augmentation(img)  <-- Tu wchodzi działka Lubomira
# 4. ready_img = final_preprocess(img)
