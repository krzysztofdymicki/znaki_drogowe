import cv2
import numpy as np

def resize_image(image, size=(32, 32)):
    """
    Krzysztof: Skalowanie do wymiaru np. 32x32 pikseli.
    Użyj np. cv2.resize lub tf.image.resize
    """
    return cv2.resize(image, size)

def normalize_image(image):
    """
    Krzysztof: Przeskalowanie wartości pikseli z [0, 255] na [0.0, 1.0].
    """
    return image / 255.0

def convert_to_grayscale(image):
    """
    Krzysztof (opcjonalne): Konwersja do szaróści obrazów RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess_pipeline(image):
    """Główna rura dla obrazka, łącząca kroki"""
    img = resize_image(image)
    # img = convert_to_grayscale(img)
    img = normalize_image(img)
    return img
