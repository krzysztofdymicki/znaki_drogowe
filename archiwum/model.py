# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.applications import MobileNetV2

def build_custom_cnn(input_shape=(32, 32, 3), num_classes=43):
    """
    Lubomir: Konstrukcja autorskiej sieci CNN zbudowanej z warstw.
    """
    print("TODO: Budowa modelu Sequential (Convolutional).")
    return None


def build_transfer_learning_model(input_shape=(32, 32, 3), num_classes=43):
    """
    Lubomir: Budowa w oparciu o Transfer Learning (np. MobileNet).
    Upewnij się czy architektura przymie wymiary 32x32, czasem trzeba 
    rozszerzyć obrazy po załadowaniu do np. 96x96 pikseli.
    """
    print("TODO: Przygotowanie MobileNet z nową głową klasyfikującą.")
    return None
