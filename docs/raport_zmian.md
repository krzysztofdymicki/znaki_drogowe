
### Postanowiłem wykonać oversamlning w RAM ponieważ wymagało to mniej zmian niż podział na zbiory treningowy i validacyjny lokalnie na dysku i zajmuje mniej przestrzeni dyskowej. 

# cnn_builder 
Dodano do warstw modelu parametr  padding="same". Pozwala to na nietracenie informacji z skrajnych pikseli obrazu. (opcjonalne)
# data_processor
Dodano nowa funkcję **apply_oversampling** dokonującą oversamplingu na zbiorze po rozdzieleniu na zbiory trening i validację.
Oversampling dokonywany jest w RAM. 
# gui oraz pipeline 
Przystosowane obie funkcje do kożystania z **apply_oversampling**, oraz zostawiono zakomentowane rozwiązanie z danymi balanced w razie zmiany koncepcji.


# uczenie modelu powinno być realizowane z augmentacja ponieważ bez niej wysokie wyniki wynikają z przeuczenia 
