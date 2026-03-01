# znaki_drogowe - Rozpoznawanie Znaków Drogowych

Projekt zespołu nr 1 dotyczący stworzenia modelu sieci CNN na danych z Kaggle. Zespół 1: `Krzysztof Dymicki` oraz `Lubomir Kwiatkowski`. Wyniki pracy trafią do zespołu 2.

## Struktura plików
- `data/` - Miejsce na zapisanie lokalnie bazy pobranej z kaggle (folder jest zignorowany przez gitar by go tam nie wrzucać).
- `docs/` - Pliki powiązane z dokumentacją dla wykładowców.
- `notebooks/` - Folder dla każdego z zespołu do eksperymentów poprzez Jupyter.
- `src/` - Główne środowisko kodu w python.

## Konfiguracja na swoim komputerze:

### 1. Sklonuj projekt z gita
```bash
git clone NAZWA_LINKU_DO_TWOJEGO_REPOZYTORIUM
```

### 2. Zainstaluj biblioteki pythona 
Pobiera i instaluje wszystkie rzeczy zawarte w pliku `requirements.txt` wymagane do działania (Tensorflow, opencv itp.)

```bash
pip install -r requirements.txt
```

### 3. Odpalenie testowe
```bash
python src/train.py
```
