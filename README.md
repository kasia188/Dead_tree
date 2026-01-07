Projekt służy do segmentacji martwych drzew na zdjęciach lotniczych  
przy użyciu informacji z kanałów RGB oraz NRG. Pipeline generuje maski,  
czyści je, łączy oraz porównuje z maskami referencyjnymi(ground truth), a następnie  
oblicza metryki jakości i tworzy wizualizacje.

Struktura projektu:  
dead\_tree/  
|-src/  
||-[main.py](http://main.py) \#główny program (tworzy maski i oblicza metryki)  
||-[analysis.py](http://analysis.py) \#funkcje do obliczania metryk (IoU, dice, precision, recall) i wykresy (PDF)  
||-[segmentation.py](http://segmentation.py) \#logika maskowania  
||-[processing.py](http://processing.py) \#czyszczenie masek, morfologia  
||-[visualization.py](http://visualization.py) \#wyświetlanie utworzonych masek  
||-data\_loader.py \#wczytanie ścieżek danych  
||-analyse\_data.py \#analiza danych przed tworzeniem masek  
|  
|-data/ \#dane wejściowe (nie w repozytorium)  
|  
|-results/ \#zapisywane PDF-y  
|  
|-requirements.txt  
|-[README.md](http://README.md)  
|-.gitignore

Instalacja danych wejściowych:  
Pobierz dane z linku i wstaw je do folderu data.  
[https://www.kaggle.com/datasets/meteahishali/aerial-imagery-for-standing-dead-tree-segmentation?resource=download](https://www.kaggle.com/datasets/meteahishali/aerial-imagery-for-standing-dead-tree-segmentation?resource=download)  
Projekt wymaga następującej struktury danych:  
|-data/  
||-USA\_segmentation/  
|||-RGB\_images/  
|||-NRG\_images/  
|||-masks/

Uruchamianie projektu:  
python src/analyse\_data.py  
python src/[main.py](http://main.py)

Po uruchomieniu analyse\_data.py:

1) Wczytanie ścieżek danych;  
2) Sprawdzenie czy foldery data, RGB, NRG, masks istnieją, oraz ile plików jest w folderach RGB, NRG, masks;  
3) Do folderu results zapisane zostaną:   
- data\_display.pdf (wyświetlenie danych wejściowych);  
- channel\_histograms.pdf (wyświetlenie histogramów poszczególnych kanałów RGB i HSV);  
- best\_channels.pdf (wyświetlenie histogramu pokazującego, który kanał najlepiej separuje martwe drzewa od reszty zdjęcia);  
4) Wypisane zostaną wyniki analizy, ile razy dany kanał był najleprzy w separacji martwych drzew od tła, dla danych wejściowych.

Po uruchomieniu [main.py](http://main.py):

1) Wczytanie ścieżek danych;  
2) Sprawdzenie czy foldery data, RGB, NRG, masks istnieją, oraz ile plików jest w folderach RGB, NRG, masks;  
3) Utworzenie masek;  
4)  Do folderu results zapisane zostaną:   
- final\_masks.pdf (wyświetlenie masek R/B/H i NRG i final mask \- po połączeniu masek, oraz maski ground truth \- z danych wejściowych);  
- iou\_histogram.pdf (wyświetlenie histogramu ilości obrazów o danej wartości IoU);  
- iou\_per\_image.pdf (wyświetlenie wykresu jaką wartość IoU miał każdy obrazek);  
5) Obliczenie metryk;  
6) Wyświetlenie tabelki z wartościami metryk;  
7) Wypisanie średnich wartości obliczonych metryk.

Wymagania:  
Zainstaluj requirements.txt:  
pip install \-r requirements.txt

Metryki użyte w projekcie:  
IoU \- Intersection over Union  
Dice Score  
Precision  
Recall  
TP / FP/ FN / TN