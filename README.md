# Projekt_ZPO
Projekt w ramach zajęć Zaawansowane Przetwarzanie Obrazów

Projekt polegał na stworzeniu modelu Bag of Visual Words na podstawie zdjeć popularnych budynków Poznania.

Wykorzystując odpowiedni detektor/deskryptor należało wyodrębnić unikalne cechy każdego z budynków a następnie podzielić je na odpowiednią ilość słów (grup) z wykorzystaniem funkcji KMeans. Następnie dobrano odpowiedni klasyfikator wraz z parametrami, który został wytrenowany na zbiorze uczącym z etykietami. Aby zastosować optymalne parametry klasyfikatora wykorzystano funkcję GridSearch.

Kod treningowy znajduje się w pliku train.py, natomiast kod testowy w main.py. Nie załączono do repozytorium danych testowych. 
