## Porównanie PyTorch vs TensorFlow
Przeprowadziliśmy wstępne testy mające na celu wybór optymalnego frameworka dla modelu, który docelowo zostanie zaimplementowany na Raspberry Pi. W projekcie korzystamy z wersji 8 GB pamięci RAM, która daje nam znaczący zapas technologiczny. Jednak ze względu na architekturę procesorów ARM, obok dokładności, jako główne kryterium oceny pozostaje nam szybkość wnioskowania mierzona w klatkach na sekundę (FPS). W celu rzetelnej oceny obu środowisk przetestowaliśmy dwie architektury sieci konwolucyjnych (CNN):
1. Płytką sieć (Shallow Network): Złożoną z niewielkiej liczby warstw. W tym scenariuszu zdecydowanym zwycięzcą okazał się PyTorch, który dzięki mniejszemu narzutowi systemowemu przy prostych operacjach wygenerował niemal dwukrotnie więcej FPS.
2. Głęboką sieć (Deep Network): Złożoną z wielu warstw splotowych i mechanizmów zapobiegających przeuczeniu (Dropout). Architektura ta jest absolutnie niezbędna, aby uzyskać zadowalającą dokładność. W tym przypadku TensorFlow zadziałał znacznie szybciej, udowadniając, że jego mechanizmy lepiej radzą sobie z optymalizacją ciężkich grafów obliczeniowych.

W związku z wynikami testów, zdecydowaliśmy się na ostateczne wykorzystanie środowiska TensorFlow. Wyższa wydajność tego frameworku w przypadku głębokich sieci jest dla naszego projektu kluczowa. Ponadto, wybrane przez nas Raspberry Pi nie powinno mieć problemów z obsługą tego ekosystemu, a sam wytrenowany model zostanie docelowo wyeksportowany do dedykowanego, lekkiego formatu TensorFlow Lite (TFLite). Pozwoli to na maksymalne odciążenie procesora i zagwarantuje płynną analizę obrazu z kamery w czasie rzeczywistym.

## Testy sieci neuronowej
Przeprowadzono testy jakościowe na zbiorach danych zaprezentowanych w tabeli nr 1.1. Działanie to pozwoliło na wyłonienie najlepiej zaanotowanego oraz najbardziej zróżnicowanego zbioru. Do testów wykorzystano 7-warstwową sieć neuronową (na rysunku 3.2 przedstawiono schemat modelu wykorzystanego do treningu), zaimplementowaną i wytrenowaną w środowisku TensorFlow. Całkowita liczba parametrów modelu wynosiła około 2,8 mln. Wszystkie procesy uczenia przeprowadzono w środowisku Google Colab ze względu na dostępność układów graficznych NVIDIA T4, charakteryzujących się wysoką mocą obliczeniową.

![Zdjęcie nr 3.1 - schemat blokowy modelu wykorzystanego do treningu](img/schemat_modelu.png)
*Zdjęcie nr 3.1 - schemat blokowy modelu wykorzystanego do treningu*

Na podstawie testów, których wyniki zestawiono w tabeli nr 3.2, oszacowano realistyczną skuteczność modelu na poziomie od 60% do 78%. Wartość tę uznano za zadowalającą. Zbiór NAVRASA, jako jedyny wykorzystywany do treningu w przestrzeni barw RGB, pozwolił na uzyskanie stosunkowo wysokiej dokładności wynoszącej 78%. Może to sugerować, że informacja o kolorze niesie ze sobą dodatkowe, istotne cechy ułatwiające klasyfikację. Najniższe wyniki odnotowano dla zbioru agregowanego (FER2013 oraz AffectNet) oraz Facial Affect Dataset Balanced. Zjawisko to wskazuje na znaczną trudność w ekstrakcji uniwersalnych cech, gdy dane uczące pochodzą z wysoce zróżnicowanych środowisk wizualnych.

Z kolei dla zbioru RAF-DB model osiągnął anomalnie wysoką dokładność na poziomie 91%. Po weryfikacji struktury tego zbioru zauważono, że część zawartych w nim obrazów powstała w wyniku augmentacji pozostałych zdjęć bazowych. Doprowadziło to do zjawiska wycieku danych (ang. data leakage). W takiej sytuacji sieć nie uczyła się rozpoznawania uniwersalnych cech określających emocje, lecz jedynie zapamiętywała specyficzne, powtarzające się obrazy.

**Tabela nr 3.2 - Wyniki przeprowadzonych treningów**
| Dataset | Liczba epok | BatchSize | Przestrzeń barw | Dokładność na zbiorze testowym |
|---|---|---|---|---|
| (fer2013+affectnet) dataset - Emotions | 20 | 32 | GrayScale | 64% |
| Balanced Affectnet Dataset (75×75, RGB) | 20 | 32 | GrayScale | 72% |
| Balanced RAF-DB Dataset (75×75) | 20 | 32 | GrayScale | 91% |
| Balanced NAVRASA Dataset (75×75, RGB) | 20 | 32 | RGB | 78% |
| Facial Affect Dataset Balanced | 20 | 32 | GrayScale | 60% |
