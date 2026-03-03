# Kamienie milowe – Uncanny Head

## Kamień milowy 1

- Wydruk 3D, obróbka oraz złożenie elementów konstrukcyjnych a także wszystkich niezbędnych elementów wykonawczych projektu głowy robota (Nilheim Mechatronics Robot Head 2.0).
- Zaprojektowanie, montaż fizyczny oraz walidacja sprzętowa układu zasilania, potwierdzona stabilnym wysterowaniem wszystkich podłączonych serwomechanizmów.
- Opracowanie i przetestowanie autorskiego algorytmu wymiany informacji, zapewniającego zapis rozpoznanej tożsamości oraz estymowanego stanu emocjonalnego użytkownika w ustandaryzowanym formacie w pliku JSON.
- Akwizycja, oczyszczenie i segmentacja publicznie dostępnych zbiorów danych (datasetów) oraz przeprowadzenie wstępnego, lokalnego treningu sieci neuronowych (autorska architektura w środowisku TensorFlow) do klasyfikacji twarzy i emocji, osiągających na tym etapie minimum 50% skuteczności walidacyjnej w środowisku izolowanym (PC).
- Zestawienie bazowego potoku konwersacyjnego (STT -> LLM -> TTS) zintegrowanego z dynamicznym odczytem kontekstu z plików JSON oraz informacji o salach i miejscach pracy nauczycieli akademickich na wydziale ETI. Rozszerzenie potoku o implementację modelu GLiNER (Named Entity Recognition) do skutecznej ekstrakcji encji. Na tym etapie możliwa będzie rozmowa w języku polskim z modelem uruchomionym lokalnie (laptop), co zostanie mierzalnie potwierdzone poprzez poprawne rozpoznanie intencji i wyciągnięcie encji (np. numer sali, imię i nazwisko profesora) dla minimum 80% z puli 20 przygotowanych zapytań testowych, a także skuteczne przeprowadzenie 10 ciągłych, testowych wymian zdań, w których czas wygenerowania adekwatnej odpowiedzi głosowej na zapytanie nie przekroczy 10 sekund.

## Kamień milowy 2

- Instalacja środowiska na NVIDIA Jetson Orin Nano w celu lokalnego hostowania modelu LLM (Gemma3 4B lub Ministral 3 3B) oraz na Raspberry Pi 5 8GB (obsługa detekcji i motoryki), co zostanie zweryfikowane poprzez podanie na wejście modelu LLM zamkniętego zestawu 5 predefiniowanych zapytań testowych i uzyskanie na nie merytorycznie spójnych odpowiedzi tekstowych. Dodatkowo, weryfikacja obejmie bezbłędne uruchomienie skryptów diagnostycznych na Raspberry Pi, potwierdzających poprawną inicjalizację strumienia wideo z kamery.
- Uruchomienie autorskiej sieci CNN bezpośrednio na Raspberry Pi 5 8GB. Osiągnięcie zakładanej skuteczności identyfikacji użytkowników oraz detekcji emocji na poziomie 60%-78%. Zestawienie stabilnej komunikacji sieciowej przekazującej dane telemetryczne (JSON) z RPi do Jetson Orin Nano.
- Uruchomienie pełnego łańcucha: od modułu Intent Detection, przez ekstrakcję parametrów modelem GLiNER, aż po odpytanie systemu RAG. System musi poprawnie rozróżniać intencje zapytań o pogodę w Gdańsku od zapytań o plan zajęć WETI albo innych zapytań i poprawnie odpowiadać, osiągając skuteczność poprawnej klasyfikacji intencji oraz ekstrakcji encji na poziomie minimum 85% weryfikowaną na zamkniętym zbiorze 30 różnorodnych zapytań testowych.
- Zaimplementowanie algorytmów sterujących mechaniką twarzy w celu wyrażenia 4 fizycznych stanów (radość, smutek, szok, neutralny) poprzez mimikę ust robota.
- Wdrożenie na platformie bufora pamięciowego o pojemności do 200 profili, uwzględniającego czas życia rekordu (TTL wynoszący 7 dni) oraz spełniającego założenia braku zapisu fizycznych zdjęć użytkowników.

## Kamień milowy 3

- Przeprowadzenie testów obciążeniowych modelu (Gemma3 4B / Ministral 3 3B) na Jetson Orin Nano, potwierdzających osiągnięcie parametrów - czas oczekiwania na odpowiedź głosową modelu zajmuje systemowi średnio maksymalnie 10 sekund.
- Udokumentowanie poprawnego działania systemu wyłącznie w wyznaczonych warunkach: środowisko ciche z natężeniem głosu powyżej 50 dB (odległość 1-1.5 m), oświetlenie biurowe/dzienne oraz całkowicie odsłonięta, frontalnie skierowana twarz.
- Przeprowadzenie próbnych konwersacji walidujących logikę bazy wiedzy. Zmierzenie skuteczności udzielania poprawnych informacji o salach i nauczycielach akademickich oraz pogodzie w Gdańsku na poziomie >85% (przy założeniu bezbłędnej pracy STT oraz łącza >10 Mb/s).
- Weryfikacja mechanizmu automatycznego resetowania pamięci profilu po 7 dniach. Potwierdzenie, że system aktywnie manifestuje rozpoznanie użytkownika, zwracając się do niego po imieniu w wygenerowanej wypowiedzi głosowej.
- Potwierdzenie, że w przypadku wykrycia określonych warunków konwersacyjnych, system poprawnie wdraża trzy dodatkowe stany emocjonalne (złość, strach, wstręt) modyfikując wyłącznie treść wypowiedzi, bez konieczności odzwierciedlania ich w mechanice ust.
- Sprawdzenie płynności działania systemu akwizycji obrazu i analizy. Testy zadeklarowanej dokładności działania modelu rozpoznawania twarzy i modelu rozpoznawania emocji. Znalezienie optymalnej (wg. kryteriów dokładności i szybkości) działania architektury.
- Pełna integracja programów sterujących mimiką z algorytmami napisanymi przez pozostałe zespoły.
- Zestrojenie serwomechanizmów szczęki z sygnałem audio. Osiągnięcie płynnego ruchu imitującego mowę z maksymalnym opóźnieniem synchronizacji nieprzekraczającym 250 ms.
