# Mettings Report
W celu finalnego połączenia wszystkich modułów w jeden gotowy i poprawnie działający system, spotkaliśmy się w sali koła naukowego CHIP. W spotkaniu uczestniczyła co najmniej jedna osoba z każdego zespołu projektowego (**vision, hardware, speech**).



# Metting at 13.05.2026 from 14:30 to 21:00
**Uczestnicy:** Wszyscy członkowie zespołu (obecność rotacyjna w trakcje trwania spotkania).
## Zrealizowane zadania
Podczas spotkania udało się zrealizować:

 - **Zgromadzono** w jednym miejscu wszystkie niezbędne elementy hardwearowe: kamerę, głośnik, mikrofon, czujnik FIR, Raspberry Pi, Jetson Orin Nano, zasilacze oraz wydrukowaną głowę robota z zamontowanymi serwomechanizmami.
 - **Zaprojektowane druki 3D:**
	 - Zaprojektowano i wydrukowano statyw do mikrofonu.
	 - Zaprojektowano i wydrukowano uchwyt do kamery.
- **Rozmieszczenie elementów:**
Ustalono docelowe położenie komponentów w obudowie oraz w otoczeniu robota. Kamera zostanie umieszczona na czole robota, natomiast mikrofon i głośnik znajdą się na zewnątrz jego podstawy. Z kolei Jetson, i Raspberry Pi wraz z zasilaczem zostaną umieszoczne wewnątrz podstawy robota.
- **Przeprowadzono testy mechaniki:**
prawdzono poprawne działanie poszczególnych serwomechanizmów za pomocą skryptu przygotowanego przez zespół hardwareowy.
-**Integracja skryptów programowych:**
Połączono skrypt odpowiedzialny za wykrywanie emocji z modułem śledzenia twarzy użytkownika (podążanie oczami robota za użytkownikiem).
-**Prace elektryczne:** Przedłużono zbyt krótkie przewody zasilacza.
-**Obwiednia sygnału:** Rozpoczęto prace nad przesyłaniem obwiedni sygnału audio (z modułu speech)  w celu symulowania ruchów ust robota podczas mowy.

## Napotkane problemy
W trakcie integracji wszystkich komponentów napotkano kilka problemów technicznych, z których większość udało się rozwiązać na miejscu.

 - **Problemy z połączeniem z RPI:** Wystąpiły trudności z połączeniem się z Raspberry Pi, co było spowodowane zakłóceniami – w pomieszczeniu uruchomionych było zbyt wiele hotspotów jednocześnie.
 - **Spadek liczby FPS'ów:** Mała liczba klatkek wynikała z błędnej konfiguracji modułu komunikacyjnego pomiędzy Raspberry Pi a Jetsonem. Po zmianie parametru _dropout_ z 50 ms na 200 ms problem ustąpił, a płynność wróciła do normy.
 - **Konfiguracja magistrali I2C:** Domyślne porty na Raspberry Pi nie wykrywały sterowników. Wymagało to przepięcia sygnałów SDA i SCL na inne, odpowiednie piny GPIO.
 - **Uszkodzenia wydrukuów:** Wydrukowany uchwyt kamery okazał się zbyt delikatny – podstawa pękła podczas dokręcania śruby. **TODO:** Element wymaga poprawy modelu i ponownego wydruku.

<img src="../images/unncany_head.jpg" alt="Figure 1 - Głowa robota (Problem komunikacji ze sterownikami)" width="50%">