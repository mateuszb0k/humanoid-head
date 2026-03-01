import json

def szukaj_zajec(sciezka_bazy, sciezka_info):
    with open(sciezka_bazy, 'r', encoding='utf-8') as f:
        baza = json.load(f)
        
    with open(sciezka_info, 'r', encoding='utf-8') as f:
        info = json.load(f)

    # 1. Pobranie parametrów 
    semestr = info.get("Semestr")
    kierunek = info.get("Kierunek")
    grupa = info.get("Grupa")
    podgrupa = info.get("Podgrupa")
    
    szukane_zajecia = info.get("Zajecia")
    szukany_typ = info.get("Typ")
    if szukany_typ:
        szukany_typ = szukany_typ.lower().replace("ł", "l")     # Raczej do pozbycia się - będziemy przekazywać dane już zmienione, by były bez polskich znaków
        
    szukana_godz_od = info.get("godzina_od")
    szukana_godz_do = info.get("godzina_do")

    wyniki = []

    # 2. odpowiednia grupa
    try:
        plan_grupy = baza[str(semestr)][kierunek]["Ogólna"][str(grupa)]
    except KeyError:
        return {"blad": "Nie znaleziono planu dla podanego semestru, kierunku lub grupy."}

    # 3. które listy zajęć sprawdzamy
    podgrupy_do_sprawdzenia = ["Cała grupa"]
    if podgrupa:
        podgrupy_do_sprawdzenia.append(podgrupa)

    # 4. Przeszukiwanie i ścisłe filtrowanie (AND)
    for podg in podgrupy_do_sprawdzenia:
        if podg in plan_grupy:
            for zajecia in plan_grupy[podg]:
                pasuje = True
                
                # Sprawdzamy po kolei warunki. 
                # Jeśli jakiś parametr został podany w info.json, ale NIE ZGADZA SIĘ z bazą -> odrzucamy (pasuje = False)
                
                if szukane_zajecia and zajecia.get("przedmiot") != szukane_zajecia:
                    pasuje = False
                if szukany_typ and zajecia.get("typ_zajec") != szukany_typ:
                    pasuje = False
                if szukana_godz_od and zajecia.get("godzina_od") != szukana_godz_od:
                    pasuje = False
                if szukana_godz_do and zajecia.get("godzina_do") != szukana_godz_do:
                    pasuje = False

                # Jeśli zajęcia przetrwały wszystkie powyższe testy (czyli wszystkie podane dane się zgadzają)
                if pasuje:
                    znalezione = zajecia.copy()
                    znalezione["dedykowane_dla"] = podg
                    
                    if znalezione not in wyniki:
                        wyniki.append(znalezione)

    # 5. Jeśli nic nie znaleziono (lista jest pusta), zwracamy błąd
    if not wyniki:
        return {"blad": "blad lub nie masz zajec"}

    return wyniki

wynik = szukaj_zajec('monday.json', 'info.json')
print(json.dumps(wynik, ensure_ascii=False, indent=2))